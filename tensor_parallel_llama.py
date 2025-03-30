import torch
from torch import nn
import transformers
from transformers.models.llama.modeling_llama import LlamaConfig, \
    LlamaAttention, LlamaMLP, LlamaRotaryEmbedding, LlamaRMSNorm, \
    LlamaModel, LlamaPreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPast
from typing import Optional, Tuple
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, DeviceMesh, Replicate, Shard
import torch.distributed.tensor.parallel as tp


class ComputeWithAllReduce(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tp_shard: nn.Module, input: torch.Tensor, kwargs):
        input = input.detach().requires_grad_(input.requires_grad)
        ctx.save_for_backward(input)
        ctx._kwargs = kwargs
        ctx._tp_shard = tp_shard
        if kwargs is not None:
            output = tp_shard(input, **kwargs)
        else:
            output = tp_shard(input)
        dist.all_reduce(output[0])
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor, mask):
        with torch.enable_grad():
            if ctx._kwargs is not None:
                output = ctx._tp_shard(ctx.saved_tensors[0], **ctx._kwargs)[0]
            else:
                output = ctx._tp_shard(ctx.saved_tensors[0])
            output.backward(grad_output)
        dist.all_reduce(ctx.saved_tensors[0].grad)
        return None, ctx.saved_tensors[0].grad, None


class AllReduceModule(nn.Module):
    def __init__(self, tp_module):
        super().__init__()
        self.tp_module = tp_module

    def forward(self, input: torch.Tensor, kwargs=None):
        return ComputeWithAllReduce.apply(self.tp_module, input, kwargs)


class MyLlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = AllReduceModule(LlamaAttention(config=config,
                                                        layer_idx=layer_idx))

        self.mlp = AllReduceModule(LlamaMLP(config))
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, 
                                            eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, 
                                                     eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(hidden_states, {
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_value": None,
            "output_attentions": output_attentions,
            "use_cache": use_cache,
            "cache_position": cache_position,
            "position_embeddings": position_embeddings},
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs


class MyLlama(LlamaPreTrainedModel):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, 
                                         config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [MyLlamaDecoderLayer(config, layer_idx) 
             for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.lm_head = nn.Linear(config.hidden_size, 
                                 config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = LlamaModel._update_causal_mask(
            self=self,
            attention_mask=attention_mask,
            input_tensor=inputs_embeds,
            cache_position=cache_position,
            past_key_values=None,
            output_attentions=output_attentions
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        output = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        ).to_tuple()

        hidden_states = output[0]
        logits = self.lm_head(hidden_states[:, -0:, :])
        return logits


def tensor_parallel_llama(rank, world_size, device):
    for active_rank in range(world_size):
        dist.barrier()  # initialize each rank sequentially to save system RAM
        if rank != active_rank: continue

        MODEL_NAME = "unsloth/Llama-3.2-1B"
        model = transformers.LlamaForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32).to(device)

        config = LlamaConfig.from_pretrained(MODEL_NAME)
        config.num_attention_heads //= world_size
        config.num_key_value_heads //= world_size
        config.intermediate_size //= world_size
        tp_module = MyLlama(config).to(device)

        with torch.no_grad():
            # copy select weights
            tp_module.embed_tokens.load_state_dict(model.model.embed_tokens.state_dict())
            tp_module.norm.load_state_dict(model.model.norm.state_dict())

            for i in range(len(tp_module.layers)):

                q_units = tp_module.layers[i].self_attn.tp_module.q_proj.out_features
                q_slice = slice(q_units * rank, q_units * (rank + 1))
                tp_module.layers[i].self_attn.tp_module.q_proj.weight[...] = model.model.layers[i].self_attn.q_proj.weight[q_slice]
                tp_module.layers[i].self_attn.tp_module.o_proj.weight[...] = model.model.layers[i].self_attn.o_proj.weight[:, q_slice]

                kv_units = tp_module.layers[i].self_attn.tp_module.k_proj.out_features
                kv_slice = slice(kv_units * rank, kv_units * (rank + 1))
                tp_module.layers[i].self_attn.tp_module.k_proj.weight[...] = model.model.layers[i].self_attn.k_proj.weight[kv_slice]
                tp_module.layers[i].self_attn.tp_module.v_proj.weight[...] = model.model.layers[i].self_attn.v_proj.weight[kv_slice]

                mlp_units = tp_module.layers[i].mlp[0].down_proj.in_features
                mlp_slice = slice(mlp_units * rank, mlp_units * (rank + 1))
                tp_module.layers[i].mlp[0].up_proj.weight[...] = model.model.layers[i].mlp.up_proj.weight[mlp_slice]
                tp_module.layers[i].mlp[0].gate_proj.weight[...] = model.model.layers[i].mlp.gate_proj.weight[mlp_slice]
                tp_module.layers[i].mlp[0].down_proj.weight[...] = model.model.layers[i].mlp.down_proj.weight[:, mlp_slice]

                tp_module.layers[i].input_layernorm.load_state_dict(model.model.layers[i].input_layernorm.state_dict())
                tp_module.layers[i].post_attention_layernorm.load_state_dict(model.model.layers[i].post_attention_layernorm.state_dict())

            tp_module.lm_head.load_state_dict(model.lm_head.state_dict())

        del model  # free RAM for next rank
    return tp_module


def dtensor_tensor_parallel_llama(rank, world_size, device):
    device_mesh = init_device_mesh(device_type="cuda", mesh_shape=(world_size,))

    MODEL_NAME = "unsloth/Llama-3.2-1B"
    model = transformers.LlamaForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32).to("cuda")

    parallelize_plan = {}
    for i in range(len(model.model.layers)):
        parallelize_plan[f"model.layers.{i}.self_attn.q_proj"] = tp.ColwiseParallel()
        parallelize_plan[f"model.layers.{i}.self_attn.k_proj"] = tp.ColwiseParallel()
        parallelize_plan[f"model.layers.{i}.self_attn.v_proj"] = tp.ColwiseParallel()
        parallelize_plan[f"model.layers.{i}.self_attn.o_proj"] = tp.RowwiseParallel()

        parallelize_plan[f"model.layers.{i}.mlp.up_proj"] = tp.ColwiseParallel()
        parallelize_plan[f"model.layers.{i}.mlp.gate_proj"] = tp.ColwiseParallel()
        parallelize_plan[f"model.layers.{i}.mlp.down_proj"] = tp.RowwiseParallel()

    tp_module = tp.parallelize_module(
        model,
        device_mesh,
        parallelize_plan=parallelize_plan,
    )
    return tp_module
