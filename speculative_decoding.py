import torch


@torch.no_grad()
def speculative_generate(big_model, small_model, prefix: str,
                         max_num_tokens: int, n: int,
                         tokenizer, device) -> tuple[str, int, int]:
    """Vanilla speculative decoding for greedy generation.

    Args:
        big_model: original big HF model (verifier).
        small_model: small HF model (draft).
        prefix (str): prompt.
        max_num_tokens (int): max tokens to generate.
        n (int): number of tokens to speculate.

    Returns: generated text, number of accepted tokens, number of all tokens.
    """

    input_ids = tokenizer(prefix, return_tensors="pt").input_ids.to(device)
    start_size = input_ids.size(1)
    cnt_accepted = 0
    cnt_all = 0
    while input_ids.size(1) - start_size < max_num_tokens:
        small_generation = small_model.generate(input_ids, max_new_tokens=n)
        num_generated_tokens = small_generation.size(1) - input_ids.size(1)
        if num_generated_tokens == 0:
            break

        big_model_logits = big_model(small_generation).logits
        big_model_generations = torch.argmax(big_model_logits[:, -(num_generated_tokens + 1) : -1, :], -1)

        mismatch = False
        small_model_generations = small_generation[:, input_ids.size(1) : input_ids.size(1) + num_generated_tokens]
        for i in range(num_generated_tokens):
            if small_model_generations[:, i] != big_model_generations[:, i]:
                mismatch = True
                if i == 0:
                    input_ids = torch.cat(
                        (input_ids, big_model_generations[:, :1]),
                        dim=1
                    )
                else:
                    input_ids = torch.cat(
                        (input_ids, small_model_generations[:, :i], big_model_generations[:, i: i + 1]),
                        dim=1
                    )
                print(f"Accepted {i}/{n} tokens")
                cnt_accepted += i
                cnt_all += n
                break

        if not mismatch:
            next_token = torch.argmax(big_model_logits[:, -1], -1)[:, None]
            input_ids = torch.cat((small_generation, next_token), dim=1)
    decoded_text = tokenizer.batch_decode(input_ids[:, start_size:start_size + max_num_tokens], skip_special_tokens=True)[0]
    return decoded_text, cnt_accepted, cnt_all
