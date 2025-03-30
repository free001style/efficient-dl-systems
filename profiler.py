import json
import time
import torch
from torch.cuda import Event
import os
from dataclasses import dataclass


@dataclass
class Schedule:
    wait: int = 1
    warmup: int = 1
    active: int = 5


class Profile:
    """
    Simple profiler that exports data in the perfetto trace event format.
    Interface similar to PyTorch Profiler.
    """
    def __init__(self, model, name="model", schedule: Schedule = None):
        self.name_map = self._build_name_map(model, name)
        self.events = []
        self.times = {}
        self.handles = []
        self.wait = schedule.wait
        self.warmup = schedule.warmup
        self.active = schedule.active
        self.step_ = 0

    def _build_name_map(self, model, name="model"):
        name_map = {}
        for full_name, module in model.named_modules():
            if full_name == "":
                full_name = name

            if self._is_leaf(module):
                name_map[module] = module.__class__.__name__
            else:
                name_map[module] = f"{full_name}: {module.__class__.__name__}"

        return name_map

    def _is_leaf(self, module):
        return len(list(module.children())) == 0

    def _forward_pre_hook(self, module, inputs):
        if self.step_ >= self.wait:
            torch.cuda.synchronize()
            self.times[self.name_map[module]] = {"ts": time.time() * 1e6, }
            module.forward_start = Event(enable_timing=True)
            module.forward_start.record()

    def _forward_post_hook(self, module, inputs, outputs):
        if self.step_ >= self.wait:
            # https://discuss.pytorch.org/t/how-to-measure-time-in-pytorch/26964
            module.forward_end = Event(enable_timing=True)
            module.forward_end.record()
            torch.cuda.synchronize()
            dur = module.forward_start.elapsed_time(module.forward_end)
            if (self.active + self.wait + self.warmup) > self.step_ >= (self.wait + self.warmup):
                self.events.append({
                    "ph": "X",
                    "name": f"{self.name_map[module]}",
                    "ts": self.times[self.name_map[module]]["ts"],
                    "pid": os.getpid(),
                    "dur": dur * 1e3,
                    "arg": {
                        "Input Dims": [list(input.shape) for input in inputs],
                        "Output Dims": [list(output.shape) for output in outputs],
                    }
                })

    def _backward_pre_hook(self, module, grad_output):
        if self.step_ >= self.wait:
            torch.cuda.synchronize()
            self.times[self.name_map[module]] = {"ts": time.time() * 1e6, }
            module.backward_start = Event(enable_timing=True)
            module.backward_start.record()

    def _backward_post_hook(self, module, grad_input, grad_output):
        if self.step_ >= self.wait:
            module.backward_end = Event(enable_timing=True)
            module.backward_end.record()
            torch.cuda.synchronize()
            dur = module.backward_start.elapsed_time(module.backward_end)
            if (self.active + self.wait + self.warmup) > self.step_ >= (self.wait + self.warmup):
                self.events.append({
                    "ph": "X",
                    "name": f"{self.name_map[module]}_backward",
                    "pid": os.getpid(),
                    "ts": self.times[self.name_map[module]]["ts"],
                    "dur": dur * 1e3
                })

    def __enter__(self):
        for module in self.name_map.keys():
            self.handles.append(module.register_forward_pre_hook(self._forward_pre_hook))
            self.handles.append(module.register_forward_hook(self._forward_post_hook))
            self.handles.append(module.register_full_backward_pre_hook(self._backward_pre_hook))
            self.handles.append(module.register_full_backward_hook(self._backward_post_hook))
        return self

    def __exit__(self, type, value, traceback):
        for handle in self.handles:
            handle.remove()

    def step(self):
        self.step_ += 1

    def summary(self):
        print("Summary:")
        for event in self.events:
            print(event)

    def to_perfetto(self, path="trace.json"):
        with open(path, "w") as f:
            json.dump({"traceEvents": self.events}, f, indent=4)


def profile(model, name="model", schedule: Schedule = None):
    return Profile(model, name, schedule)
