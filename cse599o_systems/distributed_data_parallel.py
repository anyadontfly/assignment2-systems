from dataclasses import dataclass
from typing import Any, List
import torch
import torch.distributed as dist
import torch.nn as nn


class DDP(nn.Module):
    def __init__(self, model: nn.Module, timing: bool=False):
        super().__init__()
        self.module = model
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.pendings = []
        self.timing = timing

        if self.rank == 0:
            objects = [model.state_dict()]
        else:
            objects = [None]
        dist.broadcast_object_list(objects, src=0)

        if self.rank != 0:
            model.load_state_dict(objects[0])
        if torch.cuda.device_count() >= self.world_size:
            model.to(self.rank)

        # register backward hooks
        seen_params = set()
        for param in self.module.parameters():
            if param in seen_params:
                continue
            seen_params.add(param)
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(self._post_backward_hook)

    def _post_backward_hook(self, param: torch.Tensor) -> None:
        self.pendings.append((dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, async_op=True), param))

    def forward(self, *inputs, **kwargs):
        if self.pendings:
            self.finish_gradient_synchronization()
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        if self.timing and self.rank == 0:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        for handle, param in self.pendings:
            handle.wait()
            param.grad /= self.world_size
        if self.timing and self.rank == 0:
            end_event.record()
        self.pendings.clear()
        if self.timing and self.rank == 0:
            return start_event, end_event

@dataclass
class Bucket:
    params: List[torch.Tensor]
    num_ready: int
    size_bytes: int
    flat_grads: torch.Tensor = None

class BucketedDDP(nn.Module):
    def __init__(self, model: nn.Module, bucket_size_mb: float, timing: bool=False):
        super().__init__()
        self.module = model
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.pending_comms = []
        self.is_bucketed = False
        self.buckets = []
        self.bucket_size_bytes = int(bucket_size_mb * 1024 * 1024)

        self.num_bucketed_params = 0
        
        self.timing = timing

        if self.rank == 0:
            objects = [model.state_dict()]
        else:
            objects = [None]
        dist.broadcast_object_list(objects, src=0)

        if self.rank != 0:
            model.load_state_dict(objects[0])
        if torch.cuda.device_count() >= self.world_size:
            model.to(self.rank)

        # register backward hooks
        seen_params = set()
        for param in self.module.parameters():
            if param in seen_params:
                continue
            seen_params.add(param)
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(self._add_bucket)
        self.distinct_params = len(seen_params)

    def _add_bucket(self, param: torch.Tensor) -> None:
        if not self.is_bucketed:
            if not self.buckets or (self.buckets[-1].size_bytes + param.numel() * param.element_size() > self.bucket_size_bytes):
                new_bucket = Bucket(params=[], num_ready=0, size_bytes=0)
                self.buckets.append(new_bucket)
            self.buckets[-1].params.append(param)
            self.buckets[-1].size_bytes += param.numel() * param.element_size()
            self.num_bucketed_params += 1
            self.pending_comms.append((dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, async_op=True), param))
        else:
            bucket_idx = None
            for idx, bucket in enumerate(self.buckets):
                if any(param is p for p in bucket.params):
                    bucket_idx = idx
                    break

            if bucket_idx is not None:
                bucket = self.buckets[bucket_idx]
                bucket.num_ready += 1
                
                if bucket.num_ready == len(bucket.params):
                    flat_grads = torch.cat([p.grad.flatten() for p in bucket.params])
                    handle = dist.all_reduce(flat_grads, op=dist.ReduceOp.SUM, async_op=True)
                    bucket.flat_grads = flat_grads
                    self.pending_comms.append(handle)

    def forward(self, *inputs, **kwargs):
        if self.pending_comms:
            self.finish_gradient_synchronization()
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        if self.is_bucketed:
            if self.timing and self.rank == 0:
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
            for handle in self.pending_comms:
                handle.wait()
            if self.timing and self.rank == 0:
                end_event.record()
            for bucket in self.buckets:
                if bucket.flat_grads is not None:
                    bucket.flat_grads /= self.world_size
                    offset = 0
                    for param in bucket.params:
                        numel = param.numel()
                        param.grad.copy_(bucket.flat_grads[offset:offset + numel].view_as(param))
                        offset += numel
                    bucket.num_ready = 0
                    bucket.flat_grads = None
            self.pending_comms.clear()
            if self.timing and self.rank == 0:
                return start_event, end_event
        else:
            if self.timing and self.rank == 0:
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
            for handle, param in self.pending_comms:
                handle.wait()
                param.grad /= self.world_size
            if self.timing and self.rank == 0:
                end_event.record()
            self.pending_comms.clear()
            self.is_bucketed = True
            if self.timing and self.rank == 0:
                return start_event, end_event


class ShardedOptimizer(torch.optim.Optimizer):
    def __init__(self, params, optimizer_cls: type, **kwargs):
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self._shard_to_param = {}
        self.wrapped_optim = optimizer_cls([{"params": []}], **kwargs)
        super().__init__(params, {})

    def _rank_slice(self, n: int):
        shard_size = n // self.world_size
        rem = n % self.world_size
        if self.rank < rem:
            s = self.rank * (shard_size + 1)
            e = s + shard_size + 1
        else:
            s = rem * (shard_size + 1) + (self.rank - rem) * shard_size
            e = s + shard_size
        return s, e

    def add_param_group(self, param_group: dict[str, Any]):
        local_param_group = {"params": []}
        for param in param_group["params"]:
            s, e = self._rank_slice(param.shape[0])
            view = param.data.narrow(0, s, e - s)
            shard_param = torch.nn.Parameter(view, requires_grad=param.requires_grad)

            local_param_group["params"].append(shard_param)
            self._shard_to_param[shard_param] = (param, s, e)

        self.wrapped_optim.add_param_group(local_param_group)

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.wrapped_optim.param_groups:
            for shard in group["params"]:
                full_param, s, e = self._shard_to_param[shard]
                if not shard.requires_grad or full_param.grad is None:
                    shard.grad = None
                    continue
                else:
                    shard.grad = full_param.grad[s:e].detach()

        ret = self.wrapped_optim.step(closure=closure)

        for group in self.wrapped_optim.param_groups:
            seen = set()
            for shard in group["params"]:
                full_param, s, e = self._shard_to_param[shard]
                if full_param in seen:
                    continue
                seen.add(full_param)

                # for i in range(self.world_size):
                #     if i == self.rank:
                #         objects = [shard, s, e]
                #     else:
                #         objects = [None, None, None]
                #     dist.broadcast_object_list(objects, src=i)
                #     _shard, _s, _e = objects
                #     full_param[_s:_e].data.copy_(_shard.data)

                full_param_data = full_param.data
                ar_params = torch.zeros_like(full_param_data)
                ar_params[s:e].copy_(full_param_data[s:e])
                dist.all_reduce(ar_params, op=dist.ReduceOp.SUM)
                full_param_data.copy_(ar_params)
        return ret

    def zero_grad(self, set_to_none: bool = False):
        for p in self._shard_to_param:
            if p.grad is not None:
                if set_to_none:
                    p.grad = None
                else:
                    p.grad.zero_()
        for full_param, _, _ in self._shard_to_param.values():
            if full_param.grad is not None:
                if set_to_none:
                    full_param.grad = None
                else:
                    full_param.grad.zero_()
        self.wrapped_optim.zero_grad(set_to_none=set_to_none)
