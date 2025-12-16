# benchmark_naive_ddp.py
# -------------------------------------------------------------
# CSE 599O: Distributed Training Basics
#
# Implement a naive DDP version that reproduces the same model
# state as single-process training.
#
# The TA will test your implementation with the following commands:
#
# 1. To verify that DDP matches baseline (toy model):
#     srun --gpus-per-node=2 uv run benchmark_naive_ddp.py --model toy
# Expected output: "Naive DDP matches baseline!"
#
# 2. To output communication and step time (transformer model):
#     srun --gpus-per-node=2 uv run benchmark_naive_ddp.py --model transformer
# Expected output: communication and step time statistics
#
# -------------------------------------------------------------

# Any necessary imports can be added here.
import os
import argparse

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.optim import AdamW
import torch.multiprocessing as mp
from cse599o_basics.utils import cross_entropy_loss
from cse599o_basics.model import Transformer
# from ..tests.common import ToyModel

class _FC2(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 50, bias=True)
        self.fc.bias.requires_grad = False

    def forward(self, x):
        x = self.fc(x)
        return x

class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 10, bias=False)
        self.fc2 = _FC2()
        self.fc3 = nn.Linear(50, 5, bias=False)
        self.relu = nn.ReLU()
        self.no_grad_fixed_param = nn.Parameter(torch.tensor([2.0, 2.0]), requires_grad=False)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Any necessary helper functions can be defined here.
def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    torch.set_default_dtype(torch.float32)

def cleanup():
    dist.destroy_process_group()

# You can change the function and variable names as needed.
def run_naive_ddp_worker(rank, world_size, data, num_steps, result_queue):
    """Run one DDP worker process."""
    setup(rank, world_size)
    torch.manual_seed(42)

    data_per_rank = data.size(0) // world_size
    start_idx = rank * data_per_rank
    end_idx = start_idx + data_per_rank
    local_data = data[start_idx:end_idx].to(rank)

    if rank == 0:
        model = ToyModel().to(rank)
        optim = AdamW(model.parameters())
        model.train()
        objects = [model.state_dict()]
    else:
        objects = [None]
    dist.broadcast_object_list(objects, src=0)

    if rank != 0:
        model = ToyModel().to(rank)
        model.load_state_dict(objects[0])
        optim = AdamW(model.parameters())
        model.train()

    for _ in range(num_steps):
        optim.zero_grad()
        output = model(local_data)
        loss = output.mean()
        loss.backward()
        with torch.no_grad():
            for p in model.parameters():
                if p.grad is not None:
                    dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                    p.grad /= world_size
        optim.step()

    torch.cuda.synchronize()

    if rank == 0:
        cpu_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        result_queue.put(cpu_state)

    cleanup()

# You can change the function and variable names as needed.
def run_baseline(data, num_steps):
    """Run single-process baseline for comparison."""
    torch.manual_seed(42)
    model = ToyModel()
    optim = AdamW(model.parameters())
    model.train()
    for _ in range(num_steps):
        optim.zero_grad()
        output = model(data)
        loss = output.mean()
        loss.backward()
        optim.step()
    torch.cuda.synchronize()
    return {k: v.detach().cpu() for k, v in model.state_dict().items()}

# You can change the function and variable names as needed.
def verify_naive_ddp():
    """Benchmark and verify naive DDP."""
    world_size = 2
    num_steps = 5
    data = torch.randn(10, 10)

    # Run baseline
    no_ddp_state = run_baseline(data, num_steps)

    # Set up multiprocessing for DDP
    mp.set_start_method("spawn", force=True)
    manager = mp.Manager()
    result_queue = manager.Queue()

    mp.spawn(
        run_naive_ddp_worker,
        args=(world_size, data, num_steps, result_queue),
        nprocs=world_size,
        join=True,
    )

    # Get model state from DDP
    ddp_state = result_queue.get()
    
    assert len(no_ddp_state) > 0, "model state from baseline is empty"
    for name in no_ddp_state:
        assert torch.allclose(no_ddp_state[name], ddp_state[name], atol=1e-6)
    print("Naive DDP matches baseline!")

# You can change the function and variable names as needed.
def run_timing_naive_ddp_worker(rank, world_size, data, targets, num_steps, num_warmup, result_queue):
    """Run one DDP worker process."""
    setup(rank, world_size)
    torch.manual_seed(42)

    assert data.shape == targets.shape, "Data and targets must have the same shape"

    data_per_rank = data.size(0) // world_size
    start_idx = rank * data_per_rank
    end_idx = start_idx + data_per_rank
    local_data = data[start_idx:end_idx].to(rank)
    local_targets = targets[start_idx:end_idx].to(rank)

    if rank == 0:
        print("Initializing Transformer model...")
        model = Transformer(
            d_model=1280,
            num_heads=20,
            d_ff=2560,
            vocab_size=50257,
            context_length=128,
            num_layers=36,
            rope_theta=10000,
        )
        model.train()
        objects = [model.state_dict()]
    else:
        objects = [None]
    dist.broadcast_object_list(objects, src=0)

    if rank != 0:
        model = Transformer(
            d_model=1280,
            num_heads=20,
            d_ff=2560,
            vocab_size=50257,
            context_length=128,
            num_layers=36,
            rope_theta=10000,
        )
        model.load_state_dict(objects[0])
    model.to(rank)
    optim = AdamW(model.parameters())
    model.train()

    if rank == 0:
        train_start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_warmup + num_steps)]
        train_end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_warmup + num_steps)]
        comm_start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_warmup + num_steps)]
        comm_end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_warmup + num_steps)]
    for i in range(num_warmup + num_steps):
        if rank == 0:
            train_start_events[i].record()
        optim.zero_grad()
        torch.cuda.nvtx.range_push("forward")
        output = model(local_data)
        torch.cuda.nvtx.range_pop()
        loss = cross_entropy_loss(output.view(-1, 50257), local_targets.view(-1))
        torch.cuda.nvtx.range_push("backward")
        loss.backward()
        torch.cuda.nvtx.range_pop()
        if rank == 0:
            comm_start_events[i].record()
        with torch.no_grad():
            for p in model.parameters():
                if p.grad is not None:
                    dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                    p.grad /= world_size
        if rank == 0:
            comm_end_events[i].record()
        torch.cuda.nvtx.range_push("optimizer_step")
        optim.step()
        torch.cuda.nvtx.range_pop()
        if rank == 0:
            train_end_events[i].record()

        torch.cuda.synchronize()

    if rank == 0:
        iter_times = [
            start_event.elapsed_time(end_event)
            for start_event, end_event in zip(train_start_events, train_end_events)
        ]
        comm_times = [
            start_event.elapsed_time(end_event)
            for start_event, end_event in zip(comm_start_events, comm_end_events)
        ]
        result_queue.put({
            "iter_times": iter_times,
            "comm_times": comm_times,
        })

    cleanup()
  
# You can change the function and variable names as needed.  
def timing_naive_ddp():
    """Timing benchmark for naive DDP with transformer model."""
    world_size = 2
    num_steps = 5
    num_warmup = 3
    batch_size = 1
    seq_length = 128
    data = torch.randint(0, 50257, (batch_size, seq_length))
    targets = torch.randint(0, 50257, (batch_size, seq_length))

    # Set up multiprocessing for DDP
    mp.set_start_method("spawn", force=True)
    manager = mp.Manager()
    result_queue = manager.Queue()

    mp.spawn(
        run_timing_naive_ddp_worker,
        args=(world_size, data, targets, num_steps, num_warmup, result_queue),
        nprocs=world_size,
        join=True,
    )

    timing_data = result_queue.get()
    for i in range(num_steps):
        iter_time = timing_data["iter_times"][i + num_warmup]
        comm_time = timing_data["comm_times"][i + num_warmup]
        fraction = comm_time / iter_time
        print(f"Step {i}: train time={iter_time:.3f} ms, comm time={comm_time:.3f} ms, fraction={fraction:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["toy", "transformer"], default="toy")
    args = parser.parse_args()

    if args.model == "toy":
        verify_naive_ddp()
    elif args.model == "transformer":
        timing_naive_ddp()