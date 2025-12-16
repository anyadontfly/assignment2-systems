# benchmark_optimized_ddp.py
# -------------------------------------------------------------
# CSE 599O
#
# Extend your DDP benchmark to evaluate three optimized variants
# for the Transformer model:
#   (1) run_flat       
#   (2) run_individual 
#   (3) run_bucketed   
#
# The TA will execute your script using commands like:
#     srun --gpus-per-node=2 uv run benchmark_optimized_ddp.py --mode flat
#     srun --gpus-per-node=2 uv run benchmark_optimized_ddp.py --mode individual
#     srun --gpus-per-node=2 uv run benchmark_optimized_ddp.py --mode bucketed --bucket-mb 10
#
# Each function should measure and print out the following statistics:
#   - iteration time per step  → append to iteration_times
#   - communication time per step → append to comm_times
# -------------------------------------------------------------
import os
import argparse
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.optim import AdamW
from cse599o_systems.benchmark_naive_ddp import run_timing_naive_ddp_worker
from cse599o_systems.distributed_data_parallel import DDP, BucketedDDP
from cse599o_basics.model import Transformer
from cse599o_basics.utils import cross_entropy_loss
# Any other necessary imports can be added here.

# Any necessary helper functions can be defined here.
def run_timing_flat_ddp_worker(rank, world_size, data, targets, num_steps, num_warmup, result_queue):
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
        loss = cross_entropy_loss(output.view(-1, 50257), local_targets.view(-1))
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push("backward")
        loss.backward()
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push("all_reduce")
        with torch.no_grad():
            grad_list = []
            for p in model.parameters():
                if p.grad is not None:
                    grad_list.append(p.grad.view(-1))
            
            if grad_list:
                flat_grad = torch.cat(grad_list)
                if rank == 0:
                    comm_start_events[i].record()
                dist.all_reduce(flat_grad, op=dist.ReduceOp.SUM)
                if rank == 0:
                    comm_end_events[i].record()
                flat_grad /= world_size
                
                offset = 0
                for p in model.parameters():
                    if p.grad is not None:
                        numel = p.grad.numel()
                        p.grad.copy_(flat_grad[offset:offset + numel].view_as(p.grad))
                        offset += numel
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push("optimizer_step")
        optim.step()
        torch.cuda.nvtx.range_pop()
        if rank == 0:
            train_end_events[i].record()
        if grad_list:
            del flat_grad

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

def run_timing_ddp_worker(rank, world_size, data, targets, num_steps, num_warmup, result_queue, ddp_cls, bucket_size_mb=None):
    """Run one DDP worker process."""
    setup(rank, world_size)
    torch.manual_seed(42)

    assert data.shape == targets.shape, "Data and targets must have the same shape"

    data_per_rank = data.size(0) // world_size
    start_idx = rank * data_per_rank
    end_idx = start_idx + data_per_rank
    local_data = data[start_idx:end_idx].to(rank)
    local_targets = targets[start_idx:end_idx].to(rank)

    model = Transformer(
        d_model=1280,
        num_heads=20,
        d_ff=2560,
        vocab_size=50257,
        context_length=128,
        num_layers=36,
        rope_theta=10000,
    )
    if bucket_size_mb is not None:
        ddp_model = ddp_cls(model, bucket_size_mb=bucket_size_mb, timing=True)
    else:
        ddp_model = ddp_cls(model, timing=True)
    optim = AdamW(ddp_model.parameters())
    model.train()

    if rank == 0:
        train_start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_warmup + num_steps)]
        train_end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_warmup + num_steps)]
        comm_events = []
    
    for i in range(num_warmup + num_steps):
        if rank == 0:
            train_start_events[i].record()
        optim.zero_grad()
        torch.cuda.nvtx.range_push("forward")
        output = ddp_model(local_data)
        loss = cross_entropy_loss(output.view(-1, 50257), local_targets.view(-1))
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push("backward")
        loss.backward()
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push("sync_allreduce")
        if rank == 0:
            comm_events.append(ddp_model.finish_gradient_synchronization())
        else:
            ddp_model.finish_gradient_synchronization()
        torch.cuda.nvtx.range_pop()
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
            for start_event, end_event in comm_events
        ]
        result_queue.put({
            "iter_times": iter_times,
            "comm_times": comm_times,
        })

    cleanup()

# You can change the function and variable names as needed.
def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

# ============================================================
# (0) Naive DDP
# ============================================================
# You can change the function and variable names as needed.
def run_naive(inputs, targets, num_iters, num_warmup, iteration_times, comm_times):
    """A naive DDP training loop for reference."""
    world_size = 2

    mp.set_start_method("spawn", force=True)
    manager = mp.Manager()
    result_queue = manager.Queue()

    mp.spawn(
        run_timing_naive_ddp_worker,
        args=(world_size, inputs, targets, num_iters, num_warmup, result_queue),
        nprocs=world_size,
        join=True,
    )

    timing_data = result_queue.get()
    iteration_times.extend(timing_data["iter_times"][num_warmup:])
    comm_times.extend(timing_data["comm_times"][num_warmup:])

# ============================================================
# (1) Flat DDP
# ============================================================
# You can change the function and variable names as needed.
def run_flat(inputs, targets, num_iters, num_warmup, iteration_times, comm_times):
    """All-reduce a single flattened gradient tensor."""
    world_size = 2

    mp.set_start_method("spawn", force=True)
    manager = mp.Manager()
    result_queue = manager.Queue()

    mp.spawn(
        run_timing_flat_ddp_worker,
        args=(world_size, inputs, targets, num_iters, num_warmup, result_queue),
        nprocs=world_size,
        join=True,
    )

    timing_data = result_queue.get()
    iteration_times.extend(timing_data["iter_times"][num_warmup:])
    comm_times.extend(timing_data["comm_times"][num_warmup:])


# ============================================================
# (2) Individual DDP
# ============================================================
# You can change the function and variable names as needed.
def run_individual(inputs, targets, num_iters, num_warmup, iteration_times, comm_times):
    """All-reduce each parameter's gradient individually."""
    world_size = 2

    mp.set_start_method("spawn", force=True)
    manager = mp.Manager()
    result_queue = manager.Queue()

    mp.spawn(
        run_timing_ddp_worker,
        args=(world_size, inputs, targets, num_iters, num_warmup, result_queue, DDP),
        nprocs=world_size,
        join=True,
    )

    timing_data = result_queue.get()
    iteration_times.extend(timing_data["iter_times"][num_warmup:])
    comm_times.extend(timing_data["comm_times"][num_warmup:])


# ============================================================
# (3) Bucketed DDP
# ============================================================
# You can change the function and variable names as needed.
def run_bucketed(inputs, targets, num_iters, num_warmup, iteration_times, comm_times, bucket_mb):
    """Group gradients into buckets and all-reduce each bucket."""
    world_size = 2

    mp.set_start_method("spawn", force=True)
    manager = mp.Manager()
    result_queue = manager.Queue()

    mp.spawn(
        run_timing_ddp_worker,
        args=(world_size, inputs, targets, num_iters, num_warmup, result_queue, BucketedDDP, bucket_mb),
        nprocs=world_size,
        join=True,
    )

    timing_data = result_queue.get()
    iteration_times.extend(timing_data["iter_times"][num_warmup:])
    comm_times.extend(timing_data["comm_times"][num_warmup:])


# ============================================================
# Benchmark Function
# ============================================================
# You can change the function and variable names as needed.
def benchmark_optimized_ddp():
    """Benchmark DDP variants on the Transformer model."""
    parser = argparse.ArgumentParser(description="Benchmark optimized DDP variants.")
    parser.add_argument(
        "--mode",
        type=str,
        default="flat",
        choices=["naive", "flat", "individual", "bucketed"],
        help="Select which DDP variant to benchmark.",
    )
    parser.add_argument(
        "--bucket-mb",
        type=int,
        default=10,
        help="Bucket size (in MB) for the bucketed DDP variant.",
    )
    args = parser.parse_args()

    # Example placeholders
    num_iters, num_warmup = 5, 10
    iteration_times, comm_times = [], []

    vocab_size = 50257
    batch_size = 1
    seq_length = 128

    inputs = torch.randint(0, vocab_size, (batch_size, seq_length))
    targets = torch.randint(0, vocab_size, (batch_size, seq_length))

    if args.mode == "naive":
        run_naive(inputs, targets, num_iters, num_warmup, iteration_times, comm_times)
    elif args.mode == "flat":
        run_flat(inputs, targets, num_iters, num_warmup, iteration_times, comm_times)
    elif args.mode == "individual":
        run_individual(inputs, targets, num_iters, num_warmup, iteration_times, comm_times)
    elif args.mode == "bucketed":
        run_bucketed(inputs, targets, num_iters, num_warmup, iteration_times, comm_times, args.bucket_mb)

    print(f"Mode: {args.mode}")
    print(f"Iteration times: {iteration_times}")
    print(f"Communication times: {comm_times}")
    print(f"Fraction of communication time: {[c/i for c, i in zip(comm_times, iteration_times)]}")


if __name__ == "__main__":
    benchmark_optimized_ddp()
