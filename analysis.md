# Pre-Training Performance Analysis

This document focuses only on the pre-training path used by `mini_pre_train.sh` and `scripts/pre_train.py`. The goal is maximum training throughput, even if that means replacing parts of the stack with fused kernels or custom CUDA/Triton code.

## Scope

- Entrypoint: `mini_pre_train.sh:1`
- Training loop: `scripts/pre_train.py:134`, `scripts/pre_train.py:193`
- Model: `llm/gpt.py:7`, `llm/gpt.py:53`, `llm/gpt.py:175`
- Data path: `llm/dataloader.py:8`, `llm/dataloader.py:38`
- Optimizer: `llm/optimizer.py:3`, `llm/optimizer.py:41`
- Checkpointing: `llm/checkpoint_manager.py:4`

## Current Flow

`mini_pre_train.sh` launches a single-process run with a very small sequence length (`64`) and batch size (`8`). `scripts/pre_train.py` builds CPU-side dataloaders, copies each microbatch to device, runs bf16 autocast, backpropagates through a custom transformer stack, performs a custom Muon + Adam optimizer step, then runs validation every step. Every 1000 steps it saves a checkpoint and does autoregressive sampling.

In practice the hot path is:

1. Synchronous token loading on CPU.
2. Host-to-device transfer every microstep.
3. Transformer forward/backward.
4. Python-heavy optimizer step.
5. Extra validation forward pass every optimizer step.

## Highest-Impact Bottlenecks

### 1. Grouped-query attention is expanded back into full multi-head attention

Where:

- `llm/gpt.py:107`
- `llm/gpt.py:113`
- `llm/gpt.py:130`

Why it is slow:

- The code computes fewer KV heads, but then uses `repeat_interleave` to duplicate them across query heads before attention.
- That throws away most of the memory and bandwidth benefit of GQA.
- It increases tensor size right before the most expensive kernel in the model.

What to change:

- Replace this path with native GQA support in the attention kernel.
- If your PyTorch build supports it, use `scaled_dot_product_attention(..., enable_gqa=True)` or equivalent shape layout.
- Otherwise move to FlashAttention/xFormers with native grouped-query support.
- If neither fits, this is the best custom-kernel target in the repo.

Expected benefit:

- Large reduction in attention memory traffic.
- Better GPU occupancy and lower peak memory.
- More noticeable as sequence length and model width increase.

### 2. The data loader is synchronous and reloads entire shards on the training thread

Where:

- `llm/dataloader.py:8`
- `llm/dataloader.py:11`
- `llm/dataloader.py:35`
- `llm/dataloader.py:48`

Why it is slow:

- `np.load` pulls the full shard into memory synchronously.
- `torch.tensor(...)` makes an extra copy instead of sharing memory.
- When a shard ends, the next shard is loaded inline during training.
- No overlap exists between disk I/O, CPU staging, and GPU compute.

What to change:

- Use `np.load(..., mmap_mode="r")`.
- Replace `torch.tensor(npt, dtype=torch.long)` with `torch.from_numpy(...)` and cast only if truly needed.
- Prefetch the next shard in a background worker/thread.
- Produce pinned host tensors and use `non_blocking=True` during `.to(device)`.
- Longer term, replace this with a proper `torch.utils.data.DataLoader` pipeline with prefetching.

Expected benefit:

- Removes training stalls at shard boundaries.
- Reduces CPU copying overhead.
- Lets H2D transfer overlap with compute.

### 3. Device transfers are likely serialized with compute every microstep

Where:

- `scripts/pre_train.py:201`
- `scripts/pre_train.py:202`

Why it is slow:

- Batches are created on CPU and copied immediately before compute.
- There is no pinned memory and no async prefetch stream.
- On smaller batch sizes, copy latency can become a large share of step time.

What to change:

- Introduce a CUDA prefetcher with a dedicated stream.
- Make the loader yield pinned memory tensors.
- Copy with `x.to(device, non_blocking=True)` and `y.to(device, non_blocking=True)`.
- Double-buffer the next batch while the current one is executing.

Expected benefit:

- Less GPU idle time between microsteps.
- Better throughput on consumer GPUs like the RTX 3060.

### 4. Validation runs every optimizer step

Where:

- `scripts/pre_train.py:221`
- `scripts/pre_train.py:228`

Why it is slow:

- Every training step pays for an extra forward pass.
- With gradient accumulation, this is still a fixed overhead on top of each optimizer step.
- For throughput-oriented pre-training, this cadence is far too frequent.

What to change:

- Validate every `N` steps instead of every step.
- Start with every `100` or `500` steps.
- If you still want frequent feedback, log a training-only metric every step and run a short validation window on cadence.

Expected benefit:

- Immediate wall-clock speedup with almost no implementation risk.

### 5. The optimizer is Python-heavy and mostly unfused

Where:

- `llm/optimizer.py:66`
- `llm/optimizer.py:74`
- `llm/optimizer.py:87`

Why it is slow:

- Every parameter is processed in Python.
- Muon runs multiple Newton-Schulz iterations per tensor.
- Small tensor-wise update kernels are launch-heavy and hard for the GPU to execute efficiently.
- The non-Muon part does not use fused or foreach Adam-style updates.

What to change:

- For the Adam-like group, move to fused AdamW/foreach AdamW immediately.
- Flatten or bucket Muon-managed tensors by shape to reduce Python loops.
- If Muon is important to keep, move the update path into Triton or CUDA.
- Benchmark whether Muon actually beats fused AdamW in tokens/sec, not just loss quality.

Expected benefit:

- Lower optimizer step overhead.
- Fewer small kernels.
- Better scaling as model size grows.

### 6. Full logits are materialized before cross-entropy

Where:

- `llm/gpt.py:229`
- `llm/gpt.py:230`
- `scripts/pre_train.py:206`
- `scripts/pre_train.py:228`

Why it is slow:

- The model writes a full `[B, T, vocab_size]` tensor to memory.
- With `vocab_size=50304`, this is a large bandwidth cost.
- Then cross-entropy reads it again.

What to change:

- Use a fused linear + cross-entropy implementation.
- Candidate directions: Liger Kernel, Triton fused CE, or a custom fused vocab projection + CE path.
- This is one of the most valuable custom-kernel opportunities if you plan to keep a large vocab head.

Expected benefit:

- Lower memory bandwidth pressure.
- Lower peak activation memory.
- Better end-to-end step time, especially at larger vocab sizes.

## Medium-Impact Opportunities

### 7. RMSNorm is custom and unfused

Where:

- `llm/gpt.py:14`
- `llm/gpt.py:17`

Why it is slow:

- It performs several separate elementwise and reduction passes.
- It uses custom Python module code instead of optimized fused kernels.

What to change:

- Replace with `torch.nn.RMSNorm` or `torch.nn.functional.rms_norm`.
- Best case: use a fused Triton/CUDA RMSNorm kernel.

Note:

- The module currently includes both `g` and `b`; standard RMSNorm often omits bias, so preserve behavior deliberately if you swap it.

### 8. Rotary embedding application creates extra tensor traffic

Where:

- `llm/gpt.py:41`
- `llm/gpt.py:49`

Why it is slow:

- Slicing, stacking, and flattening create extra tensor movement.
- RoPE runs on both `q` and `k` every block, every step.

What to change:

- Use a fused rotary kernel.
- Alternatively implement a packed-half or complex-number formulation that avoids `stack(...).flatten(...)`.
- Fold RoPE into the same fused path as Q/K normalization if you write a custom attention pre-processing kernel.

### 9. Q/K normalization adds extra reductions before attention

Where:

- `llm/gpt.py:120`
- `llm/gpt.py:127`

Why it is slow:

- Two norms, two divisions, and more temporary tensors are created per block.
- This is not the dominant cost, but it sits directly on the attention critical path.

What to change:

- Use `F.normalize` if it maps better to fused kernels.
- Better: fold normalization into a fused Q/K preprocessing kernel.
- Also validate whether the normalization improves quality enough to justify the overhead.

### 10. MLP uses exact GELU instead of a faster approximation or fused MLP

Where:

- `llm/gpt.py:142`

What to change:

- Use `nn.GELU(approximate="tanh")`.
- If you pursue custom kernels, fused bias+GELU+linear is a useful secondary target.

### 11. `zero_grad()` can avoid writing zeros

Where:

- `scripts/pre_train.py:195`

What to change:

- Use `optimizer.zero_grad(set_to_none=True)` if compatible with the optimizer implementation.

Expected benefit:

- Small but effectively free improvement.

### 12. NVIDIA math fast paths are not explicitly enabled

Where:

- `scripts/pre_train.py` setup section around `device` and model construction.

What to change:

- Enable TF32 where acceptable:
  - `torch.set_float32_matmul_precision("high")`
  - `torch.backends.cuda.matmul.allow_tf32 = True`
  - `torch.backends.cudnn.allow_tf32 = True`
- Keep bf16 autocast on supported hardware.

Expected benefit:

- Moderate throughput win for projection and MLP GEMMs on NVIDIA.

## Low-Impact but Worth Cleaning Up

### 13. Checkpoint saves are synchronous and delete old checkpoints first

Where:

- `llm/checkpoint_manager.py:9`
- `llm/checkpoint_manager.py:18`

Why it is slow:

- Training blocks on disk write.
- The code deletes the old checkpoint before writing the new one, which also increases pause time and risk.

What to change:

- Save asynchronously in a background thread/process.
- Write to a temp file and atomically rename.
- Consider keeping rolling checkpoints instead of deleting first.

### 14. Sampling during training recomputes the full prefix each generated token

Where:

- `scripts/pre_train.py:265`
- `scripts/pre_train.py:271`

Why it is slow:

- The model reruns attention over the entire prompt each token.
- No KV cache is used.

What to change:

- Either run this far less often, or add a KV cache for the sampling-only path.
- This does not help training throughput directly, but it removes periodic pauses.

## Structural Improvements For Bigger Gains

### 15. Sequence length is too small to efficiently use the GPU

Where:

- `mini_pre_train.sh:1`

Why it matters:

- `--seq_length 64` is tiny for transformer training on CUDA.
- Kernel launch overhead, synchronization, and host-device transfer costs become disproportionately large.
- The GPU spends more time on overhead and less time doing dense math.

What to change:

- Increase sequence length as much as memory allows.
- Then retune batch size and gradient accumulation around the new memory envelope.
- A better throughput target is usually to maximize total tokens per optimizer step while keeping kernels large enough to saturate the GPU.

### 16. Single-process training leaves scale-up options unused

Where:

- Entire pre-training path currently uses one process and one device.

What to change:

- If you plan to train on multi-GPU machines later, move to DDP/FSDP before deeper kernel work.
- For single 3060 setups this is not relevant, but for larger systems this quickly becomes mandatory.

## Best Custom-Kernel Targets

If the goal is absolute maximum speed, these are the most justified places to write custom kernels:

1. Native grouped-query flash attention without K/V expansion.
2. Fused LM head + cross-entropy.
3. Fused RMSNorm.
4. Fused RoPE + Q/K normalization pre-processing.
5. Triton/CUDA Muon optimizer update.

If you only build one custom kernel, build the GQA attention kernel first.

## Recommended Priority Order

### Fastest wins with low implementation risk

1. Reduce validation frequency.
2. Enable pinned-memory + non-blocking batch transfer.
3. Use memory-mapped shard loading and `torch.from_numpy`.
4. Enable TF32 fast paths on NVIDIA.
5. Switch exact GELU to approximate GELU.
6. Use `zero_grad(set_to_none=True)`.

### Next tier of meaningful engineering work

1. Replace the custom loader with a prefetching `DataLoader` pipeline.
2. Swap custom RMSNorm for an optimized implementation.
3. Benchmark fused AdamW against the current mixed optimizer.
4. Move validation/checkpoint/sampling off the hot path.

### Highest upside, highest complexity

1. Remove K/V expansion and use native GQA attention.
2. Fuse vocab projection with cross-entropy.
3. Fuse Muon updates or replace Muon if it loses on tokens/sec.

## How To Verify Improvements

Measure performance with actual throughput metrics, not just intuition.

### Core metrics

- Tokens/sec
- Optimizer steps/sec
- Time spent in data loading, H2D copy, forward, backward, optimizer, validation
- Peak CUDA memory
- GPU utilization over time

### Recommended tooling

- `torch.profiler` with CUDA activities
- Nsight Systems for idle-gap and transfer analysis
- Nsight Compute for the final custom-kernel tuning pass

### What to look for

- Fewer or smaller idle gaps between kernels after prefetching
- Lower attention memory traffic after removing KV expansion
- Lower optimizer time after fusing or simplifying updates
- Lower memory use and fewer giant logits writes after fused CE

## Bottom Line

The biggest current speed losses are not subtle. The pre-training stack is paying for synchronous shard loading, synchronous device copies, validation every step, Python-heavy optimizer updates, and a grouped-query attention implementation that expands K/V back to full size before the attention kernel. Fixing those issues will matter much more than small cleanup changes.

If the target is maximum speed, the best long-term path is:

1. Fix the data pipeline and validation cadence.
2. Replace K/V expansion with native GQA attention.
3. Fuse the vocab projection and loss.
4. Reevaluate whether Muon is worth its optimizer overhead.
