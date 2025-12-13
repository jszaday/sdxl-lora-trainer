# Maximizing GPU Utilization in LoRA Training

## Thesis
GPUs want **long‑lived, continuous work**. The dominant ML training stack instead feeds them **bursty, host‑driven kernel soup**. High utilization is not about raw FLOPs; it is about **eliminating idle gaps caused by Python, kernel boundaries, and CPU‑side control flow**.

The practical goal: keep the GPU resident, scheduled, and busy for as long as possible per step, with minimal host intervention.

---

## Core Failure Mode (What We’re Avoiding)

Typical training loops suffer from:

- Kernel-per-op execution → launch latency + sync edges
- Python-driven step logic → host stalls between iterations
- Optimizer steps on CPU → GPU idle during `.step()`
- Data prep / H2D copies serialized with compute
- Logging / sampling causing global synchronizations

Symptoms:

- Utilization oscillates (0% → 100% → 0%)
- Audible coil whine from power/clock cycling
- Lower throughput despite “fast” kernels

---

## Guiding Principles

1. **Amortize launch overhead**
   - Fewer, fatter kernels beat many tiny ones.

2. **Minimize host involvement per step**
   - Python should orchestrate *runs*, not *ops*.

3. **Make steps shape-static**
   - Fixed batch size, resolution, sequence length.
   - Static shapes unlock CUDA Graphs.

4. **Overlap everything that can overlap**
   - H2D copies must hide under compute.

5. **Prefer GPU-side work to CPU-side cleverness**
   - A “smart” Python optimizer that stalls the GPU is worse than a dumb fused one.

---

## Concrete Techniques

### 1. Precompute Non-Critical Work

Remove entire subsystems from the hot path:

- Precompute VAE latents
- Precompute text embeddings
- Fix padding/truncation offline

Result: the training loop becomes *pure tensor math*.

---

### 2. Static Shapes + CUDA Graphs (Pseudo GPU Residency)

CUDA Graphs are the closest thing to a GPU-resident training loop available today.

Approach:

- Warm up the model to stabilize allocations
- Allocate **static input/output tensors** on device
- Capture **forward + backward + optimizer step** into a CUDA Graph
- Per iteration:
  - Async copy next batch into static buffers
  - Replay graph

Effects:

- Kernel launch overhead amortized
- No Python between kernels
- Predictable, dense execution timeline
- GPU utilization becomes a flat bar

This effectively turns “one training step” into a single precompiled GPU program.

---

### 3. Overlap H2D With Compute

Never let the GPU wait for data:

- Use pinned memory in DataLoader
- Use a dedicated CUDA stream for H2D copies
- Double-buffer static input tensors if needed

Pattern:

- Step N compute runs
- Step N+1 data copies happen concurrently

If done correctly, H2D time disappears from the critical path.

---

### 4. Use Fused, GPU-Native Optimizers

Optimizer choice matters for utilization, not just convergence.

Good:
- AdamW / Lion with fused CUDA implementations

Bad:
- Python-only optimizers with per-parameter loops (e.g. Prodigy)

Why:

- Python optimizer `.step()` = GPU idle
- For LoRA, optimizer cost dominates relative to model compute

Rule:

> If `optimizer.step()` is visible in a profiler, it’s already too slow.

---

### 5. Separate Training From Sampling

Validation sampling is inherently disruptive.

Mitigation:

- Run sampling **infrequently** (`--sample_every`)
- Treat it as a controlled stall
- Keep the training hot loop pristine

Sampling should be an *intentional pause*, not accidental jitter every step.

---

### 6. Logging Without Sync

- Use `tqdm` for human feedback
- TensorBoard scalars are cheap; images are not
- Avoid per-step `.item()` calls where possible

If logging causes syncs, batch it or slow it down.

---

## Why This Works Especially Well for LoRA

LoRA characteristics:

- Small trainable parameter set
- Relatively cheap backward pass
- Optimizer overhead is disproportionately visible

Therefore:

- Host-side overhead hurts more
- Python optimizers hurt more
- CUDA Graphs help more

A clean LoRA pipeline is an ideal candidate for near-constant GPU occupancy.

---

## Mental Model

Think of training as:

> **Compile a GPU program and replay it many times**,
> not
> **Ask Python to micromanage thousands of tiny GPU tasks**.

Once you align the software model with the hardware model, the GPU stops screaming and starts purring.

---

## Bottom Line

To maximize GPU utilization:

- Delete unnecessary work from the hot path
- Make steps static and graphable
- Keep execution resident on the GPU
- Minimize Python between steps
- Accept fewer, intentional pauses instead of constant micro-stalls

Do this, and utilization stops oscillating, throughput improves, and coil whine disappears.
