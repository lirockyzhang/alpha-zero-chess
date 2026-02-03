# PyTorch Integration with C++ Backend

## Overview

This document describes the optimal pattern for integrating the C++ MCTS backend with PyTorch neural networks, focusing on the `channels_last` memory format optimization for maximum GPU performance.

---

## The Challenge

The C++ encoder outputs position tensors in **NHWC format** (batch, height, width, channels), which is the natural layout for row-major memory access. However, PyTorch's Conv2d layers expect **NCHW format** (batch, channels, height, width) by default.

The naive approach of permuting dimensions and transferring to GPU can leave performance on the table. The optimal approach uses PyTorch's `channels_last` memory format to achieve maximum GPU throughput.

---

## The Optimal Pattern: "Input Gate" Conversion

### Code Pattern

```python
# At model initialization (done once)
model = AlphaZeroNetwork(num_filters=64, num_blocks=5)
model = model.to(device)
model = model.to(memory_format=torch.channels_last)

# In inference loop (per batch)
obs_batch = alphazero_cpp.collect_leaves()  # C++ outputs NHWC: (N, 8, 8, 119)

# Convert at the "input gate"
obs_tensor = torch.from_numpy(obs_batch) \
    .permute(0, 3, 1, 2) \
    .contiguous(memory_format=torch.channels_last) \
    .to(device)

# Forward pass - no format conversion needed!
with torch.no_grad():
    policy_logits, values = model(obs_tensor)
```

### What Each Step Does

1. **`torch.from_numpy(obs_batch)`**
   - Creates PyTorch tensor from C++ numpy array
   - Shape: (N, 8, 8, 119) - NHWC logical and physical
   - Zero-copy operation (shares memory with numpy)

2. **`.permute(0, 3, 1, 2)`**
   - Changes logical shape to NCHW: (N, 119, 8, 8)
   - Creates a view with different strides
   - **No data movement** - cheap operation

3. **`.contiguous(memory_format=torch.channels_last)`** ⚡ **Key step!**
   - Physically reorganizes memory to NHWC layout
   - Creates truly contiguous tensor optimized for Tensor Cores
   - Ensures data matches what GPU kernels expect
   - **This is where the performance magic happens**

4. **`.to(device)`**
   - Transfers optimized tensor to GPU
   - Preserves channels_last memory layout
   - Single efficient transfer

---

## Why This Works

### Memory Layout Details

**Before optimization:**
```
C++ Output (NHWC): [N][H][W][C] physical memory
    ↓ naive permute + transfer
GPU: [N][C][H][W] NCHW physical memory
    ↓ Conv2d operation
GPU has to reorganize memory for Tensor Cores (overhead!)
```

**After optimization:**
```
C++ Output (NHWC): [N][H][W][C] physical memory
    ↓ permute (view only)
Logical shape: (N, C, H, W) but physical: [N][H][W][C]
    ↓ contiguous(channels_last)
Physical memory: [N][H][W][C] properly organized
    ↓ transfer to GPU
GPU: [N][H][W][C] physical, (N,C,H,W) logical
    ↓ Conv2d with channels_last
Tensor Cores directly use optimized layout (fast!)
```

### Tensor Core Optimization

NVIDIA Tensor Cores are optimized for NHWC memory layout because:
- Better cache locality for convolution operations
- Coalesced memory accesses in inner loop
- Reduced memory bandwidth requirements
- Optimal access patterns for warp-level operations

By ensuring data is already in NHWC layout, we eliminate GPU-side reorganization overhead and maximize Tensor Core utilization.

---

## Performance Impact

### Benchmark Results (RTX 4060, 800 simulations)

| Approach | Sims/sec | NN Evals/sec | Improvement |
|----------|----------|--------------|-------------|
| Baseline (no optimization) | 2,072 | - | - |
| First channels_last attempt | 3,298 | - | +59% |
| Permute + to(device) | 3,563 | 9,535 | +72% |
| **Complete input gate** | **6,192** | **10,292** | **+199%** 🎯 |

The complete "input gate" pattern achieves **3x performance improvement** over the baseline!

### Multiple Search Consistency

Testing 5 games with 400 simulations each:
- Mean: 10,009 sims/sec
- Best: 11,995 sims/sec
- Std dev: ±1,827 sims/sec
- Consistent high performance

---

## Common Mistakes to Avoid

### ❌ Mistake 1: No channels_last on model

```python
# Bad: Model not converted to channels_last
model = AlphaZeroNetwork().to(device)
obs_tensor = torch.from_numpy(obs).permute(0,3,1,2).to(device)
```

**Problem**: Data is in NCHW format, Conv2d has to reorganize for Tensor Cores

### ❌ Mistake 2: No contiguous() call

```python
# Bad: Missing .contiguous(memory_format=torch.channels_last)
obs_tensor = torch.from_numpy(obs).permute(0,3,1,2).to(device)
```

**Problem**: Permute creates strided view, data not truly contiguous for GPU

### ❌ Mistake 3: Calling contiguous() after to(device)

```python
# Bad: Order matters!
obs_tensor = torch.from_numpy(obs).permute(0,3,1,2) \
    .to(device) \
    .contiguous(memory_format=torch.channels_last)
```

**Problem**: GPU transfer happens before optimization, may trigger additional copies

### ✅ Correct Pattern

```python
# Good: Complete "input gate" pattern
model = model.to(memory_format=torch.channels_last)  # Once at init

obs_tensor = torch.from_numpy(obs).permute(0,3,1,2) \
    .contiguous(memory_format=torch.channels_last) \
    .to(device)
```

---

## Implementation Checklist

- [ ] Convert model to `channels_last` format at initialization
- [ ] Use `torch.from_numpy()` for zero-copy tensor creation
- [ ] Apply `.permute(0, 3, 1, 2)` to change logical shape to NCHW
- [ ] Apply `.contiguous(memory_format=torch.channels_last)` to optimize memory layout
- [ ] Transfer to device with `.to(device)`
- [ ] Verify no format conversions in model's `forward()` method
- [ ] Test with real trained model and benchmark performance

---

## Model Architecture Notes

The `AlphaZeroNetwork` should:
1. Accept NCHW-shaped tensors: `(batch, 119, 8, 8)`
2. Be converted to `channels_last` format during initialization
3. Have a clean `forward()` method with no format conversions

```python
class AlphaZeroNetwork(nn.Module):
    """AlphaZero neural network optimized for channels_last format.

    Input tensors should be NCHW shape (batch, 119, 8, 8) with
    channels_last memory layout for optimal Conv2d performance.
    """

    def forward(self, x: torch.Tensor, legal_mask: Optional[torch.Tensor] = None):
        # No format conversion - data is already optimized!
        x = self.input_conv(x)
        x = self.residual_tower(x)

        policy_logits = self.policy_head(x)
        value = self.value_head(x)

        return policy_logits, value
```

---

## Testing Your Implementation

Use this test to verify optimal performance:

```python
import torch
import numpy as np
from alphazero.neural.network import AlphaZeroNetwork
import alphazero_cpp

# Setup
device = torch.device('cuda')
model = AlphaZeroNetwork(num_filters=64, num_blocks=5)
model = model.to(device)
model = model.to(memory_format=torch.channels_last)
model.eval()

# Generate test data
obs_batch = np.random.randn(256, 8, 8, 119).astype(np.float32)

# Convert using optimal pattern
obs_tensor = torch.from_numpy(obs_batch) \
    .permute(0, 3, 1, 2) \
    .contiguous(memory_format=torch.channels_last) \
    .to(device)

# Benchmark
import time
times = []
for _ in range(100):
    start = time.time()
    with torch.no_grad():
        policy, value = model(obs_tensor)
    torch.cuda.synchronize()
    times.append(time.time() - start)

print(f"Mean inference time: {np.mean(times)*1000:.2f}ms")
print(f"Throughput: {256/np.mean(times):.0f} positions/sec")
```

Expected performance on RTX 4060: >10,000 positions/sec

---

## References

- PyTorch channels_last documentation: https://pytorch.org/tutorials/intermediate/memory_format_tutorial.html
- NVIDIA Tensor Core documentation: https://docs.nvidia.com/deeplearning/performance/
- AlphaZero paper: https://arxiv.org/abs/1712.01815

---

## Summary

The "input gate" optimization pattern:
1. Converts model to channels_last once at initialization
2. Prepares tensors with proper memory layout before GPU transfer
3. Eliminates all format conversions in forward pass
4. Achieves 3x performance improvement with proper Tensor Core utilization

This is the recommended approach for all C++ backend + PyTorch integration in AlphaZero.
