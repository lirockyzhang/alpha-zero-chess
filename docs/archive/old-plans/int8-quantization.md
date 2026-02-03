# INT8 Quantization for AlphaZero Inference

## Overview

INT8 quantization can potentially speed up inference on consumer GPUs by reducing memory bandwidth requirements and enabling faster integer arithmetic. However, for AlphaZero chess, the benefits are **nuanced** and depend on several factors.

## Should You Use INT8 Quantization?

### ✅ Potential Benefits

1. **Memory Bandwidth Reduction**
   - INT8 uses 4× less memory than FP32 (1 byte vs 4 bytes)
   - Reduces GPU memory usage from ~43MB to ~11MB for the network
   - Faster data transfer between GPU memory and compute units

2. **Theoretical Speedup**
   - Modern GPUs (Turing+, Ampere, Ada) have dedicated INT8 Tensor Cores
   - Can achieve 2-4× throughput compared to FP32 on paper
   - Lower power consumption

3. **Batch Processing Benefits**
   - Larger batches fit in GPU memory
   - Better for batched inference server with many actors

### ⚠️ Practical Limitations for AlphaZero

1. **Bottleneck Analysis**
   - **Current bottleneck**: Python game state operations (`apply_action`, `get_observation`)
   - **Not bottleneck**: Neural network inference (only ~10-20% of total time)
   - Speeding up inference by 2× only improves overall performance by ~5-10%

2. **Accuracy Concerns**
   - AlphaZero requires high precision for policy and value predictions
   - INT8 quantization introduces approximation errors
   - May degrade playing strength, especially in critical positions
   - Requires careful calibration and validation

3. **Implementation Complexity**
   - Requires PyTorch quantization API
   - Need calibration dataset (representative positions)
   - Post-training quantization (PTQ) vs Quantization-aware training (QAT)
   - Additional testing and validation overhead

4. **Hardware Requirements**
   - Significant speedup requires Tensor Cores (RTX 20xx+, GTX 16xx+)
   - Older GPUs (GTX 10xx) may see minimal benefit
   - CPU inference sees little to no benefit

## Recommendation

### For Training (Self-Play Actors)
**❌ Not Recommended**

- Bottleneck is game state operations, not inference
- Complexity outweighs minimal performance gain
- Risk of degrading training data quality

### For Evaluation/Playing
**✅ Worth Considering**

- Inference is more critical during interactive play
- Can use FP32 for training, INT8 for deployment
- Acceptable if playing strength is validated

### For Batched Inference Server
**⚠️ Maybe**

- If you have 8+ actors and GPU memory is constrained
- Allows larger batch sizes
- But test playing strength carefully

## Implementation Guide (If You Decide to Proceed)

### Option 1: Post-Training Quantization (PTQ) - Easier

```python
import torch
from alphazero.neural.network import AlphaZeroNetwork

# Load trained model
network = AlphaZeroNetwork(num_filters=192, num_blocks=15)
network.load_state_dict(torch.load('checkpoint.pt')['network_state_dict'])
network.eval()

# Prepare calibration data (representative positions)
from alphazero.chess_env import GameState
from alphazero.training.replay_buffer import ReplayBuffer

# Load some positions from replay buffer
buffer = ReplayBuffer(capacity=1000)
# ... load trajectories ...
calibration_data = []
for i in range(100):
    obs, mask, _, _ = buffer.sample_numpy(1)
    calibration_data.append((
        torch.from_numpy(obs).float(),
        torch.from_numpy(mask).float()
    ))

# Dynamic quantization (easiest, but limited speedup)
quantized_network = torch.quantization.quantize_dynamic(
    network,
    {torch.nn.Linear, torch.nn.Conv2d},
    dtype=torch.qint8
)

# Static quantization (better performance, requires calibration)
network.qconfig = torch.quantization.get_default_qconfig('fbgemm')
torch.quantization.prepare(network, inplace=True)

# Calibrate with representative data
with torch.no_grad():
    for obs, mask in calibration_data:
        network.predict(obs, mask)

torch.quantization.convert(network, inplace=True)

# Save quantized model
torch.save(quantized_network.state_dict(), 'checkpoint_int8.pt')
```

### Option 2: Quantization-Aware Training (QAT) - Better Accuracy

```python
import torch
from alphazero.neural.network import AlphaZeroNetwork

# Create network with quantization stubs
network = AlphaZeroNetwork(num_filters=192, num_blocks=15)
network.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')

# Prepare for QAT
torch.quantization.prepare_qat(network, inplace=True)

# Train normally (quantization is simulated during training)
# ... training loop ...

# Convert to actual INT8 after training
network.eval()
torch.quantization.convert(network, inplace=True)
```

### Integration with Inference Server

```python
# In inference_server.py
class InferenceServer(Process):
    def __init__(
        self,
        # ... existing params ...
        use_int8: bool = False,
        int8_checkpoint: Optional[str] = None
    ):
        self.use_int8 = use_int8
        self.int8_checkpoint = int8_checkpoint
        # ...

    def run(self):
        # Create network
        network = self.network_class(**self.network_kwargs)

        if self.use_int8 and self.int8_checkpoint:
            # Load INT8 quantized model
            network.load_state_dict(torch.load(self.int8_checkpoint))
            logger.info("Loaded INT8 quantized model")
        else:
            # Load FP32 model
            if self.initial_weights:
                network.load_state_dict(self.initial_weights)

        network = network.to(self.device)
        network.eval()
        # ...
```

## Validation Strategy

If you implement INT8 quantization, you **must** validate playing strength:

1. **Self-Play Evaluation**
   ```bash
   # Play 100 games: FP32 vs INT8
   uv run python scripts/evaluate.py \
       --checkpoint1 checkpoint_fp32.pt \
       --checkpoint2 checkpoint_int8.pt \
       --games 100
   ```

2. **Stockfish Benchmark**
   ```bash
   # Test both against Stockfish at various levels
   uv run python scripts/evaluate.py \
       --checkpoint checkpoint_int8.pt \
       --opponent stockfish \
       --stockfish-elo 1500 \
       --games 50
   ```

3. **Acceptance Criteria**
   - INT8 model should win ≥45% against FP32 model
   - Elo difference should be <50 points
   - No catastrophic failures in tactical positions

## Performance Expectations

### Realistic Speedup Estimates

| Component | Time % | INT8 Speedup | Overall Impact |
|-----------|--------|--------------|----------------|
| Game state ops | 70% | 1.0× | 0% |
| Neural inference | 20% | 2.0× | +10% |
| MCTS tree ops | 10% | 1.0× | 0% |
| **Total** | 100% | - | **~10% faster** |

### Memory Savings

| Configuration | FP32 Memory | INT8 Memory | Savings |
|---------------|-------------|-------------|---------|
| Network weights | 43 MB | 11 MB | 32 MB |
| Batch (32) activations | ~200 MB | ~50 MB | 150 MB |
| **Total per actor** | ~250 MB | ~65 MB | **~185 MB** |

**Benefit:** Can run more actors or larger batches with same GPU memory.

## Alternative Optimizations (Better ROI)

Before implementing INT8, consider these alternatives:

### 1. **Optimize Game State Operations** (Biggest Impact)
- Implement Cython/C++ for `apply_action` and `get_observation`
- Cache board encodings
- Use more efficient data structures
- **Expected speedup: 2-3×** (much better than INT8)

### 2. **Optimize MCTS** (Already Done)
- ✅ You already have Cython and C++ backends
- These provide ~1.2× speedup (limited by Python game state)

### 3. **Increase Batch Size** (Free Performance)
- Larger batches improve GPU utilization
- Try batch_size=64 or 128 in inference server
- No code changes needed, just config

### 4. **Use Mixed Precision (FP16)** (Better Trade-off)
- Already supported via `use_amp=True` in config
- 2× memory reduction, ~1.5× speedup
- Better accuracy than INT8
- Simpler implementation

## Conclusion

### TL;DR

**For most users: ❌ Don't use INT8 quantization**

Reasons:
1. Inference is not the bottleneck (~20% of time)
2. 2× inference speedup = only ~10% overall speedup
3. Risk of accuracy degradation
4. Implementation complexity

**Better alternatives:**
1. Optimize game state operations (2-3× speedup)
2. Use FP16/AMP (already supported, safer than INT8)
3. Increase batch size (free performance)

**When INT8 makes sense:**
- You have 16+ actors and GPU memory is constrained
- You've already optimized everything else
- You're willing to validate playing strength carefully
- You're deploying for inference-only (not training)

## References

- [PyTorch Quantization Documentation](https://pytorch.org/docs/stable/quantization.html)
- [NVIDIA INT8 Inference](https://developer.nvidia.com/blog/int8-inference-autonomous-vehicles-tensorrt/)
- [Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference](https://arxiv.org/abs/1712.05877)
