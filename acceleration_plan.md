 AlphaZero Training Acceleration Plan

 Goal: Achieve 5x+ training speedup while maintaining training quality (800 simulations)

 Executive Summary

 The current training system is bottlenecked by:
 1. Inference batching - GPU severely underutilized (batch_size=32, timeout=1ms)
 2. Actor parallelism - Only 4 actors generating games
 3. MCTS backend - Python backend is 5-10x slower than Cython

 Expected combined speedup: 5-8x (without reducing simulations)

 ---
 Hardware Profiles

 Profile: HIGH (A100 / Colab A100)

 PROFILE_HIGH = {
     'actors': 24,
     'inference_batch_size': 512,
     'inference_timeout': 0.02,  # 20ms
     'training_batch_size': 8192,
     'mcts_backend': 'cython',
 }

 Profile: MID (T4 / V100)

 PROFILE_MID = {
     'actors': 12,
     'inference_batch_size': 256,
     'inference_timeout': 0.015,  # 15ms
     'training_batch_size': 4096,
     'mcts_backend': 'cython',
 }

 Profile: LOW (RTX 4060 Laptop / 8GB VRAM)

 PROFILE_LOW = {
     'actors': 6,
     'inference_batch_size': 128,
     'inference_timeout': 0.01,  # 10ms
     'training_batch_size': 2048,
     'mcts_backend': 'python',  # Cython may not be available
 }

 ---
 Optimization 1: Increase Inference Batch Size and Timeout (3-4x speedup)

 Problem: Current batch_size=32 and batch_timeout=1ms severely underutilize GPU.

 Files to modify:
 - alphazero/selfplay/inference_server.py (lines 68-69)
 - alphazero/config.py (add InferenceConfig)

 Changes:
 # inference_server.py line 47-80
 class InferenceServer:
     def __init__(
         self,
         request_queue: mp.Queue,
         response_queues: Dict[int, mp.Queue],
         weight_queue: mp.Queue,
         network: nn.Module,
         device: str = "cuda",
         batch_size: int = 256,      # Was 32 - CHANGE
         batch_timeout: float = 0.015, # Was 0.001 - CHANGE
         use_amp: bool = True,
     ):

 Expected speedup: 3-4x for inference throughput

 ---
 Optimization 2: Increase Number of Actors (1.5-2x speedup)

 Problem: Only 4 actors generating games. More actors = better batch utilization.

 Files to modify:
 - scripts/train.py (add --profile argument)

 Changes:
 # Add hardware profile selection
 parser.add_argument("--profile", choices=['high', 'mid', 'low', 'auto'],
                     default='auto', help="Hardware profile for training")

 Expected speedup: 1.5-2x (combined with Optimization 1)

 ---
 Optimization 3: Use Cython MCTS Backend (2-3x speedup)

 Problem: Python MCTS is slow. Cython backend is 5-10x faster.

 Files to modify:
 - alphazero/mcts/cython/setup.py - Ensure builds correctly
 - scripts/train.py - Auto-detect and use Cython if available

 Changes:
 # Auto-detect best available backend
 def get_best_mcts_backend():
     try:
         from alphazero.mcts.cython import CythonMCTS
         return 'cython'
     except ImportError:
         try:
             from alphazero.mcts.cpp import CppMCTS
             return 'cpp'
         except ImportError:
             return 'python'

 Expected speedup: 2-3x for MCTS operations

 ---
 Optimization 4: Pinned Memory for Data Transfer (1.2x speedup)

 Problem: CPU→GPU data transfer uses pageable memory.

 Files to modify:
 - alphazero/training/learner.py (train_step method ~line 83)

 Changes:
 def train_step(self):
     batch = self.replay_buffer.sample(self.batch_size)

     # Use non_blocking transfers with pinned memory
     obs = torch.from_numpy(batch['observations']).to(
         self.device, non_blocking=True
     )
     target_policy = torch.from_numpy(batch['policies']).to(
         self.device, non_blocking=True
     )
     target_value = torch.from_numpy(batch['values']).to(
         self.device, non_blocking=True
     )

 Expected speedup: 1.2x

 ---
 Optimization 5: Adaptive Batch Collection (1.3x speedup)

 Problem: Fixed batch timeout doesn't adapt to actor count.

 Files to modify:
 - alphazero/selfplay/inference_server.py (lines 200-250)

 Changes:
 def _collect_batch(self) -> List[InferenceRequest]:
     """Collect batch with adaptive timeout."""
     batch = []
     start_time = time.time()

     # Adaptive: wait longer if batch is filling slowly
     min_batch = max(1, self.batch_size // 4)  # At least 25% full

     while len(batch) < self.batch_size:
         elapsed = time.time() - start_time

         # Exit if timeout AND we have minimum batch
         if elapsed > self.batch_timeout and len(batch) >= min_batch:
             break

         # Hard timeout at 2x configured timeout
         if elapsed > self.batch_timeout * 2:
             break

         try:
             request = self.request_queue.get(timeout=0.001)
             batch.append(request)
         except queue.Empty:
             if len(batch) >= min_batch:
                 break

     return batch

 Expected speedup: 1.3x (better batch utilization)

 ---
 Implementation Plan

 Phase 1: Quick Wins (Day 1) - Expected 4x speedup

 1. Update inference server defaults (batch_size, timeout)
 2. Add hardware profile system to train.py
 3. Increase default actor count based on profile

 Phase 2: MCTS Optimization (Day 2) - Expected +2x speedup

 4. Ensure Cython MCTS compiles on all platforms
 5. Add auto-detection of best MCTS backend
 6. Add build instructions for Cython on Windows/Linux

 Phase 3: Data Pipeline (Day 3) - Expected +1.2x speedup

 7. Add pinned memory transfers
 8. Implement adaptive batch collection
 9. Add performance monitoring/logging

 ---
 Files to Modify
 ┌────────────────────────────────────────┬───────────────────────────────────────────────────────────┐
 │                  File                  │                          Changes                          │
 ├────────────────────────────────────────┼───────────────────────────────────────────────────────────┤
 │ alphazero/config.py                    │ Add InferenceConfig, hardware profiles                    │
 ├────────────────────────────────────────┼───────────────────────────────────────────────────────────┤
 │ alphazero/selfplay/inference_server.py │ Increase batch_size/timeout defaults, adaptive collection │
 ├────────────────────────────────────────┼───────────────────────────────────────────────────────────┤
 │ scripts/train.py                       │ Add --profile argument, auto-detect backend               │
 ├────────────────────────────────────────┼───────────────────────────────────────────────────────────┤
 │ alphazero/training/learner.py          │ Pinned memory transfers                                   │
 ├────────────────────────────────────────┼───────────────────────────────────────────────────────────┤
 │ alphazero/mcts/__init__.py             │ Auto-detect best backend                                  │
 └────────────────────────────────────────┴───────────────────────────────────────────────────────────┘
 ---
 New CLI Interface

 # Auto-detect hardware and use optimal settings
 uv run python scripts/train.py --profile auto

 # Explicit profile selection
 uv run python scripts/train.py --profile high   # A100/Colab
 uv run python scripts/train.py --profile mid    # T4/V100
 uv run python scripts/train.py --profile low    # 4060 laptop

 # Override specific settings
 uv run python scripts/train.py --profile mid --actors 16 --inference-batch-size 512

 ---
 Verification Strategy

 1. Benchmark Script

 # Add benchmarking mode
 uv run python scripts/train.py --benchmark --profile high --steps 500

 2. Metrics to Track

 - Games generated per hour
 - Training steps per second
 - GPU utilization (%)
 - Inference batch fill rate (%)
 - Average inference latency (ms)

 3. Expected Results by Profile
 ┌─────────┬────────┬────────────┬──────────────────┐
 │ Profile │ Actors │ Batch Size │ Expected Speedup │
 ├─────────┼────────┼────────────┼──────────────────┤
 │ HIGH    │ 24     │ 512        │ 6-8x             │
 ├─────────┼────────┼────────────┼──────────────────┤
 │ MID     │ 12     │ 256        │ 4-5x             │
 ├─────────┼────────┼────────────┼──────────────────┤
 │ LOW     │ 6      │ 128        │ 2-3x             │
 └─────────┴────────┴────────────┴──────────────────┘
 ---
 Risk Mitigation
 ┌─────────────────────────┬───────────────────────────────────────────┐
 │          Risk           │                Mitigation                 │
 ├─────────────────────────┼───────────────────────────────────────────┤
 │ OOM on smaller GPUs     │ Profile system auto-selects safe defaults │
 ├─────────────────────────┼───────────────────────────────────────────┤
 │ Cython not available    │ Graceful fallback to Python backend       │
 ├─────────────────────────┼───────────────────────────────────────────┤
 │ Windows multiprocessing │ Already fixed with spawn method           │
 ├─────────────────────────┼───────────────────────────────────────────┤
 │ Batch timeout too long  │ Adaptive collection prevents starvation   │
 └─────────────────────────┴───────────────────────────────────────────┘
 ---
 Summary

 Minimum changes for 5x speedup:
 1. inference_server.py: batch_size=256, batch_timeout=0.015
 2. train.py: --actors 12-24 based on GPU
 3. Auto-detect and use Cython MCTS backend

 Files to create/modify:
 1. alphazero/config.py - Add hardware profiles
 2. alphazero/selfplay/inference_server.py - Better batching
 3. scripts/train.py - Profile selection CLI
 4. alphazero/training/learner.py - Pinned memory