# Updated AlphaZero Training Acceleration Plan

 Goal: Achieve 5x+ training speedup for 192×15 network on A100/Colab, deployable to 4060 laptop

 Executive Summary

 Target Configuration:
 - Network: 192 filters × 15 blocks (~11M parameters)
 - Training: A100/Colab (cloud)
 - Inference: RTX 4060 laptop (~0.5-0.8 sec/move)
 - MCTS: 800 simulations (maintain quality)

 Bottlenecks to fix:
 1. Inference batching (batch_size=32 → 512)
 2. Actor parallelism (4 → 24 actors)
 3. MCTS backend (Python → Cython)

 Expected speedup: 5-8x

 ---
 Hardware Profiles

 Profile: HIGH (A100 / Colab A100) - RECOMMENDED FOR TRAINING

 PROFILE_HIGH = {
     # Network architecture (optimized for 4060 inference)
     'filters': 192,
     'blocks': 15,
     'network_params': '~11M parameters',

     # Self-play configuration
     'actors': 24,
     'simulations': 800,
     'mcts_backend': 'cython',

     # Inference server
     'inference_batch_size': 512,
     'inference_timeout': 0.02,  # 20ms

     # Training
     'training_batch_size': 8192,
     'replay_buffer_size': 1_000_000,
     'min_buffer_size': 50_000,

     # Memory estimates
     'gpu_memory_usage': '~20GB',
     'cpu_memory_usage': '~16GB',
 }

 Profile: MID (T4 16GB / V100 16GB)

 PROFILE_MID = {
     'filters': 192,
     'blocks': 15,
     'actors': 12,
     'simulations': 800,
     'mcts_backend': 'cython',
     'inference_batch_size': 256,
     'inference_timeout': 0.015,
     'training_batch_size': 4096,
     'replay_buffer_size': 500_000,
     'min_buffer_size': 20_000,
 }

 Profile: LOW (RTX 4060 Laptop 8GB) - FOR LOCAL TRAINING ONLY

 PROFILE_LOW = {
     'filters': 64,
     'blocks': 5,
     'actors': 4,
     'simulations': 800,
     'mcts_backend': 'python',
     'inference_batch_size': 128,
     'inference_timeout': 0.01,
     'training_batch_size': 2048,
     'replay_buffer_size': 200_000,
     'min_buffer_size': 10_000,
 }

 ---
 Optimization 1: Increase Inference Batch Size and Timeout (3-4x speedup)

 Problem: Current batch_size=32 and batch_timeout=1ms severely underutilize GPU.

 Files to modify:
 - alphazero/selfplay/inference_server.py (lines 68-69)

 Changes:
 # inference_server.py - InferenceServer.__init__
 class InferenceServer:
     def __init__(
         self,
         request_queue: mp.Queue,
         response_queues: Dict[int, mp.Queue],
         weight_queue: mp.Queue,
         network: nn.Module,
         device: str = "cuda",
         batch_size: int = 512,       # Was 32
         batch_timeout: float = 0.02,  # Was 0.001 (20ms vs 1ms)
         use_amp: bool = True,
     ):

 Expected speedup: 3-4x for inference throughput

 ---
 Optimization 2: Increase Number of Actors (1.5-2x speedup)

 Problem: Only 4 actors generating games.

 Files to modify:
 - scripts/train.py
 - alphazero/config.py

 Changes to scripts/train.py:
 # Add profile argument
 parser.add_argument("--profile", choices=['high', 'mid', 'low'],
                     default='high', help="Hardware profile")

 # Update actor default
 parser.add_argument("--actors", type=int, default=None,
                     help="Number of actors (default: from profile)")

 Changes to alphazero/config.py:
 from dataclasses import dataclass
 from typing import Optional

 @dataclass
 class TrainingProfile:
     """Hardware-specific training configuration."""
     name: str
     filters: int
     blocks: int
     actors: int
     simulations: int
     inference_batch_size: int
     inference_timeout: float
     training_batch_size: int
     replay_buffer_size: int
     min_buffer_size: int
     mcts_backend: str = 'cython'

 PROFILES = {
     'high': TrainingProfile(
         name='high',
         filters=192,
         blocks=15,
         actors=24,
         simulations=800,
         inference_batch_size=512,
         inference_timeout=0.02,
         training_batch_size=8192,
         replay_buffer_size=1_000_000,
         min_buffer_size=50_000,
     ),
     'mid': TrainingProfile(
         name='mid',
         filters=192,
         blocks=15,
         actors=12,
         simulations=800,
         inference_batch_size=256,
         inference_timeout=0.015,
         training_batch_size=4096,
         replay_buffer_size=500_000,
         min_buffer_size=20_000,
     ),
     'low': TrainingProfile(
         name='low',
         filters=64,
         blocks=5,
         actors=4,
         simulations=800,
         inference_batch_size=128,
         inference_timeout=0.01,
         training_batch_size=2048,
         replay_buffer_size=200_000,
         min_buffer_size=10_000,
         mcts_backend='python',
     ),
 }

 Expected speedup: 1.5-2x

 ---
 Optimization 3: Use Cython MCTS Backend (2-3x speedup)

 Problem: Python MCTS is slow.

 Files to modify:
 - alphazero/mcts/__init__.py
 - scripts/train.py

 Changes to alphazero/mcts/__init__.py:
 def get_best_backend() -> str:
     """Auto-detect best available MCTS backend."""
     try:
         from alphazero.mcts.cython.search import CythonMCTS
         return 'cython'
     except ImportError:
         pass

     try:
         from alphazero.mcts.cpp import CppMCTS
         return 'cpp'
     except ImportError:
         pass

     return 'python'

 Expected speedup: 2-3x for MCTS operations

 ---
 Optimization 4: Adaptive Batch Collection (1.3x speedup)

 Problem: Fixed batch timeout doesn't adapt to load.

 Files to modify:
 - alphazero/selfplay/inference_server.py

 Changes:
 def _collect_batch(self) -> List[InferenceRequest]:
     """Collect batch with adaptive timeout."""
     batch = []
     start_time = time.time()
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

 Expected speedup: 1.3x

 ---
 Optimization 5: Pinned Memory Transfers (1.2x speedup)

 Files to modify:
 - alphazero/training/learner.py

 Changes:
 def train_step(self):
     batch = self.replay_buffer.sample(self.batch_size)

     # Use non_blocking transfers
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
 Implementation Order
 ┌───────┬──────────────────────────────────┬────────────────────────────┬─────────┐
 │ Phase │               Task               │           Files            │ Speedup │
 ├───────┼──────────────────────────────────┼────────────────────────────┼─────────┤
 │ 1     │ Add TrainingProfile to config.py │ alphazero/config.py        │ -       │
 ├───────┼──────────────────────────────────┼────────────────────────────┼─────────┤
 │ 2     │ Add --profile to train.py        │ scripts/train.py           │ -       │
 ├───────┼──────────────────────────────────┼────────────────────────────┼─────────┤
 │ 3     │ Increase batch_size/timeout      │ inference_server.py        │ 3-4x    │
 ├───────┼──────────────────────────────────┼────────────────────────────┼─────────┤
 │ 4     │ Apply profile actors             │ scripts/train.py           │ 1.5x    │
 ├───────┼──────────────────────────────────┼────────────────────────────┼─────────┤
 │ 5     │ Auto-detect MCTS backend         │ alphazero/mcts/__init__.py │ 2x      │
 ├───────┼──────────────────────────────────┼────────────────────────────┼─────────┤
 │ 6     │ Adaptive batch collection        │ inference_server.py        │ 1.3x    │
 ├───────┼──────────────────────────────────┼────────────────────────────┼─────────┤
 │ 7     │ Pinned memory                    │ learner.py                 │ 1.2x    │
 └───────┴──────────────────────────────────┴────────────────────────────┴─────────┘
 Total expected speedup: 5-8x

 ---
 Files to Modify (Summary)
 ┌────────────────────────────────────────┬─────────────────────────────────────────────────┐
 │                  File                  │                     Changes                     │
 ├────────────────────────────────────────┼─────────────────────────────────────────────────┤
 │ alphazero/config.py                    │ Add TrainingProfile dataclass and PROFILES dict │
 ├────────────────────────────────────────┼─────────────────────────────────────────────────┤
 │ alphazero/selfplay/inference_server.py │ Increase defaults, add adaptive batching        │
 ├────────────────────────────────────────┼─────────────────────────────────────────────────┤
 │ scripts/train.py                       │ Add --profile argument, apply profile settings  │
 ├────────────────────────────────────────┼─────────────────────────────────────────────────┤
 │ alphazero/training/learner.py          │ Add non_blocking transfers                      │
 ├────────────────────────────────────────┼─────────────────────────────────────────────────┤
 │ alphazero/mcts/__init__.py             │ Add get_best_backend() function                 │
 └────────────────────────────────────────┴─────────────────────────────────────────────────┘
 ---
 New CLI Usage

 # Train on A100/Colab with optimal settings (192×15, 24 actors)
 uv run python scripts/train.py --profile high --batched-inference

 # Train on T4 (192×15, 12 actors)
 uv run python scripts/train.py --profile mid --batched-inference

 # Train locally on 4060 (64×5, 4 actors) - for testing only
 uv run python scripts/train.py --profile low --batched-inference

 # Override specific settings
 uv run python scripts/train.py --profile high --actors 32 --inference-batch-size 1024

 ---
 Deployment to 4060 Laptop

 After training on A100/Colab:

 # Copy checkpoint to laptop
 scp checkpoint_192x15.pt user@laptop:/path/to/checkpoints/

 # Run web interface on laptop
 uv run python web/run.py --checkpoint checkpoint_192x15.pt --simulations 800

 # Expected: ~0.5-0.8 seconds per move

 ---
 Verification

  1. Benchmark Script

 # Add benchmarking mode
 uv run python scripts/train.py --benchmark --profile high --steps 500

 2. Metrics to Track

 - Games generated per hour
 - Training steps per second
 - GPU utilization (%)
 - Inference batch fill rate (%)
 - Average inference latency (ms)

 3. Benchmark Training Speed

 # Before optimization
 Network: 5 blocks, 64 filters
 MCTS: 800 simulations, backend=cpp
 Actors: 28
 Batched inference: True
 Mixed precision training: True
 Mixed precision inference: True
 Batch size: 2048
 Replay buffer capacity: 1000000
 Min buffer size: 10000
 Continuous training: 10000 steps
 Network parameters: 1,054,151
 Resuming from checkpoints\checkpoint_run2_f64_b5.pt
 Using batched GPU inference mode
 
 150 min to fill 10000 positions in replay buffer
 32 min to train 10000 steps 

 # After optimization
 uv run python scripts/train.py --benchmark --profile high --steps 500 --batched-inference 
 # Compare: should be 5-8x faster

 2. Verify Inference on 4060

 # Test inference speed
 uv run python -c "
 import torch
 import time
 from alphazero.neural import AlphaZeroNetwork

 net = AlphaZeroNetwork(filters=192, blocks=15).cuda()
 x = torch.randn(1, 119, 8, 8).cuda()
 mask = torch.ones(1, 4672).cuda()

 # Warmup
 for _ in range(10):
     net.predict(x, mask)

 # Benchmark
 start = time.time()
 for _ in range(100):
     net.predict(x, mask)
 print(f'Inference: {(time.time()-start)/100*1000:.1f}ms per position')
 "
 # Expected: ~5-10ms per position on 4060
 # With 800 sims: ~0.5-0.8 seconds per move

 ---
 Summary

 Your setup:
 - Train 192×15 network on A100/Colab with HIGH profile
 - Deploy to 4060 laptop for play (~0.5-0.8 sec/move)
 - Expected training speedup: 5-8x

 Key changes:
 1. inference_server.py: batch_size=512, timeout=20ms
 2. train.py: --profile high (24 actors)
 3. Auto-detect Cython MCTS backend
 4. Adaptive batch collection
 5. Pinned memory transfers
 6. Cython MCTS micro-optimizations

 ---
 Optimization 6: Cython MCTS Micro-Optimizations (1.2-1.5x speedup)

 Current Cython Code Analysis

 ✅ Already Optimized:
 - Compiler directives: boundscheck=False, wraparound=False, cdivision=True
 - Typed variables with cdef
 - cpdef methods for C-level calls
 - libc.math.sqrt instead of Python math

 ⚠️ Optimization Opportunities:

 6.1: Inline q_value in select_child (node.pyx:137)

 Current code:
 for action, child in self._children.items():
     q = child.q_value  # Property call - Python overhead

 Optimized:
 for action, child in self._children.items():
     # Inline q_value calculation to avoid property overhead
     q = child.value_sum / child.visit_count if child.visit_count > 0 else 0.0

 6.2: Pre-allocate path list (search.pyx:142)

 Current code:
 cdef list path = []
 # ... in loop:
 path.append((node, action))

 Optimized:
 # Pre-allocate with max expected depth (chess games rarely exceed 200 moves)
 cdef int max_depth = 200
 cdef list path = [None] * max_depth
 cdef int path_len = 0
 # ... in loop:
 path[path_len] = (node, action)
 path_len += 1

 6.3: Use typed memoryview for visit counts (node.pyx:180)

 Current code:
 cdef np.ndarray[FLOAT_t, ndim=1] counts = np.zeros(num_actions, dtype=np.float32)

 Optimized:
 cdef float[:] counts_view
 cdef np.ndarray[FLOAT_t, ndim=1] counts = np.zeros(num_actions, dtype=np.float32)
 counts_view = counts  # Typed memoryview for faster access

 6.4: Cache children iteration (node.pyx:135)

 Current code:
 for action, child in self._children.items():

 Optimized:
 cdef dict children = self._children
 cdef list items = list(children.items())  # Single dict access
 for action, child in items:

 Files to modify:

 - alphazero/mcts/cython/node.pyx (lines 135-150, 180-188)
 - alphazero/mcts/cython/search.pyx (lines 142-152)

 Expected speedup: 1.2-1.5x for MCTS operations

 ---
 Cython Verification Checklist

 Before implementation, verify Cython is working:

 # Check if Cython MCTS is compiled
 uv run python -c "from alphazero.mcts.cython import CythonMCTS; print('Cython MCTS OK')"

 # Benchmark Cython vs Python
 uv run python -c "
 from alphazero.mcts import create_mcts
 from alphazero.config import MCTSConfig
 from alphazero.chess_env import GameState
 import time

 config = MCTSConfig(num_simulations=100)

 # Test Python
 py_mcts = create_mcts(config, backend='python')
 state = GameState()

 start = time.time()
 for _ in range(10):
     py_mcts.search(state, lambda o, m: (m/m.sum(), 0.0), add_noise=False)
 py_time = time.time() - start

 # Test Cython
 cy_mcts = create_mcts(config, backend='cython')
 start = time.time()
 for _ in range(10):
     cy_mcts.search(state, lambda o, m: (m/m.sum(), 0.0), add_noise=False)
 cy_time = time.time() - start

 print(f'Python: {py_time:.2f}s, Cython: {cy_time:.2f}s, Speedup: {py_time/cy_time:.1f}x')
 "