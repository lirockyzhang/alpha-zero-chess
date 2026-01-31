Batched Leaf Evaluation in MCTS - Implementation Plan

 Problem Statement

 The AlphaZero training pipeline has low GPU utilization (20-50%) because:
 - Actors generate inference requests sequentially (one leaf at a time)
 - Average batch size is much smaller than configured (40-60 vs 128-512 target)
 - Each MCTS simulation evaluates one leaf node, waits for response, then continues

 Goal: Implement batched leaf evaluation to generate more inference requests per actor and improve GPU utilization.

 ---
 Alternative Architecture: Fully Synchronized Batched MCTS

 Overview

 A more aggressive approach suggested by the user involves running hundreds of parallel games with synchronized
 batching:

 1. Pause and Collect: Run hundreds of games in parallel on the same machine
 2. Sync Inference: When a game needs leaf evaluation, it pauses and adds state to a queue
 3. Batch Execution: Dynamic batcher collects states until batch size (256-512) is reached
 4. Massive Forward Pass: GPU runs single forward pass on large batch
 5. Unpause: Results distributed back, all games continue

 Viability Analysis

 ✅ Advantages

 1. Maximum GPU Utilization
   - Guaranteed large batches (256-512 positions)
   - GPU utilization: 90-95% (vs 70-85% with our plan)
   - No partial batches or timeout issues
 2. Simpler Synchronization
   - All games pause at evaluation points
   - No complex queue management across processes
   - Clear batch boundaries for profiling
 3. Proven Approach
   - This is how DeepMind implemented AlphaGo Zero and AlphaZero
   - Well-documented in the original papers
   - Known to work at scale
 4. Better Throughput
   - Expected: 5-10x improvement (vs 3-4x with our plan)
   - More games per hour due to better GPU utilization
   - Lower latency per batch (no waiting for requests to accumulate)

 ❌ Disadvantages

 1. Major Architectural Changes Required
   - Complete rewrite of actor/coordinator system
   - Current architecture: separate processes per actor
   - New architecture: single process managing hundreds of games
   - Estimated implementation time: 2-3 weeks (vs 3-5 hours for our plan)
 2. Synchronization Overhead
   - All games must wait for slowest game to reach evaluation point
   - If one game is in deep tree search, others wait
   - Potential idle time if games are out of sync
 3. Memory Intensive
   - Need to maintain hundreds of game states in memory
   - Each game: ~10 MB (MCTS tree + game state)
   - 500 games × 10 MB = 5 GB RAM minimum
   - May require careful memory management
 4. CPU Bottleneck
   - Hundreds of MCTS tree operations on CPU
   - Python GIL may become bottleneck
   - May need to use multiprocessing or Cython/C++ MCTS
 5. Complexity
   - Harder to debug (hundreds of games in flight)
   - More complex state management
   - Requires careful handling of game lifecycle

  Comparison to Current Pl
 ┌──────────────────────┬───────────────────────────────────────┬─────────────────────────────────────────────────┐
 │        Aspect        │ Current Plan (BatchInferenceRequest)  │      Suggested (Synchronized Batched MCTS)      │
 ├──────────────────────┼───────────────────────────────────────┼─────────────────────────────────────────────────┤
 │ Architecture         │ Distributed actors + inference server │ Centralized game manager                        │
 ├──────────────────────┼───────────────────────────────────────┼─────────────────────────────────────────────────┤
 │ Parallelism          │ 4-64 actors (separate processes)      │ 100-500 games (single process or few processes) │
 ├──────────────────────┼───────────────────────────────────────┼─────────────────────────────────────────────────┤
 │ Batch Size           │ 8-16 per actor, 128-512 server batch  │ 256-512 global batch                            │
 ├──────────────────────┼───────────────────────────────────────┼─────────────────────────────────────────────────┤
 │ Synchronization      │ Asynchronous (actors independent)     │ Synchronous (all games pause together)          │
 ├──────────────────────┼───────────────────────────────────────┼─────────────────────────────────────────────────┤
 │ GPU Utilization      │ 70-85%                                │ 90-95%                                          │
 ├──────────────────────┼───────────────────────────────────────┼─────────────────────────────────────────────────┤
 │ Implementation Time  │ 3-5 hours                             │ 2-3 weeks                                       │
 ├──────────────────────┼───────────────────────────────────────┼─────────────────────────────────────────────────┤
 │ Complexity           │ Low (incremental change)              │ High (architectural rewrite)                    │
 ├──────────────────────┼───────────────────────────────────────┼─────────────────────────────────────────────────┤
 │ Risk                 │ Low                                   │ Medium-High                                     │
 ├──────────────────────┼───────────────────────────────────────┼─────────────────────────────────────────────────┤
 │ Expected Improvement │ 3-4x                                  │ 5-10x                                           │
 └──────────────────────┴───────────────────────────────────────┴─────────────────────────────────────────────────┘
 JAX + MCTX Alternative

 The suggestion also mentions JAX + MCTX (DeepMind's library):

 What it is:
 - MCTX: JIT-compiled MCTS implementation in JAX
 - Entire MCTS search runs as GPU kernel
 - Significantly faster than Python MCTS

 Pros:
 - ✅ 10-100x faster MCTS (GPU-accelerated tree operations)
 - ✅ Automatic batching and parallelization
 - ✅ Proven at scale (used by DeepMind)

 Cons:
 - ❌ Requires complete rewrite in JAX
 - ❌ Current codebase is PyTorch
 - ❌ Steep learning curve
 - ❌ Harder to debug (JIT-compiled code)
 - ❌ Estimated implementation time: 1-2 months

 Verdict: Not viable for current project unless willing to rewrite everything.

 ---
 Recommendation

 For Idea Validation (Current Goal)

 Recommended: Stick with BatchInferenceRequest protocol plan

 Rationale:
 1. ✅ Incremental: Builds on existing architecture
 2. ✅ Low risk: Small, focused changes
 3. ✅ Fast implementation: 3-5 hours vs 2-3 weeks
 4. ✅ Good improvement: 3-4x is significant for validation
 5. ✅ Learning opportunity: Understand bottlenecks before major rewrite

 For Production (Future)

 Consider: Synchronized Batched MCTS architecture

 When to implement:
 1. After validating that GPU utilization is the bottleneck
 2. After measuring actual improvement from BatchInferenceRequest
 3. When ready to invest 2-3 weeks in architectural changes
 4. When scaling to larger training runs

 Hybrid Approach (Best of Both Worlds)

 Possible middle ground:

 1. Phase 1 (Now): Implement BatchInferenceRequest protocol
   - Quick win: 3-4x improvement
   - Validate GPU utilization is the bottleneck
   - Measure actual performance gains
 2. Phase 2 (If needed): Increase games per actor
   - Each actor manages 4-8 games instead of 1
   - Collect leaf nodes from all games before batching
   - Moderate complexity increase
   - Expected: 5-6x improvement
 3. Phase 3 (If still needed): Full synchronized batching
   - Implement centralized game manager
   - Run hundreds of games with global synchronization
   - Maximum GPU utilization
   - Expected: 8-10x improvement

 ---
 Updated Implementation Plan

 Given the analysis above, I recommend proceeding with the BatchInferenceRequest protocol as planned, with the
 understanding that:

 1. This is an incremental optimization (3-4x improvement)
 2. It's a stepping stone to more aggressive optimizations
 3. It allows us to validate assumptions about GPU bottleneck
 4. It provides learning about the system's behavior

 If after implementing this, GPU utilization is still low or more improvement is needed, we can consider the
 synchronized batched MCTS architecture.

 ---
 Current Implementation Analysis

 ✅ What Already Exists

 1. ParallelMCTS.search_with_batching() (parallel.py:262-328)

 This method is already implemented and provides batched leaf evaluation:

 def search_with_batching(
     self,
     state,
     evaluator,
     move_number: int = 0,
     add_noise: bool = True,
     batch_size: int = 8  # Collects 8 leaves before evaluating
 ) -> Tuple[np.ndarray, MCTSNode, MCTSStats]:

 How it works:
 1. Collects batch_size pending leaf nodes using virtual loss
 2. Batch evaluates all leaves together
 3. Expands nodes and backpropagates values
 4. Repeats until num_simulations is reached

 Key implementation details:
 - Uses virtual loss to prevent multiple selections of the same path
 - Calls evaluator.evaluate_batch() if available
 - Falls back to sequential evaluate() calls if batch method not available
 - Thread-safe with proper locking

 2. NetworkEvaluator.evaluate_batch() (evaluator.py:87-111)

 Already supports batched evaluation:
 def evaluate_batch(
     self,
     observations: np.ndarray,  # (batch, 119, 8, 8)
     legal_masks: np.ndarray    # (batch, 4672)
 ) -> Tuple[np.ndarray, np.ndarray]:

 ❌ What's Missing

 1. BatchedEvaluator.evaluate_batch() - CRITICAL

 The BatchedEvaluator (used with centralized inference server) only has evaluate() for single evaluations:

 # Current implementation (inference_server.py:301-336)
 def evaluate(self, observation, legal_mask):
     # Sends ONE request, waits for ONE response
     request = InferenceRequest(...)
     self.request_queue.put(request)
     response = self.response_queue.get(timeout=self.timeout)
     return response.policy, response.value

 Missing: evaluate_batch() method that can send multiple requests and collect multiple responses.

 2. SelfPlayGame doesn't use batched search

 Current implementation (game.py:52-58):
 # Uses regular search (one leaf at a time)
 policy, root, stats = self.mcts.search(
     state,
     self.evaluator,
     move_number=move_number,
     add_noise=True
 )

 Missing: Call to search_with_batching() instead of search().

 3. No configuration for MCTS batch size

 MCTSConfig (config.py:16-24) doesn't have a batch_size parameter.

 ---
 Implementation Plan

 Phase 1: Design New BatchInferenceRequest Protocol

 Why this approach is better:
 - ✅ Explicit batching: Server knows these requests belong together
 - ✅ More efficient: One message instead of N messages
 - ✅ Lower latency: No waiting for requests to accumulate
 - ✅ Guaranteed batching: Server processes as a true batch
 - ✅ Simpler response handling: Single batch response
 - ✅ Better for profiling: Clear batch boundaries

 File: alphazero/selfplay/inference_server.py

 Step 1: Add new message types (after line 36)

 @dataclass
 class BatchInferenceRequest:
     """Batch inference request containing multiple positions."""
     request_id: int
     actor_id: int
     observations: np.ndarray  # (batch_size, 119, 8, 8)
     legal_masks: np.ndarray   # (batch_size, 4672)
     batch_size: int

 @dataclass
 class BatchInferenceResponse:
     """Batch inference response containing multiple evaluations."""
     request_id: int
     actor_id: int
     policies: np.ndarray  # (batch_size, 4672)
     values: np.ndarray    # (batch_size,)
     batch_size: int

 Step 2: Update InferenceServer to handle batch requests (lines 138-197)

 Modify the main loop to handle both single and batch requests:

 def run(self):
     """Main server loop."""
     while not self._shutdown_event.is_set():
         try:
             # Check for weight updates
             self._check_for_weight_updates()

             # Collect batch of requests (mixed single and batch requests)
             requests = self._collect_batch()

             if not requests:
                 continue

             # Separate single and batch requests
             single_requests = [r for r in requests if isinstance(r, InferenceRequest)]
             batch_requests = [r for r in requests if isinstance(r, BatchInferenceRequest)]

             # Process single requests (existing logic)
             if single_requests:
                 policies, values = self._run_inference(self.network, single_requests)
                 for i, request in enumerate(single_requests):
                     response = InferenceResponse(
                         request_id=request.request_id,
                         actor_id=request.actor_id,
                         policy=policies[i],
                         value=values[i]
                     )
                     self.response_queues[request.actor_id].put(response)

             # Process batch requests (new logic)
             for batch_request in batch_requests:
                 policies, values = self._run_batch_inference(
                     self.network,
                     batch_request.observations,
                     batch_request.legal_masks
                 )
                 response = BatchInferenceResponse(
                     request_id=batch_request.request_id,
                     actor_id=batch_request.actor_id,
                     policies=policies,
                     values=values,
                     batch_size=batch_request.batch_size
                 )
                 self.response_queues[batch_request.actor_id].put(response)

         except Exception as e:
             logger.error(f"Error in inference server: {e}")

 Step 3: Add _run_batch_inference method (after line 268)

 def _run_batch_inference(
     self,
     network: torch.nn.Module,
     observations: np.ndarray,  # (batch_size, 119, 8, 8)
     legal_masks: np.ndarray    # (batch_size, 4672)
 ) -> Tuple[np.ndarray, np.ndarray]:
     """Run inference on a batch of positions.

     Args:
         network: Neural network
         observations: Batch of observations
         legal_masks: Batch of legal masks

     Returns:
         Tuple of (policies, values)
     """
     # Convert to tensors
     obs_tensor = torch.from_numpy(observations).float().to(self.device)
     mask_tensor = torch.from_numpy(legal_masks).float().to(self.device)

     # Run inference with optional mixed precision
     with torch.no_grad():
         if self.use_amp:
             with autocast('cuda'):
                 policies, values = network.predict(obs_tensor, mask_tensor)
         else:
             policies, values = network.predict(obs_tensor, mask_tensor)

     return policies.cpu().numpy(), values.cpu().numpy().flatten()

 Step 4: Add evaluate_batch() to BatchedEvaluator (after line 336)

 def evaluate_batch(
     self,
     observations: np.ndarray,  # (batch, 119, 8, 8)
     legal_masks: np.ndarray    # (batch, 4672)
 ) -> Tuple[np.ndarray, np.ndarray]:
     """Evaluate a batch of positions via the inference server.

     Sends a single BatchInferenceRequest containing all positions.
     """
     batch_size = observations.shape[0]
     request_id = self.request_counter
     self.request_counter += 1

     # Create batch request
     request = BatchInferenceRequest(
         request_id=request_id,
         actor_id=self.actor_id,
         observations=observations.copy(),
         legal_masks=legal_masks.copy(),
         batch_size=batch_size
     )

     # Send request
     self.request_queue.put(request)

     # Wait for batch response
     try:
         response = self.response_queue.get(timeout=self.timeout)

         # Verify it's a batch response
         if not isinstance(response, BatchInferenceResponse):
             raise TypeError(f"Expected BatchInferenceResponse, got {type(response)}")

         if response.batch_size != batch_size:
             raise ValueError(
                 f"Response batch size {response.batch_size} != request batch size {batch_size}"
             )

         return response.policies, response.values

     except Empty:
         raise TimeoutError(
             f"Batch inference request {request_id} timed out after {self.timeout}s"
         )

 ---
 Phase 2: Add MCTS batch size configuration

 File: alphazero/config.py

 Changes:

 1. Add parameter to MCTSConfig (lines 16-24):
 @dataclass
 class MCTSConfig:
     num_simulations: int = 800
     c_puct: float = 1.25
     dirichlet_alpha: float = 0.3
     dirichlet_epsilon: float = 0.25
     temperature: float = 1.0
     temperature_threshold: int = 30
     backend: MCTSBackend = MCTSBackend.PYTHON
     batch_size: int = 16  # NEW: Number of leaves to batch together

 2. Update hardware profiles (lines 104-142) to include MCTS batch sizes:
 PROFILES = {
     'high': TrainingProfile(
         actors=64,
         simulations=800,
         inference_batch_size=1024,
         inference_timeout=0.03,
         mcts_batch_size=32,  # NEW: Larger batches for high-end GPUs
     ),
     'mid': TrainingProfile(
         actors=32,
         simulations=800,
         inference_batch_size=256,
         inference_timeout=0.015,
         mcts_batch_size=16,  # NEW
     ),
     'low': TrainingProfile(
         actors=28,
         simulations=800,
         inference_batch_size=128,
         inference_timeout=0.01,
         mcts_batch_size=8,   # NEW: Smaller batches for low-end GPUs
     ),
 }

 3. Add mcts_batch_size to TrainingProfile dataclass (lines 90-102):
 @dataclass
 class TrainingProfile:
     name: str
     actors: int
     simulations: int
     training_batch_size: int
     min_buffer_size: int
     filters: int
     blocks: int
     inference_batch_size: int
     inference_timeout: float
     mcts_backend: str
     mcts_batch_size: int  # NEW

 ---
 Phase 3: Update SelfPlayGame to use batched search

 File: alphazero/selfplay/game.py

 Changes:

 1. Import ParallelMCTS (line 11):
 from ..mcts import MCTSBase, create_mcts
 from ..mcts.python.parallel import ParallelMCTS  # NEW

 2. Update play() method to use batched search (lines 52-58):

 Before:
 # Run MCTS search
 policy, root, stats = self.mcts.search(
     state,
     self.evaluator,
     move_number=move_number,
     add_noise=True
 )

 After:
 # Run MCTS search with batched leaf evaluation
 if isinstance(self.mcts, ParallelMCTS) and hasattr(self.evaluator, 'evaluate_batch'):
     # Use batched search if supported
     batch_size = getattr(self.mcts.config, 'batch_size', 16)
     policy, root, stats = self.mcts.search_with_batching(
         state,
         self.evaluator,
         move_number=move_number,
         add_noise=True,
         batch_size=batch_size
     )
 else:
     # Fall back to regular search
     policy, root, stats = self.mcts.search(
         state,
         self.evaluator,
         move_number=move_number,
         add_noise=True
     )

 ---
 Phase 4: Update actor to use ParallelMCTS

 File: alphazero/selfplay/batched_actor.py

 Changes:

 Update MCTS creation (lines 112-173):

 Before:
 # Create MCTS
 mcts = create_mcts(config=self.config.mcts)

 After:
 # Create ParallelMCTS for batched leaf evaluation
 from ..mcts.python.parallel import ParallelMCTS
 mcts = ParallelMCTS(config=self.config.mcts)

 Rationale: ParallelMCTS supports search_with_batching(), while the default MCTS from create_mcts() may not.

 ---
 Phase 5: Update training script to pass MCTS batch size

 File: scripts/train.py

 Changes:

 Update config creation (lines 319-332):

 # Create configuration
 config = AlphaZeroConfig(
     mcts=MCTSConfig(
         num_simulations=simulations,
         backend=mcts_backend,
         batch_size=profile.mcts_batch_size if profile else 16  # NEW
     ),
     network=NetworkConfig(num_filters=num_filters, num_blocks=num_blocks),
     training=TrainingConfig(
         batch_size=batch_size,
         log_interval=1000,
         use_amp=not args.no_amp_training,
         use_amp_inference=not args.no_amp_inference
     ),
     replay_buffer=ReplayBufferConfig(min_size_to_train=min_buffer),
     device=args.device,
     checkpoint_dir=args.checkpoint_dir,
     log_dir=args.log_dir,
 )

 ---
 Expected Impact

 Before (Current)

 - Inference requests per move: 801 (1 root + 800 simulations)
 - Requests sent: One at a time (sequential)
 - Protocol: Single InferenceRequest per position
 - GPU batch utilization: 20-50%
 - Average batch size: 40-60 (vs target 128-512)

 After (With BatchInferenceRequest Protocol)

 - Inference requests per move: 801 (same total)
 - Requests sent: 16 at a time (true batches)
 - Protocol: BatchInferenceRequest with 16 positions
 - GPU batch utilization: 70-85% (estimated)
 - Average batch size: 200-300 (estimated)

 Key improvements:
 1. Guaranteed batching: Server receives true batches, not hoping requests arrive together
 2. Lower latency: ~50% reduction (no waiting for batch collection)
 3. Better GPU utilization: Requests arrive as cohesive batches
 4. Cleaner protocol: Explicit batch semantics
 5. More throughput: 3-4x more games per hour (vs 2-3x with sequential approach)

 ---
 Validation Plan

 Test 1: Unit Test for evaluate_batch()

 Create test in tests/test_batched_inference.py:

 def test_batched_evaluator_evaluate_batch():
     """Test that BatchedEvaluator.evaluate_batch() works correctly."""
     # Setup
     request_queue = Queue()
     response_queue = Queue()
     evaluator = BatchedEvaluator(
         actor_id=0,
         request_queue=request_queue,
         response_queue=response_queue,
         timeout=5.0
     )

     # Create batch of observations
     batch_size = 4
     observations = np.random.randn(batch_size, 119, 8, 8).astype(np.float32)
     legal_masks = np.ones((batch_size, 4672), dtype=np.float32)

     # Mock responses
     def mock_server():
         for i in range(batch_size):
             request = request_queue.get()
             response = InferenceResponse(
                 request_id=request.request_id,
                 actor_id=request.actor_id,
                 policy=np.random.randn(4672).astype(np.float32),
                 value=0.5
             )
             response_queue.put(response)

     # Run in thread
     import threading
     thread = threading.Thread(target=mock_server)
     thread.start()

     # Test
     policies, values = evaluator.evaluate_batch(observations, legal_masks)

     # Verify
     assert policies.shape == (batch_size, 4672)
     assert values.shape == (batch_size,)

     thread.join()

 Test 2: Integration Test for search_with_batching()

 Create test in tests/test_mcts_batching.py:

 def test_parallel_mcts_search_with_batching():
     """Test that ParallelMCTS.search_with_batching() works correctly."""
     from alphazero.mcts.python.parallel import ParallelMCTS
     from alphazero.mcts.evaluator import NetworkEvaluator
     from alphazero.chess_env import GameState
     from alphazero.neural import AlphaZeroNetwork

     # Setup
     config = MCTSConfig(num_simulations=100, batch_size=8)
     mcts = ParallelMCTS(config=config)

     network = AlphaZeroNetwork(num_filters=64, num_blocks=5)
     evaluator = NetworkEvaluator(network, device='cpu', use_amp=False)

     state = GameState()

     # Test
     policy, root, stats = mcts.search_with_batching(
         state,
         evaluator,
         move_number=0,
         add_noise=True,
         batch_size=8
     )

     # Verify
     assert policy.shape == (4672,)
     assert np.isclose(np.sum(policy), 1.0, atol=1e-5)
     assert stats.num_simulations <= 100
     assert stats.nodes_created > 0

 Test 3: End-to-End Training Test

 Run short training session and monitor metrics:

 # Run training with batched leaf evaluation
 uv run python scripts/train.py \
     --profile low \
     --steps 100 \
     --actors 4 \
     --batched-inference

 # Monitor GPU utilization
 nvidia-smi dmon -s u -d 1

 Success criteria:
 - Training completes without errors
 - GPU utilization increases from 20-50% to 60-80%
 - Average batch size increases from 40-60 to 120-200
 - Games per hour increases by 2-3x

 ---
 Critical Files to Modify
 ┌────────────────────────────────────────┬─────────┬─────────────────────────────────────────────────┐
 │                  File                  │  Lines  │                     Changes                     │
 ├────────────────────────────────────────┼─────────┼─────────────────────────────────────────────────┤
 │ alphazero/selfplay/inference_server.py │ 336+    │ Add evaluate_batch() method to BatchedEvaluator │
 ├────────────────────────────────────────┼─────────┼─────────────────────────────────────────────────┤
 │ alphazero/config.py                    │ 16-24   │ Add batch_size to MCTSConfig                    │
 ├────────────────────────────────────────┼─────────┼─────────────────────────────────────────────────┤
 │ alphazero/config.py                    │ 90-102  │ Add mcts_batch_size to TrainingProfile          │
 ├────────────────────────────────────────┼─────────┼─────────────────────────────────────────────────┤
 │ alphazero/config.py                    │ 104-142 │ Update hardware profiles with MCTS batch sizes  │
 ├────────────────────────────────────────┼─────────┼─────────────────────────────────────────────────┤
 │ alphazero/selfplay/game.py             │ 11      │ Import ParallelMCTS                             │
 ├────────────────────────────────────────┼─────────┼─────────────────────────────────────────────────┤
 │ alphazero/selfplay/game.py             │ 52-58   │ Use search_with_batching() when available       │
 ├────────────────────────────────────────┼─────────┼─────────────────────────────────────────────────┤
 │ alphazero/selfplay/batched_actor.py    │ 112-173 │ Create ParallelMCTS instead of default MCTS     │
 ├────────────────────────────────────────┼─────────┼─────────────────────────────────────────────────┤
 │ scripts/train.py                       │ 319-332 │ Pass mcts_batch_size from profile to config     │
 └────────────────────────────────────────┴─────────┴─────────────────────────────────────────────────┘
 ---
 Implementation Risks and Mitigations

 Risk 1: evaluate_batch() timeout issues

 Problem: If one request in the batch times out, the entire batch fails.

 Mitigation:
 - Use generous timeout (10s instead of 5s)
 - Add retry logic for failed batches
 - Log warnings for slow requests

 Risk 2: Response ordering

 Problem: Responses may arrive out of order from the inference server.

 Mitigation:
 - Current implementation doesn't rely on ordering
 - Each response has request_id for matching
 - Queue guarantees FIFO for single actor

 Risk 3: Virtual loss correctness

 Problem: Virtual loss implementation in ParallelMCTS may have bugs.

 Mitigation:
 - Already implemented and tested in codebase
 - Used in parallel search (lines 182-260)
 - Add specific tests for batched search

 Risk 4: Memory usage

 Problem: Batching 16 leaves means 16× memory for observations/masks.

 Mitigation:
 - 16 × (119×8×8 + 4672) × 4 bytes = ~2.5 MB per batch
 - Negligible compared to GPU memory (8-80 GB)
 - Can reduce batch_size if needed

 ---
 Alternative Approaches Considered

 Alternative 1: Sequential requests (original plan)

 Approach: Send N individual InferenceRequests rapidly, hope server batches them.

 Pros:
 - Simple - no protocol changes
 - Compatible with existing infrastructure

 Cons:
 - ❌ No guarantee requests will be batched together
 - ❌ Higher latency (waiting for batch collection timeout)
 - ❌ More messages (N requests + N responses)
 - ❌ Harder to profile (unclear batch boundaries)

 Decision: Rejected in favor of BatchInferenceRequest protocol.

 Alternative 2: Async evaluate_batch() with threading

 Approach: Use threading to send requests asynchronously.

 Pros:
 - Requests sent faster
 - Better overlap with inference server

 Cons:
 - More complex implementation
 - Threading overhead
 - Harder to debug
 - Still relies on server batching individual requests

 Decision: Not needed. BatchInferenceRequest is cleaner.

 Alternative 3: Increase num_simulations instead

 Approach: Run 1600 simulations instead of 800.

 Pros:
 - Simple config change
 - More inference requests

 Cons:
 - Slower games (2x longer)
 - Doesn't improve batch utilization
 - Lower games per hour

 Decision: Not recommended. Batching is better approach.

 ---
 Summary

 This plan implements batched leaf evaluation in MCTS by:

 1. ✅ Leveraging existing code: ParallelMCTS.search_with_batching() is already implemented
 2. ✅ Adding missing piece: BatchedEvaluator.evaluate_batch() method
 3. ✅ Simple approach: Sequential requests, no protocol changes
 4. ✅ Configuration: Add mcts_batch_size parameter to profiles
 5. ✅ Integration: Update game loop and actor to use batched search

 Expected result: 3-4x improvement in GPU utilization and training throughput with clean protocol design.

 Implementation time: 3-5 hours for coding + testing.

 Risk level: Low - clean protocol design, leverages existing tested code.