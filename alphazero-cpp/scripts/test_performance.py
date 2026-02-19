"""
Performance testing script for C++ MCTS with real trained models.

Tests:
1. Single-game MCTS performance with real neural network
2. Batch encoding performance
3. Multiple searches consistency
4. Memory usage and throughput
5. Sequential self-play performance
6. Parallel self-play performance (cross-game batching)

Updated for 123-channel encoding and new checkpoint format (v2.0).
"""

import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "build" / "Release"))

try:
    import alphazero_cpp
except ImportError:
    print("ERROR: alphazero_cpp not found. Build it first.")
    sys.exit(1)

import chess


# =============================================================================
# Neural Network (matches train.py for checkpoint compatibility)
# =============================================================================

INPUT_CHANNELS = 123
POLICY_SIZE = 4672


class ConvBlock(nn.Module):
    """Convolutional block with batch norm and ReLU."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
    """Residual block matching Python backend."""
    def __init__(self, num_filters: int):
        super().__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_filters)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class PolicyHead(nn.Module):
    """Policy head: outputs action probabilities."""
    def __init__(self, in_channels: int, num_filters: int = 2, num_actions: int = POLICY_SIZE):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, num_filters, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(num_filters)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(num_filters * 8 * 8, num_actions)

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        x = x.reshape(x.size(0), -1)
        return self.fc(x)


class ValueHead(nn.Module):
    """Value head: outputs position evaluation."""
    def __init__(self, in_channels: int, num_filters: int = 1, hidden_size: int = 256):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, num_filters, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(num_filters)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(num_filters * 8 * 8, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        x = x.reshape(x.size(0), -1)
        x = self.relu(self.fc1(x))
        return self.tanh(self.fc2(x))


class AlphaZeroNet(nn.Module):
    """AlphaZero neural network - compatible with train.py checkpoints."""

    def __init__(
        self,
        input_channels: int = INPUT_CHANNELS,
        num_filters: int = 192,
        num_blocks: int = 15,
        num_actions: int = POLICY_SIZE,
        policy_filters: int = 2,
        value_filters: int = 1,
        value_hidden: int = 256
    ):
        super().__init__()
        self.input_channels = input_channels
        self.num_filters = num_filters
        self.num_blocks = num_blocks

        # Input convolution
        self.input_conv = ConvBlock(input_channels, num_filters)

        # Residual tower
        self.residual_tower = nn.Sequential(
            *[ResidualBlock(num_filters) for _ in range(num_blocks)]
        )

        # Output heads
        self.policy_head = PolicyHead(num_filters, policy_filters, num_actions)
        self.value_head = ValueHead(num_filters, value_filters, value_hidden)

    def forward(self, x, mask=None):
        # Shared trunk
        x = self.input_conv(x)
        x = self.residual_tower(x)

        # Policy head
        policy_logits = self.policy_head(x)

        if mask is not None:
            policy_logits = policy_logits.masked_fill(mask == 0, -1e4)

        # Value head
        value = self.value_head(x)

        return policy_logits, value


def load_model(checkpoint_path):
    """Load trained model from checkpoint (supports both old and new formats)."""
    print(f"\nLoading model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Detect checkpoint format
    version = checkpoint.get('version', '1.0')
    backend = checkpoint.get('backend', 'unknown')
    print(f"Checkpoint version: {version}, backend: {backend}")

    # Get model config (support both old and new formats)
    if 'config' in checkpoint:
        config = checkpoint['config']
        num_filters = config.get('num_filters', 192)
        num_blocks = config.get('num_blocks', 15)
        input_channels = config.get('input_channels', 123)
        policy_filters = config.get('policy_filters', 2)
        value_hidden = config.get('value_hidden', 256)
    else:
        # Old format
        num_filters = checkpoint.get('num_filters', 192)
        num_blocks = checkpoint.get('num_blocks', 15)
        input_channels = 119  # Old format used 119 channels
        policy_filters = 32  # Old format used 32 policy filters
        value_hidden = 256

    print(f"Model config: {num_filters} filters, {num_blocks} blocks, {input_channels} channels")

    # Create model
    model = AlphaZeroNet(
        input_channels=input_channels,
        num_filters=num_filters,
        num_blocks=num_blocks,
        policy_filters=policy_filters,
        value_hidden=value_hidden
    )

    # Load state dict (support both old and new key names)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'network_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['network_state_dict'])
    else:
        raise KeyError("Checkpoint missing model state dict")

    model.eval()

    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Convert model to channels_last memory format for better conv performance
    if device.type == 'cuda':
        model = model.to(memory_format=torch.channels_last)

    print(f"Model loaded on {device} (channels_last: {device.type == 'cuda'})")

    return model, device, input_channels


def test_single_game_performance(model, device, input_channels=123, num_simulations=800,
                                  search_batch=64, c_puct=1.5):
    """Test MCTS performance for a single game with real NN."""
    print(f"\n{'='*70}")
    print(f"TEST 1: Single Game MCTS Performance ({num_simulations} simulations)")
    print(f"{'='*70}")
    print(f"  Using {input_channels}-channel encoding")
    print(f"  search_batch: {search_batch}, c_puct: {c_puct}")

    # Create MCTS
    mcts = alphazero_cpp.BatchedMCTSSearch(
        num_simulations=num_simulations,
        batch_size=search_batch,
        c_puct=c_puct
    )

    # Starting position
    board = chess.Board()
    fen = board.fen()

    # Get initial evaluation from NN
    obs = alphazero_cpp.encode_position(fen)
    # C++ outputs NHWC (8, 8, 123), convert to NCHW logical/NHWC physical format
    obs_tensor = torch.from_numpy(obs).permute(2, 0, 1).unsqueeze(0).to(device)
    if device.type == 'cuda':
        obs_tensor = obs_tensor.contiguous(memory_format=torch.channels_last)

    with torch.no_grad():
        policy_logits, value = model(obs_tensor)
        policy = torch.softmax(policy_logits, dim=1).cpu().numpy()[0]
        value = value.cpu().item()

    # Initialize search
    mcts.init_search(fen, policy.astype(np.float32), value)

    # Run search with timing
    total_leaves = 0
    total_nn_time = 0.0
    total_mcts_time = 0.0
    iteration = 0

    start_time = time.time()

    while not mcts.is_complete():
        # Collect leaves (CPU work)
        mcts_start = time.time()
        num_leaves, obs_batch, masks = mcts.collect_leaves()
        mcts_time = time.time() - mcts_start

        if num_leaves == 0:
            break

        total_leaves += num_leaves
        total_mcts_time += mcts_time

        # NN evaluation (GPU work)
        nn_start = time.time()
        # C++ outputs NHWC (N, 8, 8, 122), convert to NCHW for PyTorch
        obs_tensor = torch.from_numpy(obs_batch[:num_leaves]).permute(0, 3, 1, 2).to(device)
        if device.type == 'cuda':
            obs_tensor = obs_tensor.contiguous(memory_format=torch.channels_last)

        with torch.no_grad():
            policy_logits, values = model(obs_tensor)
            policies = torch.softmax(policy_logits, dim=1).cpu().numpy()
            values = values.cpu().numpy().flatten()

        nn_time = time.time() - nn_start
        total_nn_time += nn_time

        # Update leaves
        mcts.update_leaves(policies.astype(np.float32), values.astype(np.float32))

        iteration += 1

    total_time = time.time() - start_time

    # Get results
    visit_counts = mcts.get_visit_counts()

    # Print results
    print(f"\nResults:")
    print(f"  Total time:           {total_time:.3f}s")
    print(f"  MCTS time (CPU):      {total_mcts_time:.3f}s ({total_mcts_time/total_time*100:.1f}%)")
    print(f"  NN time (GPU):        {total_nn_time:.3f}s ({total_nn_time/total_time*100:.1f}%)")
    print(f"  Iterations:           {iteration}")
    print(f"  Total leaves:         {total_leaves}")
    print(f"  Avg leaves/iter:      {total_leaves/iteration:.1f}")
    print(f"  Simulations:          {mcts.get_simulations_completed()}")
    print(f"  Sims/sec:             {num_simulations/total_time:.1f}")
    print(f"  NN evals/sec:         {total_leaves/total_nn_time:.1f}")
    print(f"  Legal moves:          {(visit_counts > 0).sum()}")
    print(f"  Most visited move:    {visit_counts.max()} visits")

    return {
        'total_time': total_time,
        'mcts_time': total_mcts_time,
        'nn_time': total_nn_time,
        'iterations': iteration,
        'total_leaves': total_leaves,
        'sims_per_sec': num_simulations / total_time,
        'nn_evals_per_sec': total_leaves / total_nn_time if total_nn_time > 0 else 0,
    }


def test_batch_encoding_performance():
    """Test batch encoding performance with OpenMP."""
    print(f"\n{'='*70}")
    print(f"TEST 2: Batch Encoding Performance (123 channels)")
    print(f"{'='*70}")

    # Generate test positions
    batch_sizes = [1, 4, 16, 64, 256]
    positions = []

    board = chess.Board()
    for _ in range(256):
        positions.append(board.fen())
        # Make a random legal move
        legal_moves = list(board.legal_moves)
        if legal_moves:
            import random
            board.push(random.choice(legal_moves))
        else:
            board = chess.Board()

    print(f"\nGenerated {len(positions)} test positions")

    for batch_size in batch_sizes:
        fens = positions[:batch_size]

        # Test single position encoding (used for root node)
        if batch_size == 1:
            start = time.time()
            for _ in range(100):
                obs = alphazero_cpp.encode_position(fens[0])
            single_time = (time.time() - start) / 100
            print(f"\nSingle position encoding: {single_time*1000:.3f}ms")
            print(f"  Output shape: {obs.shape}")
            continue

        # For batches, encode individually and measure
        start = time.time()
        for _ in range(10):  # Average over 10 runs
            batch_obs = []
            for fen in fens:
                obs = alphazero_cpp.encode_position(fen)
                batch_obs.append(obs)
            batch_array = np.stack(batch_obs)
        encoding_time = (time.time() - start) / 10

        throughput = batch_size / encoding_time if encoding_time > 0 else 0

        print(f"\nBatch size {batch_size:3d}:")
        print(f"  Time:       {encoding_time*1000:.2f}ms")
        print(f"  Throughput: {throughput:.0f} positions/sec")
        print(f"  Per pos:    {encoding_time/batch_size*1000:.3f}ms")


def test_multiple_searches(model, device, input_channels=123, num_games=5, num_simulations=400,
                           search_batch=64, c_puct=1.5):
    """Test multiple MCTS searches to measure consistency."""
    print(f"\n{'='*70}")
    print(f"TEST 3: Multiple Searches ({num_games} games, {num_simulations} sims each)")
    print(f"{'='*70}")
    print(f"  search_batch: {search_batch}, c_puct: {c_puct}")

    times = []
    sims_per_sec = []

    for game_idx in range(num_games):
        mcts = alphazero_cpp.BatchedMCTSSearch(
            num_simulations=num_simulations,
            batch_size=search_batch,
            c_puct=c_puct
        )

        board = chess.Board()
        fen = board.fen()

        # Get initial evaluation
        obs = alphazero_cpp.encode_position(fen)
        # C++ outputs NHWC (8, 8, 123), convert to NCHW for PyTorch
        obs_tensor = torch.from_numpy(obs).permute(2, 0, 1).unsqueeze(0).to(device)
        if device.type == 'cuda':
            obs_tensor = obs_tensor.contiguous(memory_format=torch.channels_last)

        with torch.no_grad():
            policy_logits, value = model(obs_tensor)
            policy = torch.softmax(policy_logits, dim=1).cpu().numpy()[0]
            value = value.cpu().item()

        mcts.init_search(fen, policy.astype(np.float32), value)

        # Run search
        start_time = time.time()

        while not mcts.is_complete():
            num_leaves, obs_batch, masks = mcts.collect_leaves()
            if num_leaves == 0:
                break

            # C++ outputs NHWC (N, 8, 8, 122), convert to NCHW for PyTorch
            obs_tensor = torch.from_numpy(obs_batch[:num_leaves]).permute(0, 3, 1, 2).to(device)
            if device.type == 'cuda':
                obs_tensor = obs_tensor.contiguous(memory_format=torch.channels_last)

            with torch.no_grad():
                policy_logits, values = model(obs_tensor)
                policies = torch.softmax(policy_logits, dim=1).cpu().numpy()
                values = values.cpu().numpy().flatten()

            mcts.update_leaves(policies.astype(np.float32), values.astype(np.float32))

        elapsed = time.time() - start_time
        times.append(elapsed)
        sims_per_sec.append(num_simulations / elapsed)

        print(f"  Game {game_idx+1}: {elapsed:.3f}s ({num_simulations/elapsed:.1f} sims/sec)")

    print(f"\nStatistics:")
    print(f"  Mean time:      {np.mean(times):.3f}s ± {np.std(times):.3f}s")
    print(f"  Mean sims/sec:  {np.mean(sims_per_sec):.1f} ± {np.std(sims_per_sec):.1f}")
    print(f"  Min/Max:        {np.min(sims_per_sec):.1f} / {np.max(sims_per_sec):.1f} sims/sec")


def test_memory_usage():
    """Test memory usage of MCTS."""
    print(f"\n{'='*70}")
    print(f"TEST 4: Memory Usage")
    print(f"{'='*70}")

    import psutil
    import os

    process = psutil.Process(os.getpid())

    # Baseline memory
    baseline_mb = process.memory_info().rss / 1024 / 1024
    print(f"\nBaseline memory: {baseline_mb:.1f} MB")

    # Create multiple MCTS instances
    mcts_instances = []
    for i in range(10):
        mcts = alphazero_cpp.BatchedMCTSSearch(
            num_simulations=800,
            batch_size=256,
            c_puct=1.5
        )
        mcts_instances.append(mcts)

        if i % 3 == 2:
            current_mb = process.memory_info().rss / 1024 / 1024
            print(f"After {i+1} instances: {current_mb:.1f} MB (+{current_mb-baseline_mb:.1f} MB)")

    final_mb = process.memory_info().rss / 1024 / 1024
    per_instance_mb = (final_mb - baseline_mb) / len(mcts_instances)

    print(f"\nFinal memory: {final_mb:.1f} MB")
    print(f"Per instance: ~{per_instance_mb:.1f} MB")


def find_checkpoint():
    """Find the best available checkpoint."""
    base_dir = Path(__file__).parent.parent.parent / "checkpoints"

    # Priority order for checkpoints
    candidates = [
        "cpp_iter_*_emergency.pt",  # Emergency saves
        "cpp_iter_*.pt",            # Regular C++ training checkpoints
        "checkpoint_*.pt",          # Old format checkpoints
    ]

    for pattern in candidates:
        matches = sorted(base_dir.glob(pattern), reverse=True)
        if matches:
            return matches[0]

    return None


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    if seconds < 0:
        return "0 sec"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    parts = []
    if hours > 0:
        parts.append(f"{hours} hr")
    if minutes > 0:
        parts.append(f"{minutes} min")
    if secs > 0 or not parts:
        parts.append(f"{secs} sec")
    return " ".join(parts)


def test_self_play_performance(model, device, input_channels, args):
    """Test self-play performance with configurable parameters (like train.py)."""
    print(f"\n{'='*70}")
    print(f"TEST 5: Self-Play Performance (train.py simulation)")
    print(f"{'='*70}")
    print(f"  Parameters matching train.py:")
    print(f"    --simulations {args.simulations}")
    print(f"    --search-batch {args.search_batch}")
    print(f"    --c-puct {args.c_puct}")
    print(f"    --games {args.games}")

    total_moves = 0
    total_sims = 0
    total_evals = 0
    game_times = []

    for game_idx in range(args.games):
        mcts = alphazero_cpp.BatchedMCTSSearch(
            num_simulations=args.simulations,
            batch_size=args.search_batch,
            c_puct=args.c_puct
        )

        board = chess.Board()
        game_moves = 0
        game_sims = 0
        game_evals = 0
        game_start = time.time()

        while not board.is_game_over() and game_moves < 200:
            fen = board.fen()

            # Encode position
            obs = alphazero_cpp.encode_position(fen)
            obs_tensor = torch.from_numpy(obs).permute(2, 0, 1).unsqueeze(0).to(device)
            if device.type == 'cuda':
                obs_tensor = obs_tensor.contiguous(memory_format=torch.channels_last)

            # Get legal mask
            mask = np.zeros(POLICY_SIZE, dtype=np.float32)
            move_map = {}
            for move in board.legal_moves:
                idx = alphazero_cpp.move_to_index(move.uci(), fen)
                if 0 <= idx < POLICY_SIZE:
                    mask[idx] = 1.0
                    move_map[idx] = move

            # Root evaluation
            with torch.no_grad():
                policy_logits, value = model(obs_tensor)
                policy = torch.softmax(policy_logits, dim=1).cpu().numpy()[0]
                value = value.cpu().item()
            game_evals += 1

            # MCTS search
            mcts.init_search(fen, policy.astype(np.float32), value)

            while not mcts.is_complete():
                num_leaves, obs_batch, masks = mcts.collect_leaves()
                if num_leaves == 0:
                    break

                obs_tensor = torch.from_numpy(obs_batch[:num_leaves]).permute(0, 3, 1, 2).to(device)
                if device.type == 'cuda':
                    obs_tensor = obs_tensor.contiguous(memory_format=torch.channels_last)

                with torch.no_grad():
                    policy_logits, values = model(obs_tensor)
                    policies = torch.softmax(policy_logits, dim=1).cpu().numpy()
                    values = values.cpu().numpy().flatten()

                mcts.update_leaves(policies.astype(np.float32), values.astype(np.float32))
                game_evals += num_leaves

            game_sims += mcts.get_simulations_completed()

            # Get best move
            visit_counts = mcts.get_visit_counts()
            visit_counts = visit_counts * mask
            if visit_counts.sum() > 0:
                action = np.argmax(visit_counts)
            else:
                action = list(move_map.keys())[0] if move_map else 0

            if action in move_map:
                board.push(move_map[action])
            else:
                board.push(list(board.legal_moves)[0])

            game_moves += 1
            mcts.reset()

        game_time = time.time() - game_start
        game_times.append(game_time)
        total_moves += game_moves
        total_sims += game_sims
        total_evals += game_evals

        moves_per_sec = game_moves / game_time if game_time > 0 else 0
        print(f"  Game {game_idx+1}: {game_moves} moves in {format_duration(game_time)} "
              f"({moves_per_sec:.1f} moves/sec) [{board.result()}]")

    total_time = sum(game_times)
    print(f"\nSummary:")
    print(f"  Total games:      {args.games}")
    print(f"  Total moves:      {total_moves}")
    print(f"  Total time:       {format_duration(total_time)}")
    print(f"  Moves/sec:        {total_moves / total_time:.1f}")
    print(f"  Sims/sec:         {total_sims / total_time:,.0f}")
    print(f"  NN evals/sec:     {total_evals / total_time:,.0f}")
    print(f"  Avg game time:    {format_duration(np.mean(game_times))}")

    return {
        'total_moves': total_moves,
        'total_time': total_time,
        'moves_per_sec': total_moves / total_time,
        'sims_per_sec': total_sims / total_time,
    }


def test_parallel_selfplay_performance(model, device, input_channels, args):
    """Test parallel self-play with cross-game batching (ParallelSelfPlayCoordinator).

    This test measures the performance improvement from batching NN evaluations
    across multiple concurrent games, which significantly improves GPU utilization.

    The key difference from sequential self-play:
    - Sequential: Each game makes individual NN calls (low GPU utilization ~30-40%)
    - Parallel: Leaves from ALL games are batched together (high GPU utilization ~80-90%)
    """
    print(f"\n{'='*70}")
    print(f"TEST 6: Parallel Self-Play Performance (Cross-Game Batching)")
    print(f"{'='*70}")
    print(f"  Parameters:")
    print(f"    --simulations {args.simulations}")
    print(f"    --search-batch {args.search_batch}")
    print(f"    --workers {args.workers}")
    print(f"    --eval-batch {args.eval_batch}")
    print(f"    --batch-timeout-ms {args.batch_timeout_ms}")
    print(f"    --games {args.games}")

    # Check if ParallelSelfPlayCoordinator is available
    if not hasattr(alphazero_cpp, 'ParallelSelfPlayCoordinator'):
        print("\n  ERROR: ParallelSelfPlayCoordinator not available in alphazero_cpp")
        print("  Rebuild the C++ module with parallel self-play support")
        return None

    # Calculate games per worker
    games_per_worker = max(1, args.games // args.workers)
    actual_total_games = games_per_worker * args.workers

    print(f"\n  Running {args.workers} workers × {games_per_worker} games = {actual_total_games} games")

    # Create replay buffer to store results
    replay_buffer = alphazero_cpp.ReplayBuffer(capacity=100000)

    # Create parallel coordinator
    coordinator = alphazero_cpp.ParallelSelfPlayCoordinator(
        num_workers=args.workers,
        games_per_worker=games_per_worker,
        num_simulations=args.simulations,
        mcts_batch_size=args.search_batch,
        gpu_batch_size=args.eval_batch,
        c_puct=args.c_puct,
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.25,
        temperature_moves=30,
        gpu_timeout_ms=args.batch_timeout_ms
    )

    # Set replay buffer
    coordinator.set_replay_buffer(replay_buffer)

    # Track neural network calls for performance measurement
    nn_call_count = 0
    nn_total_positions = 0
    nn_total_time = 0.0

    # Create evaluator callback
    @torch.no_grad()
    def neural_evaluator(obs_array: np.ndarray, mask_array: np.ndarray, batch_size: int):
        """Neural network evaluator callback for C++ coordinator."""
        nonlocal nn_call_count, nn_total_positions, nn_total_time

        nn_start = time.time()
        nn_call_count += 1
        nn_total_positions += batch_size

        # Convert NHWC (batch, 8, 8, 122) to NCHW (batch, 122, 8, 8)
        obs_nchw = np.transpose(obs_array, (0, 3, 1, 2))

        # Move to GPU
        obs_tensor = torch.from_numpy(obs_nchw).float().to(device)
        mask_tensor = torch.from_numpy(mask_array).float().to(device)

        # Convert to channels_last for better performance
        if device.type == 'cuda':
            obs_tensor = obs_tensor.contiguous(memory_format=torch.channels_last)

        # Forward pass with AMP
        if device.type == 'cuda':
            with autocast('cuda'):
                policy_logits, values = model(obs_tensor, mask_tensor)
                policies = F.softmax(policy_logits, dim=1)
        else:
            policy_logits, values = model(obs_tensor, mask_tensor)
            policies = F.softmax(policy_logits, dim=1)

        # Convert back to numpy
        policies_np = policies.cpu().numpy().astype(np.float32)
        values_np = values.squeeze(-1).cpu().numpy().astype(np.float32)

        nn_total_time += time.time() - nn_start
        return policies_np, values_np

    # Run parallel self-play
    print(f"\n  Starting parallel generation...")
    model.eval()
    start_time = time.time()

    result = coordinator.generate_games(neural_evaluator)

    total_time = time.time() - start_time

    # Extract statistics
    if isinstance(result, dict):
        games_completed = result.get('games_completed', actual_total_games)
        total_moves = result.get('total_moves', 0)
        white_wins = result.get('white_wins', 0)
        black_wins = result.get('black_wins', 0)
        draws = result.get('draws', 0)
    else:
        games_completed = len(result) if result else actual_total_games
        total_moves = sum(t.get('num_moves', 0) for t in result) if result else 0
        white_wins = sum(1 for t in result if t.get('result', 0) > 0) if result else 0
        black_wins = sum(1 for t in result if t.get('result', 0) < 0) if result else 0
        draws = sum(1 for t in result if t.get('result', 0) == 0) if result else 0

    # Calculate performance metrics
    moves_per_sec = total_moves / total_time if total_time > 0 else 0
    sims_per_sec = (total_moves * args.simulations) / total_time if total_time > 0 else 0
    avg_batch_size = nn_total_positions / nn_call_count if nn_call_count > 0 else 0
    nn_evals_per_sec = nn_total_positions / nn_total_time if nn_total_time > 0 else 0
    gpu_utilization_estimate = nn_total_time / total_time * 100 if total_time > 0 else 0

    # Print results
    print(f"\n  Results:")
    print(f"    Games completed:    {games_completed} ({white_wins}W / {draws}D / {black_wins}L)")
    print(f"    Total moves:        {total_moves}")
    print(f"    Total time:         {format_duration(total_time)}")
    print(f"    Moves/sec:          {moves_per_sec:.1f}")
    print(f"    Sims/sec:           {sims_per_sec:,.0f}")

    print(f"\n  Neural Network Performance:")
    print(f"    NN calls:           {nn_call_count:,}")
    print(f"    Total positions:    {nn_total_positions:,}")
    print(f"    Avg batch size:     {avg_batch_size:.1f}")
    print(f"    NN time:            {format_duration(nn_total_time)}")
    print(f"    NN evals/sec:       {nn_evals_per_sec:,.0f}")
    print(f"    GPU util estimate:  {gpu_utilization_estimate:.1f}%")

    print(f"\n  Replay Buffer:")
    print(f"    Positions stored:   {replay_buffer.size():,}")

    return {
        'games_completed': games_completed,
        'total_moves': total_moves,
        'total_time': total_time,
        'moves_per_sec': moves_per_sec,
        'sims_per_sec': sims_per_sec,
        'nn_calls': nn_call_count,
        'avg_batch_size': avg_batch_size,
        'nn_evals_per_sec': nn_evals_per_sec,
        'gpu_utilization': gpu_utilization_estimate,
    }


def test_sequential_vs_parallel(model, device, input_channels, args):
    """Compare sequential vs parallel self-play performance.

    Runs both modes with the same parameters and reports the speedup.
    """
    print(f"\n{'='*70}")
    print(f"TEST 7: Sequential vs Parallel Comparison")
    print(f"{'='*70}")

    # Check if parallel is available
    if not hasattr(alphazero_cpp, 'ParallelSelfPlayCoordinator'):
        print("\n  Skipping comparison: ParallelSelfPlayCoordinator not available")
        return None

    # Use smaller game count for comparison
    comparison_games = min(args.games, 3)
    print(f"\n  Running {comparison_games} games in each mode for comparison...")

    # Create a modified args for comparison
    class ComparisonArgs:
        pass

    comp_args = ComparisonArgs()
    comp_args.simulations = args.simulations
    comp_args.search_batch = args.search_batch
    comp_args.c_puct = args.c_puct
    comp_args.games = comparison_games
    comp_args.workers = args.workers
    comp_args.eval_batch = args.eval_batch
    comp_args.batch_timeout_ms = args.batch_timeout_ms

    # Run sequential
    print(f"\n  --- Sequential Mode (1 game at a time) ---")
    seq_start = time.time()
    seq_results = test_self_play_performance(model, device, input_channels, comp_args)
    seq_time = time.time() - seq_start

    # Run parallel
    print(f"\n  --- Parallel Mode ({args.workers} workers, cross-game batching) ---")
    par_start = time.time()
    par_results = test_parallel_selfplay_performance(model, device, input_channels, comp_args)
    par_time = time.time() - par_start

    # Calculate speedup
    if seq_results and par_results and seq_time > 0:
        speedup = seq_time / par_time if par_time > 0 else 0
        moves_speedup = par_results['moves_per_sec'] / seq_results['moves_per_sec'] if seq_results['moves_per_sec'] > 0 else 0

        print(f"\n{'='*70}")
        print(f"  COMPARISON SUMMARY")
        print(f"{'='*70}")
        print(f"  Sequential:")
        print(f"    Total time:     {format_duration(seq_time)}")
        print(f"    Moves/sec:      {seq_results['moves_per_sec']:.1f}")
        print(f"  Parallel ({args.workers} workers):")
        print(f"    Total time:     {format_duration(par_time)}")
        print(f"    Moves/sec:      {par_results['moves_per_sec']:.1f}")
        print(f"    Avg batch size: {par_results['avg_batch_size']:.1f}")
        print(f"    GPU util:       {par_results['gpu_utilization']:.1f}%")
        print(f"\n  Speedup:          {speedup:.2f}x faster with parallel mode")
        print(f"  Throughput gain:  {moves_speedup:.2f}x more moves/sec")
        print(f"{'='*70}")

        return {
            'sequential_time': seq_time,
            'parallel_time': par_time,
            'speedup': speedup,
            'moves_speedup': moves_speedup,
        }

    return None


def main():
    """Run all performance tests."""
    import argparse

    parser = argparse.ArgumentParser(
        description="C++ MCTS Performance Testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic test with defaults
    python test_performance.py

    # Test with specific checkpoint
    python test_performance.py --checkpoint checkpoints/cpp_iter_50.pt

    # Test with same parameters as train.py
    python test_performance.py --simulations 800 --search-batch 64 --c-puct 1.5

    # Quick test with fewer simulations
    python test_performance.py --simulations 200 --games 3

    # Test parallel self-play (cross-game batching)
    python test_performance.py --test parallel --workers 4 --eval-batch 512

    # Compare sequential vs parallel performance
    python test_performance.py --test compare --workers 4 --games 3

    # Test with custom network (if checkpoint has different config)
    python test_performance.py --filters 128 --blocks 10
        """
    )

    # Checkpoint
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint file")

    # MCTS parameters (same as train.py)
    parser.add_argument("--simulations", type=int, default=800,
                        help="MCTS simulations per move (default: 800)")
    parser.add_argument("--search-batch", type=int, default=64,
                        help="Leaves collected per MCTS iteration (default: 64)")
    parser.add_argument("--c-puct", type=float, default=1.5,
                        help="MCTS exploration constant (default: 1.5)")

    # Network parameters (override checkpoint config)
    parser.add_argument("--filters", type=int, default=None,
                        help="Network filters (default: from checkpoint)")
    parser.add_argument("--blocks", type=int, default=None,
                        help="Residual blocks (default: from checkpoint)")

    # Test parameters
    parser.add_argument("--games", type=int, default=5,
                        help="Number of games for self-play test (default: 5)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device: cuda or cpu (default: cuda)")

    # Parallel self-play parameters
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel self-play workers (default: 4)")
    parser.add_argument("--eval-batch", type=int, default=512,
                        help="Max positions per GPU call in parallel mode (default: 512)")
    parser.add_argument("--batch-timeout-ms", type=int, default=5,
                        help="Max wait time to fill GPU batch in ms (default: 5)")

    # Test selection
    parser.add_argument("--test", type=str, default="all",
                        choices=["all", "single", "encoding", "multiple", "memory", "selfplay", "parallel", "compare"],
                        help="Which test to run (default: all)")

    args = parser.parse_args()

    print("=" * 70)
    print("C++ MCTS Performance Testing with Real Trained Model")
    print("=" * 70)

    # Handle device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU")
        args.device = "cpu"

    # Find checkpoint
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
    else:
        checkpoint_path = find_checkpoint()

    if checkpoint_path is None or not checkpoint_path.exists():
        print("ERROR: No checkpoint found")
        print("  Provide one with --checkpoint or train a model first:")
        print("  uv run python alphazero-cpp/scripts/train.py --iterations 2 --games-per-iter 5")
        sys.exit(1)

    print(f"Using checkpoint: {checkpoint_path}")

    # Load model
    model, device, input_channels = load_model(checkpoint_path)

    # Print test configuration
    print(f"\nTest Configuration:")
    print(f"  Device:           {device}")
    print(f"  Input channels:   {input_channels}")
    print(f"  Simulations:      {args.simulations}")
    print(f"  Search batch:     {args.search_batch}")
    print(f"  C-PUCT:           {args.c_puct}")
    print(f"  Test games:       {args.games}")
    print(f"  Parallel workers: {args.workers}")
    print(f"  Eval batch:       {args.eval_batch}")
    print(f"  Batch timeout:    {args.batch_timeout_ms}ms")

    # Run selected tests
    if args.test in ["all", "single"]:
        test_single_game_performance(
            model, device, input_channels,
            num_simulations=args.simulations,
            search_batch=args.search_batch,
            c_puct=args.c_puct
        )

    if args.test in ["all", "encoding"]:
        test_batch_encoding_performance()

    if args.test in ["all", "multiple"]:
        test_multiple_searches(
            model, device, input_channels,
            num_games=args.games,
            num_simulations=args.simulations // 2,
            search_batch=args.search_batch,
            c_puct=args.c_puct
        )

    if args.test in ["all", "memory"]:
        test_memory_usage()

    if args.test in ["all", "selfplay"]:
        test_self_play_performance(model, device, input_channels, args)

    if args.test in ["all", "parallel"]:
        test_parallel_selfplay_performance(model, device, input_channels, args)

    if args.test == "compare":
        test_sequential_vs_parallel(model, device, input_channels, args)

    print(f"\n{'='*70}")
    print("All Performance Tests Complete!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
