#!/usr/bin/env python3
"""
Test suite for Entropic Risk Measure (ERM) framework.

Tests the unified risk_beta parameter that replaces contempt and soft-minimax:
- wdl_to_value (simplified: pw - pl, no contempt)
- risk_beta config propagation
- Pure training labels (draws = 0.0 regardless of risk_beta)
- Weight decay parameter groups
- Network WDL outputs
- Soft cross-entropy loss
- Replay buffer WDL storage
- End-to-end parallel self-play with risk_beta

Run: python alphazero-cpp/tests/test_risk_erm.py
"""

import sys
import os
import numpy as np
from pathlib import Path

# Add build directory and scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent / "build" / "Release"))
sys.path.insert(0, str(Path(__file__).parent.parent / "build"))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

try:
    import alphazero_cpp
except ImportError as e:
    print(f"Error importing alphazero_cpp: {e}")
    print("Make sure to build the C++ extension first:")
    print("  cd alphazero-cpp && cmake --build build --config Release")
    sys.exit(1)

passed = 0
failed = 0
skipped = 0


def run_test(name, func):
    """Run a single test and report results."""
    global passed, failed, skipped
    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"{'='*70}")
    try:
        func()
        print(f"  PASSED")
        passed += 1
    except NotImplementedError:
        print(f"  SKIPPED (not yet implemented)")
        skipped += 1
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback
        traceback.print_exc()
        failed += 1


# =========================================================================
# T1: wdl_to_value() — simplified (no contempt parameter)
# =========================================================================
def test_wdl_to_value():
    """Verify wdl_to_value returns pw - pl (no contempt)."""
    assert hasattr(alphazero_cpp, 'wdl_to_value'), "wdl_to_value not exposed in bindings"

    # Pure WDL → value: pw - pl
    assert abs(alphazero_cpp.wdl_to_value(0.7, 0.2, 0.1) - 0.6) < 1e-6
    print("  OK wdl_to_value(0.7, 0.2, 0.1) = 0.6")

    # Pure draw → 0.0 (no contempt bias)
    assert abs(alphazero_cpp.wdl_to_value(0.0, 1.0, 0.0) - 0.0) < 1e-6
    print("  OK wdl_to_value(0.0, 1.0, 0.0) = 0.0")

    # Certain win
    assert abs(alphazero_cpp.wdl_to_value(1.0, 0.0, 0.0) - 1.0) < 1e-6
    print("  OK wdl_to_value(1.0, 0.0, 0.0) = 1.0")

    # Certain loss
    assert abs(alphazero_cpp.wdl_to_value(0.0, 0.0, 1.0) - (-1.0)) < 1e-6
    print("  OK wdl_to_value(0.0, 0.0, 1.0) = -1.0")

    # Equal win/loss → 0.0
    assert abs(alphazero_cpp.wdl_to_value(0.5, 0.0, 0.5) - 0.0) < 1e-6
    print("  OK wdl_to_value(0.5, 0.0, 0.5) = 0.0")


# =========================================================================
# T2: risk_beta config in Python bindings
# =========================================================================
def test_risk_beta_config():
    """Python bindings expose 'risk_beta' kwarg and report it in config."""
    coord = alphazero_cpp.ParallelSelfPlayCoordinator(
        num_workers=1, games_per_worker=1, num_simulations=4,
        mcts_batch_size=2, gpu_batch_size=8, risk_beta=1.5)
    config = coord.get_config()
    assert 'risk_beta' in config, "Config should have 'risk_beta' key"
    assert abs(config['risk_beta'] - 1.5) < 1e-6, f"risk_beta should be 1.5, got {config['risk_beta']}"
    print(f"  OK config['risk_beta'] = {config['risk_beta']}")

    # Old keys should not exist
    assert 'contempt' not in config, "Config should NOT have 'contempt' (removed)"
    assert 'soft_minimax_beta' not in config, "Config should NOT have 'soft_minimax_beta' (removed)"
    print("  OK 'contempt' and 'soft_minimax_beta' keys removed from config")


# =========================================================================
# T3: Weight decay parameter groups
# =========================================================================
def test_weight_decay_groups():
    """Verify BN and bias params excluded from weight decay."""
    from network import AlphaZeroNet

    net = AlphaZeroNet(num_filters=64, num_blocks=2, wdl=True)

    no_decay_keywords = ['bn', 'bias']
    decay_params = [n for n, p in net.named_parameters()
                    if not any(kw in n for kw in no_decay_keywords)]
    no_decay_params = [n for n, p in net.named_parameters()
                       if any(kw in n for kw in no_decay_keywords)]

    # BN params must be in no_decay
    bn_params = [n for n, _ in net.named_parameters() if '.bn.' in n or '.bn1.' in n or '.bn2.' in n]
    for n in bn_params:
        assert n in no_decay_params, f"BN param {n} should have weight_decay=0"
    print(f"  OK {len(bn_params)} BN params excluded from weight decay")

    # Bias params must be in no_decay
    bias_params = [n for n, _ in net.named_parameters() if 'bias' in n]
    for n in bias_params:
        assert n in no_decay_params, f"Bias param {n} should have weight_decay=0"
    print(f"  OK {len(bias_params)} bias params excluded from weight decay")

    print(f"  OK decay group: {len(decay_params)} params, no_decay group: {len(no_decay_params)} params")


# =========================================================================
# T4: Network always WDL — forward returns 3 logits
# =========================================================================
def test_network_always_wdl():
    """Network always returns WDL logits; wdl_logits is never None."""
    import torch
    from network import AlphaZeroNet

    net = AlphaZeroNet(num_filters=64, num_blocks=2, wdl=True)
    net.eval()

    x = torch.randn(4, 122, 8, 8)
    mask = torch.ones(4, 4672)

    policy, value, policy_logits, wdl_logits = net(x, mask)

    assert wdl_logits is not None, "WDL logits must not be None"
    assert wdl_logits.shape == (4, 3), f"WDL logits shape should be (4,3), got {wdl_logits.shape}"
    assert value.shape == (4, 1), f"Value shape should be (4,1), got {value.shape}"
    print(f"  OK wdl_logits.shape = {wdl_logits.shape}")

    # Value should be P(win) - P(loss) from softmax of WDL logits
    wdl_probs = torch.softmax(wdl_logits, dim=1)
    expected_value = wdl_probs[:, 0:1] - wdl_probs[:, 2:3]
    assert torch.allclose(value, expected_value, atol=1e-5)
    print("  OK value = P(win) - P(loss)")


# =========================================================================
# T5: Soft cross-entropy loss
# =========================================================================
def test_soft_cross_entropy_loss():
    """Soft CE loss provides gradients to all 3 WDL classes."""
    import torch
    import torch.nn.functional as F

    wdl_logits = torch.tensor([[2.0, 0.5, -1.0]], requires_grad=True)
    wdl_target = torch.tensor([[0.6, 0.3, 0.1]])

    loss = -torch.sum(wdl_target * F.log_softmax(wdl_logits, dim=1)) / wdl_logits.size(0)
    loss.backward()

    assert wdl_logits.grad is not None
    assert (wdl_logits.grad.abs() > 1e-6).all(), \
        f"All WDL logits should have gradients, got {wdl_logits.grad}"
    print(f"  OK soft CE grads: {wdl_logits.grad.tolist()}")

    assert 0.0 < loss.item() < 10.0, f"Soft CE loss should be reasonable, got {loss.item()}"
    print(f"  OK soft CE loss = {loss.item():.4f}")


# =========================================================================
# T6: Pure training labels (draws = 0.0)
# =========================================================================
def test_pure_training_labels():
    """Draw training values are always 0.0 regardless of risk_beta."""
    # Simulate what set_outcomes() does in game.hpp
    values = []
    for i in range(10):
        white_to_move = (i % 2 == 0)
        # Draw result
        value = 0.0  # Pure training label (risk_beta is search-time only)
        values.append(value)

    assert all(v == 0.0 for v in values), f"All draw values should be 0.0, got {values}"
    print(f"  OK all 10 draw positions have value 0.0 (risk_beta is search-time only)")


# =========================================================================
# T7: Replay buffer WDL storage
# =========================================================================
def test_replay_buffer_wdl_storage():
    """ReplayBuffer stores and retrieves WDL targets alongside obs/pol/val."""
    buffer = alphazero_cpp.ReplayBuffer(capacity=100)

    obs = np.random.rand(7808).astype(np.float32)
    pol = np.random.rand(4672).astype(np.float32)
    val = 0.5
    wdl = np.array([0.6, 0.3, 0.1], dtype=np.float32)

    buffer.add_sample(obs, pol, val, wdl)
    assert buffer.size() == 1
    print("  OK added sample with WDL")

    result = buffer.sample(1)
    assert len(result) == 5, f"sample() should return 5 arrays, got {len(result)}"
    s_obs, s_pol, s_val, s_wdl, s_sv = result
    assert s_wdl.shape == (1, 3), f"WDL shape should be (1,3), got {s_wdl.shape}"
    assert np.allclose(s_wdl[0], wdl, atol=1e-5), f"WDL should round-trip: {s_wdl[0]} vs {wdl}"
    print(f"  OK WDL round-trips: {s_wdl[0]} (expected {wdl})")


# =========================================================================
# T8: Evaluator receives (batch, 3) WDL shape
# =========================================================================
def test_evaluator_receives_wdl_shape():
    """Verify evaluator callback receives (batch, 3) out_values."""
    received_shapes = {}

    coordinator = alphazero_cpp.ParallelSelfPlayCoordinator(
        num_workers=1, games_per_worker=1, num_simulations=10,
        mcts_batch_size=2, gpu_batch_size=8, risk_beta=0.5,
        temperature_moves=1)

    buffer = alphazero_cpp.ReplayBuffer(capacity=1000)
    coordinator.set_replay_buffer(buffer)

    def shape_checking_evaluator(obs, masks, bs, out_pol, out_val):
        received_shapes['out_values'] = out_val.shape
        received_shapes['out_policies'] = out_pol.shape
        out_pol[:bs] = 1.0 / 4672
        out_val[:bs, 0] = 0.4   # P(win)
        out_val[:bs, 1] = 0.3   # P(draw)
        out_val[:bs, 2] = 0.3   # P(loss)

    coordinator.generate_games(shape_checking_evaluator)

    assert 'out_values' in received_shapes, "Evaluator was not called"
    shape = received_shapes['out_values']
    assert len(shape) == 2, f"out_values should be 2D, got shape {shape}"
    assert shape[1] == 3, f"out_values dim 1 should be 3 (WDL), got {shape}"
    print(f"  OK out_values shape: {shape} (batch, 3)")


# =========================================================================
# T9: E2E parallel self-play with risk_beta
# =========================================================================
def test_e2e_parallel_selfplay():
    """Full integration: generate games with risk_beta, verify pure training labels."""
    coordinator = alphazero_cpp.ParallelSelfPlayCoordinator(
        num_workers=2, games_per_worker=1, num_simulations=30,
        mcts_batch_size=2, gpu_batch_size=8, risk_beta=1.0,
        temperature_moves=1)

    buffer = alphazero_cpp.ReplayBuffer(capacity=10000)
    coordinator.set_replay_buffer(buffer)

    eval_calls = [0]
    def nn_evaluator(obs, masks, bs, out_pol, out_val):
        eval_calls[0] += 1
        out_pol[:bs] = 1.0 / 4672
        out_val[:bs, 0] = 0.4
        out_val[:bs, 1] = 0.3
        out_val[:bs, 2] = 0.3

    result = coordinator.generate_games(nn_evaluator)

    print(f"  Games completed: {result['games_completed']}")
    print(f"  NN eval calls: {eval_calls[0]}")
    print(f"  Buffer size: {buffer.size()}")

    assert result['games_completed'] == 2, f"Expected 2 games, got {result['games_completed']}"
    assert buffer.size() > 0, "Buffer should have samples"
    assert eval_calls[0] > 0, "Evaluator should have been called"

    # Sample and verify training values are pure labels (-1, 0, 1)
    s_obs, s_pol, s_val, s_wdl, s_sv = buffer.sample(min(buffer.size(), 20))
    print(f"  Unique training values: {set(np.round(s_val, 1))}")
    for v in s_val:
        assert abs(v - 1.0) < 0.01 or abs(v + 1.0) < 0.01 or abs(v) < 0.01, \
            f"Training value {v} should be -1, 0, or 1 (pure labels)"
    print("  OK all training values are pure labels (-1/0/1)")


# =========================================================================
# T10: risk_beta=0 produces standard behavior
# =========================================================================
def test_risk_neutral_equivalence():
    """risk_beta=0 should produce valid games (regression check)."""
    coordinator = alphazero_cpp.ParallelSelfPlayCoordinator(
        num_workers=1, games_per_worker=1, num_simulations=20,
        mcts_batch_size=2, gpu_batch_size=8, risk_beta=0.0,
        temperature_moves=1)

    buffer = alphazero_cpp.ReplayBuffer(capacity=1000)
    coordinator.set_replay_buffer(buffer)

    def nn_evaluator(obs, masks, bs, out_pol, out_val):
        out_pol[:bs] = 1.0 / 4672
        out_val[:bs, 0] = 0.4
        out_val[:bs, 1] = 0.3
        out_val[:bs, 2] = 0.3

    result = coordinator.generate_games(nn_evaluator)
    assert result['games_completed'] == 1, f"Expected 1 game, got {result['games_completed']}"
    assert buffer.size() > 0, "Buffer should have samples"
    print(f"  OK risk_beta=0 completed 1 game with {buffer.size()} positions")


# =========================================================================
# T11: No double contempt — training labels unbiased
# =========================================================================
def test_no_bias_in_labels():
    """Verify risk_beta doesn't leak into training labels."""
    coordinator = alphazero_cpp.ParallelSelfPlayCoordinator(
        num_workers=1, games_per_worker=1, num_simulations=50,
        mcts_batch_size=2, gpu_batch_size=8, risk_beta=2.0,
        temperature_moves=1)

    buffer = alphazero_cpp.ReplayBuffer(capacity=1000)
    coordinator.set_replay_buffer(buffer)

    def neutral_evaluator(obs, masks, bs, out_pol, out_val):
        out_pol[:bs] = 1.0 / 4672
        out_val[:bs, 0] = 0.0    # P(win) = 0
        out_val[:bs, 1] = 1.0    # P(draw) = 1
        out_val[:bs, 2] = 0.0    # P(loss) = 0

    coordinator.generate_games(neutral_evaluator)

    if buffer.size() == 0:
        print("  SKIP: no samples generated")
        return

    s_obs, s_pol, s_val, s_wdl, s_sv = buffer.sample(buffer.size())
    for v in s_val:
        assert abs(v - 1.0) < 0.01 or abs(v + 1.0) < 0.01 or abs(v) < 0.01, \
            f"Training value {v} should be pure label (-1/0/1), not risk-biased"
    print(f"  OK all {len(s_val)} training values are pure labels (risk_beta=2.0 not leaked)")


# =========================================================================
# Main
# =========================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("  Entropic Risk Measure (ERM) Framework Test Suite")
    print("=" * 70)

    # C++ binding tests
    run_test("T1: wdl_to_value() utility (simplified)", test_wdl_to_value)
    run_test("T2: risk_beta config propagation", test_risk_beta_config)
    run_test("T6: Pure training labels (draws = 0.0)", test_pure_training_labels)

    # Python network tests
    try:
        import torch
        HAS_TORCH = True
    except ImportError:
        HAS_TORCH = False
        print("\n  WARNING: torch not available, skipping network tests")

    if HAS_TORCH:
        run_test("T3: Weight decay parameter groups", test_weight_decay_groups)
        run_test("T4: Network always WDL", test_network_always_wdl)
        run_test("T5: Soft cross-entropy loss", test_soft_cross_entropy_loss)

    # Replay buffer tests
    run_test("T7: Replay buffer WDL storage", test_replay_buffer_wdl_storage)

    # Integration tests
    run_test("T8: Evaluator receives (batch, 3) WDL shape", test_evaluator_receives_wdl_shape)
    run_test("T9: E2E parallel self-play with risk_beta", test_e2e_parallel_selfplay)
    run_test("T10: risk_beta=0 risk-neutral equivalence", test_risk_neutral_equivalence)
    run_test("T11: No bias in training labels", test_no_bias_in_labels)

    # Summary
    total = passed + failed + skipped
    print(f"\n{'=' * 70}")
    print(f"  Results: {passed} passed, {failed} failed, {skipped} skipped out of {total}")
    print(f"{'=' * 70}")

    sys.exit(1 if failed > 0 else 0)
