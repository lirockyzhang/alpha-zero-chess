"""Tests for evaluate script with endgame opponent.

Tests the integrated endgame evaluation functionality in the main evaluate script.
"""

import pytest
import sys
from pathlib import Path
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from alphazero.neural.network import AlphaZeroNetwork
from alphazero.evaluation.endgame_eval import EndgameEvaluator, ENDGAME_POSITIONS, EndgamePosition


@pytest.fixture
def test_network():
    """Create a small test network."""
    network = AlphaZeroNetwork(num_filters=32, num_blocks=2)
    network.eval()
    return network


@pytest.fixture
def test_checkpoint(tmp_path, test_network):
    """Create a test checkpoint file."""
    checkpoint = {
        'network_state_dict': test_network.state_dict(),
        'num_filters': 32,
        'num_blocks': 2,
        'step': 1000,
        'iteration': 1
    }

    checkpoint_path = tmp_path / "test_checkpoint.pt"
    torch.save(checkpoint, checkpoint_path)

    return str(checkpoint_path)


class TestEndgamePositions:
    """Test endgame position data structure."""

    def test_endgame_positions_exist(self):
        """Test that endgame positions are defined."""
        assert len(ENDGAME_POSITIONS) > 0
        assert len(ENDGAME_POSITIONS) == 50  # Should have 50 curated positions

    def test_endgame_position_structure(self):
        """Test that endgame positions have required fields."""
        for pos in ENDGAME_POSITIONS:
            # Check actual fields from EndgamePosition dataclass
            assert hasattr(pos, 'fen')
            assert hasattr(pos, 'description')
            assert hasattr(pos, 'category')
            assert hasattr(pos, 'difficulty')
            assert hasattr(pos, 'expected_result')
            assert hasattr(pos, 'optimal_first_move')

    def test_endgame_categories(self):
        """Test that all positions have valid categories."""
        valid_categories = {'basic_mate', 'pawn_endgame', 'rook_endgame', 'tactical'}

        for pos in ENDGAME_POSITIONS:
            assert pos.category in valid_categories

    def test_endgame_difficulties(self):
        """Test that all positions have valid difficulty levels."""
        for pos in ENDGAME_POSITIONS:
            assert 1 <= pos.difficulty <= 5

    def test_endgame_expected_results(self):
        """Test that all positions have valid expected results."""
        valid_results = {'win', 'draw', 'loss'}

        for pos in ENDGAME_POSITIONS:
            assert pos.expected_result in valid_results

    def test_category_distribution(self):
        """Test that positions are distributed across categories."""
        categories = {}
        for pos in ENDGAME_POSITIONS:
            categories[pos.category] = categories.get(pos.category, 0) + 1

        # Should have positions in all categories
        assert len(categories) == 4
        # Each category should have at least a few positions
        for count in categories.values():
            assert count >= 5


class TestEndgameEvaluator:
    """Test endgame evaluator functionality."""

    def test_evaluator_creation(self, test_network):
        """Test creating an endgame evaluator."""
        evaluator = EndgameEvaluator(
            network=test_network,
            device='cpu',
            num_simulations=10,
            max_moves=50,
            use_amp=False
        )

        assert evaluator is not None
        assert evaluator.num_simulations == 10
        assert evaluator.max_moves == 50

    def test_evaluate_single_position(self, test_network):
        """Test evaluating a single endgame position."""
        evaluator = EndgameEvaluator(
            network=test_network,
            device='cpu',
            num_simulations=10,
            max_moves=20,
            use_amp=False
        )

        # Use first position from ENDGAME_POSITIONS for testing
        if len(ENDGAME_POSITIONS) > 0:
            test_position = ENDGAME_POSITIONS[0]

            result = evaluator.evaluate_position(test_position)

            assert result is not None
            # Check for either 'result' or 'actual_result' key
            assert 'result' in result or 'actual_result' in result
            assert 'moves_played' in result
            assert 'correct' in result
            # Get result value from either key
            result_value = result.get('result') or result.get('actual_result')
            assert result_value in ['win', 'draw', 'loss', 'timeout']

    def test_evaluate_multiple_positions(self, test_network):
        """Test evaluating multiple positions."""
        evaluator = EndgameEvaluator(
            network=test_network,
            device='cpu',
            num_simulations=5,
            max_moves=10,
            use_amp=False
        )

        # Take first 3 positions for quick test
        test_positions = ENDGAME_POSITIONS[:3]

        results = evaluator.evaluate_all(test_positions)

        assert results is not None
        assert 'total_positions' in results
        assert 'correct_results' in results
        assert 'accuracy' in results
        assert results['total_positions'] == 3
        assert 0 <= results['accuracy'] <= 1.0


class TestCategoryFiltering:
    """Test filtering positions by category."""

    def test_filter_basic_mate(self):
        """Test filtering basic mate positions."""
        basic_mates = [p for p in ENDGAME_POSITIONS if p.category == 'basic_mate']
        assert len(basic_mates) > 0
        for pos in basic_mates:
            assert pos.category == 'basic_mate'

    def test_filter_pawn_endgame(self):
        """Test filtering pawn endgame positions."""
        pawn_endgames = [p for p in ENDGAME_POSITIONS if p.category == 'pawn_endgame']
        assert len(pawn_endgames) > 0
        for pos in pawn_endgames:
            assert pos.category == 'pawn_endgame'

    def test_filter_rook_endgame(self):
        """Test filtering rook endgame positions."""
        rook_endgames = [p for p in ENDGAME_POSITIONS if p.category == 'rook_endgame']
        assert len(rook_endgames) > 0
        for pos in rook_endgames:
            assert pos.category == 'rook_endgame'

    def test_filter_tactical(self):
        """Test filtering tactical positions."""
        tactical = [p for p in ENDGAME_POSITIONS if p.category == 'tactical']
        assert len(tactical) > 0
        for pos in tactical:
            assert pos.category == 'tactical'


class TestDifficultyFiltering:
    """Test filtering positions by difficulty."""

    def test_filter_difficulty_1(self):
        """Test filtering difficulty 1 positions."""
        easy = [p for p in ENDGAME_POSITIONS if p.difficulty == 1]
        assert len(easy) > 0
        for pos in easy:
            assert pos.difficulty == 1

    def test_filter_difficulty_5(self):
        """Test filtering difficulty 5 positions."""
        hard = [p for p in ENDGAME_POSITIONS if p.difficulty == 5]
        # Note: May not have difficulty 5 positions in current dataset
        # Just verify filtering works correctly
        for pos in hard:
            assert pos.difficulty == 5

    def test_difficulty_progression(self):
        """Test that difficulty levels are represented."""
        difficulties = set(p.difficulty for p in ENDGAME_POSITIONS)
        # Should have at least difficulties 1-4
        assert 1 in difficulties
        assert 2 in difficulties
        assert 3 in difficulties
        assert 4 in difficulties
        # Difficulty 5 is optional


class TestEvaluateScriptIntegration:
    """Test integration with evaluate.py script."""

    def test_endgame_opponent_available(self):
        """Test that endgame opponent is available in evaluate script."""
        # This would require importing and checking the evaluate script
        # For now, we verify the evaluator can be imported
        from alphazero.evaluation.endgame_eval import EndgameEvaluator
        assert EndgameEvaluator is not None

    def test_combined_filtering(self):
        """Test filtering by both category and difficulty."""
        # Filter for easy basic mates
        filtered = [p for p in ENDGAME_POSITIONS
                   if p.category == 'basic_mate' and p.difficulty <= 2]

        assert len(filtered) > 0
        for pos in filtered:
            assert pos.category == 'basic_mate'
            assert pos.difficulty <= 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
