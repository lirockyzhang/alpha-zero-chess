# AlphaZero Chess Training on Google Colab

Train AlphaZero chess models on Google Colab with GPU acceleration (T4 or A100).

## Quick Start

### 1. Upload to Google Colab

1. Go to [Google Colab](https://colab.research.google.com/)
2. Click **File → Upload notebook**
3. Upload `train_alphazero.ipynb` from this folder
4. Select **Runtime → Change runtime type → GPU** (T4 for free tier, A100 for Colab Pro+)

### 2. Run the Notebook

Simply run all cells in order:
1. **Setup & GPU Check** - Verifies GPU availability
2. **Mount Google Drive** - Saves checkpoints to your Drive
3. **Clone Repository** - Gets the AlphaZero code
4. **Configuration** - Choose T4 or A100 settings
5. **Training** - Runs iterative training with progress bars
6. **Evaluation** - Tests the trained model
7. **Play Interactive Game** - Play against your model

### 3. Monitor Training

The notebook displays:
- **Games/hour** - Self-play generation speed
- **Loss metrics** - Policy loss, value loss, total loss
- **Buffer size** - Number of training positions
- **Iteration progress** - Current iteration and step

Training automatically saves checkpoints to Google Drive every 50-100 steps.

## Configuration Presets

### A100 (40GB) - Long Training (Colab Pro+)
**Best for serious training runs (8-12 hours)**
```python
ITERATIONS = 5              # 5 training iterations
STEPS_PER_ITERATION = 4000  # 4000 steps per iteration (20k total)
NUM_ACTORS = 2              # 2 sequential actors
NUM_FILTERS = 192           # Large network
NUM_BLOCKS = 15
BATCH_SIZE = 8192           # Large batch for A100
SIMULATIONS = 400           # High-quality MCTS
MIN_BUFFER_SIZE = 8192
CHECKPOINT_INTERVAL = 100
```
**Expected results**: Strong tactical play, ~1800-2000 Elo after full training

### A100 (40GB) - Short Training (2-3 hours)
**Good for testing and quick experiments**
```python
ITERATIONS = 3
STEPS_PER_ITERATION = 1500  # 4500 steps total
NUM_ACTORS = 2
NUM_FILTERS = 128
NUM_BLOCKS = 10
BATCH_SIZE = 4096
SIMULATIONS = 200
MIN_BUFFER_SIZE = 4096
CHECKPOINT_INTERVAL = 50
```
**Expected results**: Beats random play, learns basic tactics

### T4 (16GB) - Free Tier
**Optimized for Colab free tier (12-hour limit)**
```python
ITERATIONS = 3
STEPS_PER_ITERATION = 1500  # 4500 steps total
NUM_ACTORS = 1              # Single actor for memory
NUM_FILTERS = 64
NUM_BLOCKS = 5
BATCH_SIZE = 2048
SIMULATIONS = 200
MIN_BUFFER_SIZE = 2048
CHECKPOINT_INTERVAL = 50
```
**Expected results**: Learns basic chess rules, better than random

## Iterative Training Explained

The notebook uses **iterative training** for faster learning:

1. **Iteration 1**: Train on games from random initialization
2. **Iteration 2**: Generate new games with improved model, train on fresh data
3. **Iteration 3+**: Repeat - each iteration uses stronger self-play games

**Why this works**: Early games are weak (random moves). By refreshing the replay buffer each iteration, the model trains on progressively stronger games, accelerating learning.

## Features

### ✅ Single-Process Design
- No multiprocessing (works reliably in Jupyter)
- Sequential actor execution
- No complex process management

### ✅ Google Drive Integration
- Automatic checkpoint saving to Drive
- Resume training after session timeout
- Checkpoints persist across sessions

### ✅ Progress Monitoring
- Real-time progress bars (tqdm)
- Loss metrics and training stats
- Games/hour throughput

### ✅ A100 Optimizations
- `torch.compile()` for 20-30% speedup
- Large batch sizes (8192-16384)
- Mixed precision (FP16) training
- Efficient memory management

### ✅ Self-Contained
- All code inline in notebook
- No external Python files needed
- Easy to modify and experiment

## Troubleshooting

### "CUDA out of memory"
- Reduce `BATCH_SIZE` (try 4096 → 2048)
- Reduce `NUM_FILTERS` (try 128 → 64)
- Reduce `NUM_BLOCKS` (try 10 → 5)
- Use T4 configuration instead of A100

### "Session disconnected"
- Training auto-saves to Google Drive
- Re-run the notebook - it will resume from latest checkpoint
- Consider Colab Pro+ for longer sessions

### "Training is slow"
- Verify GPU is enabled: Runtime → Change runtime type → GPU
- Check GPU type: Run `!nvidia-smi` (should show T4 or A100)
- Reduce `SIMULATIONS` for faster self-play (400 → 200)

### "Import errors"
- Make sure you ran the "Clone Repository" cell
- Verify the repository cloned successfully
- Try restarting runtime and re-running all cells

## Performance Expectations

### A100 (40GB)
- **Self-play**: ~100-150 games/hour (400 sims)
- **Training**: ~2-3 steps/second (batch size 8192)
- **Memory**: ~25-30GB used (192 filters, 15 blocks)

### T4 (16GB)
- **Self-play**: ~40-60 games/hour (200 sims)
- **Training**: ~1-2 steps/second (batch size 2048)
- **Memory**: ~8-12GB used (64 filters, 5 blocks)

## Tips for Best Results

1. **Use A100 for serious training** - 2.5x faster than T4
2. **Enable iterative training** - Refreshes replay buffer for better learning
3. **Save frequently** - Set `CHECKPOINT_INTERVAL = 50` for Colab's session limits
4. **Monitor Drive space** - Checkpoints are ~100-500MB each
5. **Test configurations** - Start with short training to verify setup

## Advanced: Resuming Training

The notebook automatically resumes from the latest checkpoint in Google Drive. To manually specify a checkpoint:

```python
# In the "Training Setup" cell, modify:
RESUME_CHECKPOINT = "/content/drive/MyDrive/alphazero_checkpoints/checkpoint_2000_f192_b15.pt"
```

## Advanced: Custom Configurations

You can customize any hyperparameter in the "Configuration" cell:

```python
# Network architecture
NUM_FILTERS = 128        # Width (64, 128, 192, 256)
NUM_BLOCKS = 10          # Depth (5, 10, 15, 19)

# MCTS settings
SIMULATIONS = 400        # Quality vs speed tradeoff
C_PUCT = 1.25           # Exploration constant
TEMPERATURE = 1.0        # Action selection randomness

# Training settings
LEARNING_RATE = 0.2      # Initial learning rate
BATCH_SIZE = 4096        # Training batch size
WEIGHT_DECAY = 1e-4      # L2 regularization

# Replay buffer
MIN_BUFFER_SIZE = 4096   # Positions before training starts
BUFFER_CAPACITY = 500000 # Maximum positions to store
```

## Support

For issues or questions:
- Check the main repository README
- Open an issue on GitHub
- Review the inline code comments in the notebook

## License

MIT License - Same as the main AlphaZero Chess project
