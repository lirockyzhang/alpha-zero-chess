# AlphaZero Chess Web Interface

Interactive web interface for playing chess against trained AlphaZero models.

## Features

- 🎮 **Interactive Chessboard**: Drag-and-drop interface powered by chessboard.js
- 🤖 **AI Opponent**: Play against your trained AlphaZero model
- 🎨 **Modern UI**: Clean, responsive design with real-time move validation
- 📊 **Game Statistics**: Track move history and game outcomes
- ⚙️ **Configurable**: Adjust MCTS simulations and other parameters

## Quick Start

### 1. Install Dependencies

```bash
pip install flask flask-cors
```

### 2. Launch the Web Interface

```bash
# From the project root directory
python web/run.py --checkpoint checkpoints/your_model.pt

# Or with custom settings
python web/run.py \
  --checkpoint checkpoints/your_model.pt \
  --simulations 400 \
  --device cuda \
  --port 5000
```

### 3. Open in Browser

Navigate to `http://localhost:5000` in your web browser.

## Command-Line Options

```
--checkpoint PATH    Path to model checkpoint (required)
--simulations N      MCTS simulations per move (default: 400)
--device DEVICE      Device to run on: cuda or cpu (default: cuda)
--port PORT          Port to run server on (default: 5000)
--debug              Enable Flask debug mode
```

## Usage

1. **Select Your Color**: Choose to play as White or Black
2. **Start Game**: Click "New Game" to begin
3. **Make Moves**: Drag and drop pieces to make your moves
4. **AI Response**: The AI will automatically respond after your move
5. **Game Over**: The interface will notify you when the game ends

## Architecture

```
web/
├── app.py           # Flask application and game logic
├── run.py           # Entry point script
├── templates/       # HTML templates
│   └── chess.html   # Main game interface
├── static/          # Static assets (CSS, JS, images)
└── README.md        # This file
```

## Development

### Running in Debug Mode

```bash
python web/run.py --checkpoint checkpoints/model.pt --debug
```

Debug mode enables:
- Auto-reload on code changes
- Detailed error messages
- Flask debugger

### Adding Custom Styling

Place custom CSS files in `web/static/css/` and reference them in `templates/chess.html`.

### Modifying the UI

Edit `templates/chess.html` to customize the interface. The template uses:
- **chessboard.js**: For the interactive board
- **chess.js**: For move validation and game logic
- **jQuery**: For AJAX requests

## API Endpoints

The web interface exposes the following REST API endpoints:

### `POST /api/new_game`
Start a new game.

**Request:**
```json
{
  "session_id": "unique_session_id",
  "color": "white"  // or "black"
}
```

**Response:**
```json
{
  "success": true,
  "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
  "model_move": "e2e4",  // Only if human plays black
  "game_over": false
}
```

### `POST /api/make_move`
Make a move and get AI response.

**Request:**
```json
{
  "session_id": "unique_session_id",
  "move": "e2e4"  // UCI format
}
```

**Response:**
```json
{
  "success": true,
  "fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
  "model_move": "e7e5",
  "game_over": false,
  "result": null
}
```

### `POST /api/get_legal_moves`
Get legal moves for current position.

**Request:**
```json
{
  "session_id": "unique_session_id"
}
```

**Response:**
```json
{
  "success": true,
  "legal_moves": ["e2e4", "e2e3", "d2d4", ...]
}
```

## Troubleshooting

### Port Already in Use

If port 5000 is already in use, specify a different port:

```bash
python web/run.py --checkpoint model.pt --port 8080
```

### CUDA Out of Memory

If you encounter CUDA memory errors, try:

1. Reduce MCTS simulations: `--simulations 200`
2. Use CPU instead: `--device cpu`
3. Use a smaller model

### Flask Not Found

Install Flask and Flask-CORS:

```bash
pip install flask flask-cors
```

## Performance Tips

- **MCTS Simulations**: Higher values (800+) provide stronger play but slower response
- **GPU Acceleration**: Use `--device cuda` for faster inference
- **Model Size**: Smaller models (64 filters, 5 blocks) respond faster than large models

## License

Part of the AlphaZero Chess project. See main project README for license information.
