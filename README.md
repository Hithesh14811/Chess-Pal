# Chess Pal

A chess assistant that combines a **React web app**, a **Stockfish-powered
Node.js API**, a **computer-vision board reader** (Python, OpenCV, TensorFlow),
and a small **Tkinter desktop tool**.

It can:

1. **Play against Stockfish** from the standard start or any custom position.
2. **Read a chessboard off your screen**: capture it, find the board, classify
   all 64 squares with a CNN, and load the position into the web app.
3. **Suggest the best move** for any position you set up in the desktop tool.

---

## Architecture

```
+------------------------+   POST /best-move   +------------------------------+
|  chess-ai (React)      | ------------------> |  backend (Node + Express)    |
|  - position editor     |                     |  - talks UCI to Stockfish    |
|  - play vs Stockfish   |                     |  - port 5000                 |
|  - legal-move hints    |                     +------------------------------+
+------------------------+
            |
            | POST /analyze
            v
+-------------------------------------------------------------+
|  chess-python-backend (Flask)                    port 5001  |
|  capture screen (mss) -> find board (OpenCV contours)       |
|  -> perspective warp -> split into 64 squares               |
|  -> classify each square (EfficientNetB0 CNN, 13 classes)   |
|  -> return { "e4": "wP", ... }                              |
+-------------------------------------------------------------+

stockfish/chess_pal_tool.py   standalone Tkinter tool (place pieces, get best move)
```

---

## Features

**Web app (`chess-ai`)**: built with React, `react-chessboard` and `chess.js`.

- Load the standard board, or build any position piece by piece (with a remove mode).
- Choose the AI's color and who moves first.
- Fully legal play via `chess.js`: castling, en passant, promotion, check,
  checkmate, stalemate, threefold repetition and other draws.
- Legal-move hints while dragging, last-move and check highlighting, move list.
- *Free mode* for custom positions: move whenever you like and ask Stockfish to
  play on demand. *Strict mode* alternates turns automatically.
- Optional deeper Stockfish search.

**Stockfish API (`backend`)**

- `POST /best-move` with `{ "fen": "...", "checkmateEnabled": bool }` returns
  `{ "bestMove": "e2e4" }` (or `null` when there is no legal move).
- Searches at depth 12 by default and 18 when `checkmateEnabled` is set.
- Validates the FEN before passing it to the engine; returns `400` for bad
  input, `500` for engine errors and `504` if the search times out.
- `GET /health` reports status and the Stockfish binary in use.

**Board reader (`chess-python-backend`)**

1. **Capture**: grab a configurable screen region with `mss`.
2. **Detect**: adaptive threshold + contour search for the largest roughly
   square 4-corner shape.
3. **Warp**: four-point perspective transform to a top-down board.
4. **Split**: crop 64 squares.
5. **Classify**: one batched prediction over all squares. The input size is
   read from the model.
6. **Map**: convert square indices to `a1`–`h8`, flipped by board orientation.

Debug images are written to `chess-python-backend/identify/` (`1a.png` raw
capture, `1b.png` detected outline, `1c.png` warped board, `1.png`–`64.png`
squares).

Classes (same order in training and inference):

```
bB  bK  bN  bP  bQ  bR  empty  wB  wK  wN  wP  wQ  wR
```

---

## Model

`chess-python-backend/chess_model_efficientnet_best2.keras` is the model the
board reader uses: an **EfficientNetB0** backbone (transfer learning) with a
global-average-pooling + dense classification head, 128×128 input, 13
classes, trained on a dataset of ~67,000 labeled square images (≈4,000–11,500
per class).

`train.py` trains a lighter alternative from scratch (64×64 input,
depthwise-separable convolutions with squeeze-and-excitation blocks, AdamW,
early stopping and LR scheduling). Point it at a dataset with
`CHESS_DATASET_PATH` (one folder per class name); it saves
`ultra_chess_model_<accuracy>.h5`. To use a different model, place it in
`chess-python-backend/` and add it to `MODEL_CANDIDATES` in `cropper_backend.py`.

---

## Getting started

### Prerequisites

- Node.js 18+
- Python 3.10+
- **Stockfish**: download it from <https://stockfishchess.org/download/> and
  either put the binary in the `stockfish/` folder, add it to your `PATH` as
  `stockfish`, or set `STOCKFISH_PATH` to its full path.

### 1. Stockfish API

```bash
cd backend
npm install
npm start          # http://localhost:5000
```

| Variable            | Default                                        |
|---------------------|------------------------------------------------|
| `STOCKFISH_PATH`    | `stockfish/` folder, then `stockfish` on PATH  |
| `PORT`              | `5000`                                         |
| `SEARCH_DEPTH`      | `12`                                           |
| `DEEP_SEARCH_DEPTH` | `18`                                           |
| `SEARCH_TIMEOUT_MS` | `120000`                                       |

### 2. Board reader

```bash
cd chess-python-backend
pip install -r requirements.txt
python cropper_backend.py   # http://127.0.0.1:5001
```

The capture region defaults to one specific monitor layout. Set it to wherever
the board appears on your screen:

```bash
set CAPTURE_X=1526
set CAPTURE_Y=4
set CAPTURE_X2=3070
set CAPTURE_Y2=1632
python cropper_backend.py
```

### 3. Web app

```bash
cd chess-ai
npm install
npm start          # http://localhost:3000
```

API URLs can be overridden in `chess-ai/.env`:

| Variable                      | Default                  |
|-------------------------------|--------------------------|
| `REACT_APP_BEST_MOVE_API_URL` | `http://localhost:5000`  |
| `REACT_APP_ANALYZE_API_URL`   | `http://127.0.0.1:5001`  |

### 4. Desktop tool (optional)

```bash
python stockfish/chess_pal_tool.py
```

Pick a piece type and color, click squares to place pieces (or use the eraser),
choose who moves, then **Get Best Move**. Both kings must be on the board.

---

## Project structure

```
chess-ai/                  React web app (Create React App)
  src/App.js               board, game flow, Stockfish and analyzer calls
backend/
  index.js                 Express API that drives Stockfish over UCI
chess-python-backend/
  cropper_backend.py       Flask API: capture -> detect -> classify
  train.py                 CNN training script
  requirements.txt
  chess_model_efficientnet_best2.keras
stockfish/
  chess_pal_tool.py        Tkinter best-move tool
```

---

## Known limitations

- The board reader captures a fixed screen rectangle, so it must be set for
  your monitor with the `CAPTURE_*` variables.
- Custom positions start without castling or en-passant rights, since those
  can't be inferred from piece placement alone.
- The board reader runs inference on CPU; no GPU path is configured.

---

## Troubleshooting

| Symptom                              | Fix                                                        |
|--------------------------------------|------------------------------------------------------------|
| `Stockfish binary not found`         | Put Stockfish in `stockfish/`, on `PATH`, or set `STOCKFISH_PATH` |
| "The AI could not move"              | Start the Node API (`npm start` in `backend/`)             |
| "Board analysis failed"              | Start the board reader (`python cropper_backend.py`)       |
| Board not detected                   | Make sure the whole board is inside the `CAPTURE_*` region |
| Can't start a custom game            | Place exactly one white king and one black king            |
