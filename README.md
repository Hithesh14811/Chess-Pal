# Chess Pal

Chess Pal started with a simple annoyance: I'd be looking at a position on a chess site and want to know what Stockfish thinks, without setting the whole board up again by hand. So I taught it to read the board straight off my screen.

It ended up as four pieces:

- a **React app** where you set up any position and play against Stockfish,
- a small **Node API** that talks to the Stockfish engine,
- a **Python board reader** that screenshots a board, finds it, and recognizes every piece with a CNN,
- a tiny **Tkinter tool** for when I just want the best move for a position.

## How the board reader works

This is the part I spent the most time on.

1. **Grab the screen.** `mss` captures the region of the screen where the board is.
2. **Find the board.** Adaptive thresholding, then a contour search for the biggest roughly square shape with four corners.
3. **Straighten it.** A four-point perspective transform gives a clean top-down board even if the capture is slightly skewed.
4. **Cut it into 64 squares.**
5. **Classify every square.** All 64 squares go through the model in one batch. There are 13 classes: six white pieces, six black pieces and empty.
6. **Turn it into a position.** Square indices map to `a1`–`h8`, flipped depending on which side you're playing, and the result goes back to the web app as something like `{ "e4": "wP", "g8": "bN", ... }`.

It saves the intermediate images to `chess-python-backend/identify/` (raw capture, detected outline, warped board and each square), which helps a lot when detection goes wrong.

### The model

The model is EfficientNetB0 pretrained on ImageNet, with a small head on top (global average pooling, then 256 and 128 unit dense layers with batch norm, dropout and L2, then a 13-way softmax). It takes 128×128 squares. I trained it on about 67,000 labeled square images, roughly 4,000 to 11,500 per class.

Training happens in two phases, in `train_efficientnet.py`:

1. Freeze the backbone and train only the head (Adam, lr 1e-3).
2. Unfreeze the backbone from block 4b up and fine-tune at lr 1e-4.

It uses class weights because the classes aren't balanced (there are far more empty squares and pawns than kings), early stopping on validation accuracy, and LR reduction on plateau, and it tracks top-2 accuracy alongside accuracy. Augmentation is limited to small shifts, zoom, brightness and contrast. No flips or rotations, because pieces on a screen are always upright and a flipped piece isn't something the model will ever see.

One thing that bit me: the training pipeline has to feed images exactly the way the inference code does (OpenCV's BGR channel order, scaled to 0–1). If they don't match, accuracy quietly drops, so both now do it the same way.

`train.py` is an earlier experiment: a smaller network built from scratch (64×64 input, depthwise-separable convs with squeeze-and-excitation). It's faster to train but the EfficientNet model was more accurate, so that's the one the app uses.

```bash
cd chess-python-backend
python train_efficientnet.py --data path/to/dataset
```

The requirements pin TensorFlow 2.19 / Keras 3.9 on purpose. The saved model was made with Keras 3.9, and newer Keras versions refuse to load it.

The dataset should have one folder per class: `bB bK bN bP bQ bR empty wB wK wN wP wQ wR`.

## The web app

- Load the normal starting position, or build any position piece by piece.
- Pick which color Stockfish plays and who moves first.
- All the rules come from `chess.js`, so castling, en passant, promotion, check, checkmate, stalemate and repetition all work properly.
- Legal moves light up while you drag a piece, and the last move and checks are highlighted.
- In a custom position you can play in "free mode" (move whenever, ask Stockfish to move when you want) or strict mode (turns alternate).
- There's a "deeper search" toggle if you want Stockfish to think harder.

## The Stockfish API

`POST /best-move` with `{ "fen": "...", "checkmateEnabled": true|false }` returns `{ "bestMove": "e2e4" }`.

It starts Stockfish, sends the position over UCI and returns the engine's move. The default search depth is 12, or 18 with deeper search on. The FEN is checked before it goes anywhere near the engine, so bad input gets a `400` instead of confusing Stockfish. Slow searches time out with a `504`.

## Running it

You'll need Node 18+, Python 3.10–3.12 (TensorFlow 2.19 doesn't support newer versions yet) and **Stockfish**. Stockfish isn't in the repo. Download it from [stockfishchess.org](https://stockfishchess.org/download/) and either drop the binary into the `stockfish/` folder, put it on your `PATH` as `stockfish`, or set `STOCKFISH_PATH`.

**1. Stockfish API**

```bash
cd backend
npm install
npm start           # http://localhost:5000
```

Optional settings: `STOCKFISH_PATH`, `PORT` (5000), `SEARCH_DEPTH` (12), `DEEP_SEARCH_DEPTH` (18), `SEARCH_TIMEOUT_MS` (120000).

**2. Board reader**

```bash
cd chess-python-backend
pip install -r requirements.txt
python cropper_backend.py    # http://127.0.0.1:5001
```

The capture area is set for my monitor, so you'll want to point it at wherever the board is on yours:

```bash
set CAPTURE_X=1526
set CAPTURE_Y=4
set CAPTURE_X2=3070
set CAPTURE_Y2=1632
python cropper_backend.py
```

**3. Web app**

```bash
cd chess-ai
npm install
npm start           # http://localhost:3000
```

If your APIs run somewhere else, set `REACT_APP_BEST_MOVE_API_URL` and `REACT_APP_ANALYZE_API_URL` in `chess-ai/.env`.

**4. Desktop tool (optional)**

```bash
python stockfish/chess_pal_tool.py
```

Pick a piece and a color, click squares to place pieces, choose whose move it is and hit **Get Best Move**. You need both kings on the board.

## Project layout

```
chess-ai/                 React app (board, game logic, API calls in src/App.js)
backend/                  Express API that drives Stockfish
chess-python-backend/
  cropper_backend.py      board reader API
  train_efficientnet.py   trains the model the app uses
  train.py                earlier from-scratch CNN experiment
  chess_model_efficientnet_best2.keras
stockfish/
  chess_pal_tool.py       Tkinter best-move tool
```

## Things I'd still like to improve

- The capture region is a fixed rectangle. Detecting the board anywhere on screen would be nicer.
- For custom positions there's no way to know castling or en passant rights from the pieces alone, so those start switched off.
- Inference runs on the CPU. That's fine for 64 small images, but there's no GPU path.

## Troubleshooting

| Problem | Fix |
| --- | --- |
| `Stockfish binary not found` | Put Stockfish in `stockfish/`, on your `PATH`, or set `STOCKFISH_PATH` |
| "The AI could not move" | The Node API isn't running (`npm start` in `backend/`) |
| "Board analysis failed" | The board reader isn't running (`python cropper_backend.py`) |
| Board not detected | Make sure the whole board is inside the `CAPTURE_*` area |
| Can't start a custom game | You need exactly one white king and one black king |
