import React, { useEffect, useRef, useState } from "react";
import { Chess } from "chess.js";
import { Chessboard } from "react-chessboard";
import "./App.css";

const BEST_MOVE_API_URL =
  process.env.REACT_APP_BEST_MOVE_API_URL || "http://localhost:5000";
const ANALYZE_API_URL =
  process.env.REACT_APP_ANALYZE_API_URL || "http://127.0.0.1:5001";

const emptyPosition = {};
const piecePalette = [
  ["wK", "♔"], ["wQ", "♕"], ["wR", "♖"], ["wB", "♗"], ["wN", "♘"], ["wP", "♙"],
  ["bK", "♚"], ["bQ", "♛"], ["bR", "♜"], ["bB", "♝"], ["bN", "♞"], ["bP", "♟"],
];

function positionToFen(position, activeColor) {
  const rows = [];
  for (let rank = 8; rank >= 1; rank -= 1) {
    let row = "";
    let empty = 0;
    for (const file of "abcdefgh") {
      const piece = position[`${file}${rank}`];
      if (!piece) {
        empty += 1;
      } else {
        if (empty) row += empty;
        empty = 0;
        const type = piece[1].toLowerCase();
        row += piece[0] === "w" ? type.toUpperCase() : type;
      }
    }
    if (empty) row += empty;
    rows.push(row);
  }
  return `${rows.join("/")} ${activeColor} - - 0 1`;
}

function defaultPosition() {
  const game = new Chess();
  const position = {};
  for (const row of game.board()) {
    for (const square of row) {
      if (square) position[square.square] = `${square.color}${square.type.toUpperCase()}`;
    }
  }
  return position;
}

function App() {
  const gameRef = useRef(new Chess());
  const aiThinkingRef = useRef(false);
  const [setupPosition, setSetupPosition] = useState(emptyPosition);
  const [boardPosition, setBoardPosition] = useState(emptyPosition);
  const [selectedPiece, setSelectedPiece] = useState(null);
  const [removeMode, setRemoveMode] = useState(false);
  const [isGameStarted, setIsGameStarted] = useState(false);
  const [gameMode, setGameMode] = useState("custom");
  const [aiColor, setAiColor] = useState("black");
  const [strictMode, setStrictMode] = useState(false);
  const [firstMover, setFirstMover] = useState("player");
  const [lastMove, setLastMove] = useState(null);
  const [possibleMoves, setPossibleMoves] = useState([]);
  const [selectedSquare, setSelectedSquare] = useState(null);
  const [checkSquare, setCheckSquare] = useState(null);
  const [moveHistory, setMoveHistory] = useState([]);
  const [status, setStatus] = useState("Set up a position, then start the game.");
  const [error, setError] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [checkmateEnabled, setCheckmateEnabled] = useState(false);

  const playerColor = aiColor === "white" ? "black" : "white";
  const boardOrientation = isGameStarted ? playerColor : aiColor;
  const enforceTurns = gameMode === "default" || strictMode;

  const forceTurn = (game, color) => {
    const fields = game.fen().split(" ");
    fields[1] = color[0];
    return new Chess(fields.join(" "));
  };

  const findKingSquare = (game, color) => {
    for (const file of "abcdefgh") {
      for (let rank = 1; rank <= 8; rank += 1) {
        const square = `${file}${rank}`;
        const piece = game.get(square);
        if (piece?.type === "k" && piece.color === color[0]) return square;
      }
    }
    return null;
  };

  const updateStatus = (game) => {
    setCheckSquare(game.isCheck() ? findKingSquare(game, game.turn()) : null);
    if (game.isCheckmate()) {
      const winner = game.turn() === "w" ? "Black" : "White";
      setStatus(`Checkmate — ${winner} wins.`);
    } else if (game.isStalemate()) {
      setStatus("Draw by stalemate.");
    } else if (game.isThreefoldRepetition()) {
      setStatus("Draw by threefold repetition.");
    } else if (game.isDraw()) {
      setStatus("Draw.");
    } else if (game.isCheck()) {
      const color = game.turn() === "w" ? "White" : "Black";
      setStatus(`${color} is in check!`);
    } else {
      const color = game.turn() === "w" ? "White" : "Black";
      setStatus(`${color} to move.`);
    }
  };

  const recordMove = (move) => {
    setMoveHistory((history) => [...history, move.san]);
    setLastMove({ from: move.from, to: move.to });
    setPossibleMoves([]);
    setSelectedSquare(null);
    setBoardPosition(gameRef.current.fen());
    updateStatus(gameRef.current);
  };

  const requestAiMove = async () => {
    const game = gameRef.current;
    if (aiThinkingRef.current || game.isGameOver() || game.turn() !== aiColor[0]) return;
    aiThinkingRef.current = true;
    setIsLoading(true);
    setError("");
    try {
      const response = await fetch(`${BEST_MOVE_API_URL}/best-move`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          fen: game.fen(),
          checkmateEnabled,
        }),
      });
      if (!response.ok) throw new Error(`AI service returned ${response.status}`);
      const { bestMove } = await response.json();
      if (!bestMove) {
        updateStatus(game);
        return;
      }
      const move = game.move({
        from: bestMove.slice(0, 2),
        to: bestMove.slice(2, 4),
        promotion: bestMove.slice(4, 5) || "q",
      });
      if (!move) throw new Error("AI returned an invalid move");
      if (!enforceTurns) gameRef.current = forceTurn(game, playerColor);
      recordMove(move);
    } catch (err) {
      setError(`The AI could not move: ${err.message}`);
      setStatus("AI move failed. Check that the Stockfish service is running.");
    } finally {
      aiThinkingRef.current = false;
      setIsLoading(false);
    }
  };

  useEffect(() => {
    if (isGameStarted && gameRef.current.turn() === aiColor[0] && !gameRef.current.isGameOver()) {
      requestAiMove();
    }
  // requestAiMove intentionally runs only after a game/turn state change.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isGameStarted, boardPosition, aiColor, enforceTurns]);

  const playDefaultBoard = () => {
    const position = defaultPosition();
    setSetupPosition(position);
    setBoardPosition(position);
    setGameMode("default");
    setSelectedPiece(null);
    setRemoveMode(false);
    setError("");
    setStatus("Default board loaded. Choose the AI color, then press Start Game.");
  };

  const analyzeBoard = async () => {
    setIsLoading(true);
    setError("");
    try {
      const response = await fetch(`${ANALYZE_API_URL}/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ aiColor }),
      });
      if (!response.ok) throw new Error(`Analyzer returned ${response.status}`);
      const position = await response.json();
      setSetupPosition(position);
      setBoardPosition(position);
      setGameMode("custom");
      setSelectedPiece(null);
      setRemoveMode(false);
      setStatus("Board analyzed. Pick the first mover and press Start Game.");
    } catch (err) {
      setError(`Board analysis failed: ${err.message}. Ensure the Python analyzer is running on port 5001.`);
    } finally {
      setIsLoading(false);
    }
  };

  const startGame = () => {
    try {
      const activeColor = gameMode === "default" ? "white" : (firstMover === "ai" ? aiColor : playerColor);
      // The standard start position keeps its castling rights; for custom
      // setups they can't be known, so positionToFen marks them unavailable.
      const game = gameMode === "default"
        ? new Chess()
        : new Chess(positionToFen(setupPosition, activeColor[0]));
      gameRef.current = game;
      setBoardPosition(game.fen());
      setMoveHistory([]);
      setLastMove(null);
      setCheckSquare(null);
      setError("");
      setIsGameStarted(true);
      updateStatus(game);
    } catch (err) {
      setError("Add a legal position with one white king and one black king before starting.");
    }
  };

  const resetGame = () => {
    gameRef.current = new Chess();
    aiThinkingRef.current = false;
    setSetupPosition({});
    setBoardPosition({});
    setIsGameStarted(false);
    setGameMode("custom");
    setSelectedPiece(null);
    setRemoveMode(false);
    setLastMove(null);
    setCheckSquare(null);
    setPossibleMoves([]);
    setSelectedSquare(null);
    setMoveHistory([]);
    setError("");
    setStatus("Set up a position, then start the game.");
  };

  const handleSetupSquareClick = (square) => {
    setGameMode("custom");
    if (removeMode) {
      setSetupPosition((position) => {
        const next = { ...position };
        delete next[square];
        setBoardPosition(next);
        return next;
      });
    } else if (selectedPiece) {
      setSetupPosition((position) => {
        const next = { ...position, [square]: selectedPiece };
        setBoardPosition(next);
        return next;
      });
    }
  };

  const handlePlayerMove = (from, to) => {
    const game = gameRef.current;
    if (isLoading || game.isGameOver() || (enforceTurns && game.turn() !== playerColor[0])) return false;
    try {
      if (!enforceTurns && game.turn() !== playerColor[0]) gameRef.current = forceTurn(game, playerColor);
      const currentGame = gameRef.current;
      const move = currentGame.move({ from, to, promotion: "q" });
      if (!move) return false;
      if (!enforceTurns) gameRef.current = forceTurn(currentGame, playerColor);
      recordMove(move);
      return true;
    } catch {
      return false;
    }
  };

  const handlePieceDragBegin = (piece, sourceSquare) => {
    const game = gameRef.current;
    if (!isGameStarted || isLoading || game.isGameOver()) return;
    if (enforceTurns && game.turn() !== playerColor[0]) return;
    if (!enforceTurns && game.turn() !== playerColor[0]) gameRef.current = forceTurn(game, playerColor);
    const currentGame = gameRef.current;
    const selectedPiece = currentGame.get(sourceSquare);
    if (!selectedPiece || selectedPiece.color !== playerColor[0]) return;
    setSelectedSquare(sourceSquare);
    setPossibleMoves(currentGame.moves({ square: sourceSquare, verbose: true }).map((move) => move.to));
  };

  const handlePieceDragEnd = () => {
    setSelectedSquare(null);
    setPossibleMoves([]);
  };

  const exitGame = () => {
    setIsGameStarted(false);
    setMoveHistory([]);
    setLastMove(null);
    setCheckSquare(null);
    setPossibleMoves([]);
    setSelectedSquare(null);
    setStatus("Game ended. You can set up another position.");
  };

  const playAgain = () => {
    const position = defaultPosition();
    const game = new Chess();
    gameRef.current = game;
    setGameMode("default");
    setSetupPosition(position);
    setBoardPosition(game.fen());
    setMoveHistory([]);
    setLastMove(null);
    setCheckSquare(null);
    setError("");
    setIsGameStarted(true);
    updateStatus(game);
  };

  return (
    <main className="app-shell">
      <section className="hero">
        <p className="eyebrow">CHESS PAL</p>
        <h1>Chess Pal</h1>
        <p>Build a position or load a classic board. Stockfish takes every AI turn automatically.</p>
      </section>

      <section className="game-layout">
        <div className="board-card">
          <Chessboard
            position={boardPosition}
            boardWidth={Math.min(560, Math.max(300, window.innerWidth - 80))}
            boardOrientation={boardOrientation}
            onSquareClick={!isGameStarted ? handleSetupSquareClick : undefined}
            onPieceDrop={isGameStarted ? handlePlayerMove : undefined}
            onPieceDragBegin={isGameStarted ? handlePieceDragBegin : undefined}
            onPieceDragEnd={isGameStarted ? handlePieceDragEnd : undefined}
            customDarkSquareStyle={{ backgroundColor: "#779556" }}
            customLightSquareStyle={{ backgroundColor: "#ebecd0" }}
            customSquareStyles={{
              ...(lastMove && { [lastMove.from]: { backgroundColor: "rgba(255, 214, 10, .55)" }, [lastMove.to]: { backgroundColor: "rgba(255, 214, 10, .55)" } }),
              ...(checkSquare && { [checkSquare]: { backgroundColor: "rgba(220, 45, 45, .75)", boxShadow: "inset 0 0 0 4px #9f1212" } }),
              ...(selectedSquare && { [selectedSquare]: { backgroundColor: "rgba(82, 142, 204, .5)" } }),
              ...possibleMoves.reduce((styles, square) => ({ ...styles, [square]: { background: "radial-gradient(circle, rgba(135, 140, 145, .72) 0 18%, transparent 20%)" } }), {}),
            }}
          />
          {isGameStarted && gameRef.current.isCheckmate() && <div className="checkmate-overlay">
            <p>GAME OVER</p>
            <h2>Checkmate</h2>
            <span>{status}</span>
            <div><button className="primary" onClick={playAgain}>Play Again</button><button className="secondary" onClick={exitGame}>Exit Game</button></div>
          </div>}
        </div>

        <aside className="control-panel">
          <div className={`status ${error ? "status-error" : ""}`}>{error || status}</div>

          {!isGameStarted ? <>
            <button className="primary" onClick={playDefaultBoard}>Play Default Board</button>
            <button className="secondary" onClick={analyzeBoard} disabled={isLoading}>{isLoading ? "Analyzing board…" : "Analyze Board Screenshot"}</button>
            <p className="help-text">Captures the configured area on the right side of your screen, detects the board, and uses the ML model to place the pieces.</p>
            <h2>Custom setup</h2>
            <div className="piece-picker">
              {piecePalette.map(([type, label]) => <button key={type} className={selectedPiece === type && !removeMode ? "selected" : ""} onClick={() => { setSelectedPiece(type); setRemoveMode(false); }}>{label}</button>)}
            </div>
            <button className={removeMode ? "danger active" : "secondary"} onClick={() => { setRemoveMode((value) => !value); setSelectedPiece(null); }}>Remove piece</button>
            <label className="option">AI plays as
              <select value={aiColor} onChange={(event) => setAiColor(event.target.value)}>
                <option value="black">Black — you move first</option>
                <option value="white">White — AI moves first</option>
              </select>
            </label>
            {gameMode === "custom" && <>
              <label className="option">First move
                <select value={firstMover} onChange={(event) => setFirstMover(event.target.value)}>
                  <option value="player">You play first</option>
                  <option value="ai">AI plays first</option>
                </select>
              </label>
              <label className="toggle"><input type="checkbox" checked={strictMode} onChange={(event) => setStrictMode(event.target.checked)} /> Strict mode — alternate turns automatically</label>
            </>}
            <label className="toggle"><input type="checkbox" checked={checkmateEnabled} onChange={(event) => setCheckmateEnabled(event.target.checked)} /> Deeper AI search</label>
            <button className="primary" onClick={startGame}>Start Game</button>
          </> : <>
            <div className="turn-info"><strong>You are {playerColor}.</strong><br />{isLoading ? "Stockfish is thinking…" : enforceTurns ? "Drag a piece to make a legal move." : "You can move whenever you like, or ask Stockfish to play."}</div>
            {gameMode === "custom" && !strictMode && <button className="primary" onClick={() => { gameRef.current = forceTurn(gameRef.current, aiColor); requestAiMove(); }} disabled={isLoading || gameRef.current.isGameOver()}>AI plays {aiColor}</button>}
            <h2>Moves</h2>
            <ol className="moves">{moveHistory.map((move, index) => <li key={`${move}-${index}`}>{move}</li>)}</ol>
          </>}
          <button className="secondary reset" onClick={resetGame}>Reset board</button>
        </aside>
      </section>
    </main>
  );
}

export default App;
