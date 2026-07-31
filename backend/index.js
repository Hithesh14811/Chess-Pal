const express = require("express");
const cors = require("cors");
const bodyParser = require("body-parser");
const { spawn } = require("child_process");
const path = require("path");
const fs = require("fs");

// Resolve the Stockfish binary. Priority:
//   1. STOCKFISH_PATH environment variable
//   2. ../stockfish-windows-x86-64-avx2.exe (project root, current location)
//   3. ../stockfish/stockfish/stockfish-windows-x86-64-avx2.exe (vendored copy)
function resolveStockfishPath() {
  if (process.env.STOCKFISH_PATH) {
    return process.env.STOCKFISH_PATH;
  }
  const candidates = [
    path.join(__dirname, "..", "stockfish-windows-x86-64-avx2.exe"),
    path.join(__dirname, "..", "stockfish", "stockfish", "stockfish-windows-x86-64-avx2.exe"),
  ];
  const found = candidates.find((p) => fs.existsSync(p));
  if (!found) {
    console.error(
      "Stockfish binary not found. Set the STOCKFISH_PATH environment variable to point to a stockfish executable."
    );
    console.error("Looked in:", candidates.join(", "));
    process.exit(1);
  }
  return found;
}

const STOCKFISH_PATH = resolveStockfishPath();
const PORT = Number(process.env.PORT) || 5000;
const SEARCH_TIMEOUT_MS = Number(process.env.SEARCH_TIMEOUT_MS) || 120000;

const app = express();
app.use(cors());
app.use(bodyParser.json());

app.get("/health", (req, res) => {
  res.json({ status: "ok", stockfish: STOCKFISH_PATH });
});

app.post("/best-move", (req, res) => {
  const fen = req.body.fen;
  if (!fen) {
    return res.status(400).json({ error: "Missing 'fen' field" });
  }

  const checkmateEnabled = req.body.checkmateEnabled || false;
  const analyzerMode = req.body.analyzerMode || false;
  const moveCount = req.body.moveCount || 0;

  let depth;
  if (checkmateEnabled) {
    depth = 15;
  } else if (analyzerMode) {
    // Depths for when analyzerMode is ON
    const depths = [12, 10, 8, 6, 5, 4, 3];
    depth = moveCount < depths.length ? depths[moveCount] : 2; // Minimum depth set to 2
  } else {
    // Depths for when analyzerMode is OFF
    const depths = [11, 9, 7, 5, 4, 3];
    depth = moveCount < depths.length ? depths[moveCount] : 2; // Minimum depth set to 2
  }

  let stockfish;
  let settled = false;

  const finish = (code, payload) => {
    if (settled) return;
    settled = true;
    clearTimeout(timer);
    if (stockfish && !stockfish.killed) stockfish.kill();
    res.status(code).json(payload);
  };

  const timer = setTimeout(() => {
    finish(504, { error: "Stockfish search timed out" });
  }, SEARCH_TIMEOUT_MS);

  try {
    stockfish = spawn(STOCKFISH_PATH, { windowsHide: true });
  } catch (err) {
    console.error("Failed to spawn Stockfish:", err);
    return finish(500, { error: "Failed to spawn Stockfish" });
  }

  let buffer = "";
  stockfish.stdout.on("data", (data) => {
    buffer += data.toString();

    // Stockfish may split its output across multiple chunks, so only
    // parse lines that are complete (end with a newline).
    const lines = buffer.split("\n");
    buffer = lines.pop();

    for (const line of lines) {
      if (line.includes("bestmove")) {
        const bestMove = line.trim().split(/\s+/)[1] || "";
        if (!bestMove || bestMove === "(none)") {
          finish(200, { bestMove: null });
        } else {
          finish(200, { bestMove });
        }
        return;
      }
    }
  });

  stockfish.stderr.on("data", (data) => {
    console.error("Stockfish stderr:", data.toString());
  });

  stockfish.on("error", (err) => {
    console.error("Stockfish process error:", err);
    finish(500, { error: "Stockfish process error" });
  });

  stockfish.on("exit", (code, signal) => {
    if (!settled) {
      finish(500, { error: `Stockfish exited unexpectedly (code=${code})` });
    }
  });

  stockfish.stdin.write("uci\n");
  stockfish.stdin.write("setoption name Skill Level value 20\n");
  stockfish.stdin.write(`position fen ${fen}\n`);
  stockfish.stdin.write(`go depth ${depth}\n`);
});

app.listen(PORT, () => {
  console.log(`Stockfish backend running on http://localhost:${PORT}`);
  console.log(`Using Stockfish at: ${STOCKFISH_PATH}`);
});
