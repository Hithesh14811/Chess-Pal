const express = require("express");
const cors = require("cors");
const { spawn } = require("child_process");
const path = require("path");
const fs = require("fs");

// Resolve the Stockfish binary. Priority:
//   1. STOCKFISH_PATH environment variable
//   2. A binary placed in ../stockfish/ (see README)
//   3. `stockfish` on the system PATH
function findOnPath(name) {
  const exts = process.platform === "win32" ? [".exe", ""] : [""];
  for (const dir of (process.env.PATH || "").split(path.delimiter)) {
    if (!dir) continue;
    for (const ext of exts) {
      const candidate = path.join(dir, name + ext);
      if (fs.existsSync(candidate)) return candidate;
    }
  }
  return null;
}

function resolveStockfishPath() {
  if (process.env.STOCKFISH_PATH) {
    return process.env.STOCKFISH_PATH;
  }
  const dir = path.join(__dirname, "..", "stockfish");
  const candidates = [
    path.join(dir, "stockfish-windows-x86-64-avx2.exe"),
    path.join(dir, "stockfish.exe"),
    path.join(dir, "stockfish"),
  ];
  const found = candidates.find((p) => fs.existsSync(p)) || findOnPath("stockfish");
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
const DEFAULT_DEPTH = Number(process.env.SEARCH_DEPTH) || 12;
const DEEP_SEARCH_DEPTH = Number(process.env.DEEP_SEARCH_DEPTH) || 18;
const FEN_PATTERN = /^[pnbrqkPNBRQK1-8\/]+ [wb] (-|[KQkq]{1,4}) (-|[a-h][36]) \d+ \d+$/;

const app = express();
app.use(cors());
app.use(express.json());

app.get("/health", (req, res) => {
  res.json({ status: "ok", stockfish: STOCKFISH_PATH });
});

app.post("/best-move", (req, res) => {
  const fen = typeof req.body?.fen === "string" ? req.body.fen.trim() : "";
  if (!fen) {
    return res.status(400).json({ error: "Missing 'fen' field" });
  }
  // The FEN is written straight into the engine's stdin, so only accept
  // well-formed FEN (no newlines that could inject extra UCI commands).
  if (!FEN_PATTERN.test(fen)) {
    return res.status(400).json({ error: "Invalid FEN" });
  }

  const depth = req.body.checkmateEnabled ? DEEP_SEARCH_DEPTH : DEFAULT_DEPTH;

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
