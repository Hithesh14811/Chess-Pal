import os
import subprocess
import tkinter as tk
from tkinter import messagebox, ttk


def resolve_stockfish_path():
    """Locate the Stockfish binary relative to this script, or via STOCKFISH_PATH."""
    env_path = os.environ.get("STOCKFISH_PATH")
    if env_path:
        return env_path

    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, "stockfish", "stockfish-windows-x86-64-avx2.exe"),
        os.path.join(here, "..", "stockfish-windows-x86-64-avx2.exe"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(
        "Stockfish binary not found. Set the STOCKFISH_PATH environment "
        "variable to point to a stockfish executable."
    )


class ZiffieChessTool:
    # (label, FEN letter). Uppercase = White, lowercase = Black.
    PIECES = [
        ("♔ King", "K"),
        ("♕ Queen", "Q"),
        ("♖ Rook", "R"),
        ("♗ Bishop", "B"),
        ("♘ Knight", "N"),
        ("♙ Pawn", "P"),
    ]

    def __init__(self, root):
        self.root = root
        self.root.title("Ziffie Chess Strategy Tool")

        self.board = [["" for _ in range(8)] for _ in range(8)]
        self.buttons = []

        board_frame = tk.Frame(root)
        board_frame.grid(row=0, column=0, rowspan=8, columnspan=8, padx=10, pady=10)

        for row in range(8):
            row_buttons = []
            for col in range(8):
                color = "#f0d9b5" if (row + col) % 2 == 0 else "#b58863"
                btn = tk.Button(
                    board_frame,
                    text="",
                    width=4,
                    height=2,
                    bg=color,
                    command=lambda r=row, c=col: self.select_piece(r, c),
                )
                btn.grid(row=row, column=col)
                row_buttons.append(btn)
            self.buttons.append(row_buttons)

        # Side panel
        panel = tk.Frame(root)
        panel.grid(row=0, column=8, rowspan=8, padx=10)

        tk.Label(panel, text="Piece type").grid(row=0, column=0, sticky="w")
        self.piece_var = tk.StringVar(value="Q")
        piece_dropdown = ttk.Combobox(
            panel,
            textvariable=self.piece_var,
            state="readonly",
            values=[p[1] for p in self.PIECES],
        )
        piece_dropdown.grid(row=1, column=0, sticky="w")

        tk.Label(panel, text="Piece color").grid(row=2, column=0, sticky="w")
        self.color_var = tk.StringVar(value="White")
        color_dropdown = ttk.Combobox(
            panel,
            textvariable=self.color_var,
            state="readonly",
            values=["White", "Black"],
        )
        color_dropdown.grid(row=3, column=0, sticky="w")

        tk.Label(panel, text="Clear square").grid(row=4, column=0, sticky="w")
        self.clear_mode = tk.BooleanVar(value=False)
        tk.Checkbutton(
            panel, text="Eraser", variable=self.clear_mode, command=self.toggle_clear
        ).grid(row=5, column=0, sticky="w")

        tk.Label(panel, text="First mover").grid(row=6, column=0, sticky="w")
        self.mover_var = tk.StringVar(value="w")
        tk.OptionMenu(panel, self.mover_var, "w", "b").grid(row=7, column=0, sticky="w")

        # Status bar
        status_frame = tk.Frame(root)
        status_frame.grid(row=9, column=0, columnspan=9, pady=10)

        self.submit_button = tk.Button(
            status_frame, text="Get Best Move", command=self.calculate_best_move
        )
        self.submit_button.grid(row=0, column=0, padx=5)

        tk.Button(status_frame, text="Clear Board", command=self.clear_board).grid(
            row=0, column=1, padx=5
        )

        self.status_var = tk.StringVar(value="Ready")
        tk.Label(status_frame, textvariable=self.status_var).grid(row=1, column=0, columnspan=2)

    def toggle_clear(self):
        if self.clear_mode.get():
            self.status_var.set("Eraser mode: click a square to remove its piece")
        else:
            self.status_var.set("Placement mode")

    def clear_board(self):
        for row in range(8):
            for col in range(8):
                self.board[row][col] = ""
                self.buttons[row][col].config(text="")
        self.status_var.set("Board cleared")

    def select_piece(self, row, col):
        if self.clear_mode.get():
            self.board[row][col] = ""
            self.buttons[row][col].config(text="")
            return

        piece = self.piece_var.get()
        color = "w" if self.color_var.get() == "White" else "b"
        fen_char = piece if color == "w" else piece.lower()

        self.board[row][col] = fen_char
        self.buttons[row][col].config(text=piece)

    def calculate_best_move(self):
        try:
            fen = self.generate_fen()
        except ValueError as e:
            messagebox.showerror("Invalid position", str(e))
            return

        self.status_var.set("Analyzing...")
        self.submit_button.config(state="disabled")
        self.root.update_idletasks()

        try:
            best_move = self.run_stockfish(fen)
        except Exception as e:
            messagebox.showerror("Stockfish error", str(e))
            self.status_var.set("Analysis failed")
            self.submit_button.config(state="normal")
            return

        self.submit_button.config(state="normal")
        self.status_var.set("Ready")
        messagebox.showinfo("Best Move", f"Best move: {best_move}")

    def generate_fen(self):
        """Convert the board into a valid FEN string (rank 8 first, as FEN expects)."""
        fen_board = ""
        for row in self.board:
            empty = 0
            row_data = ""
            for piece in row:
                if piece:
                    if empty:
                        row_data += str(empty)
                        empty = 0
                    row_data += piece
                else:
                    empty += 1
            if empty:
                row_data += str(empty)
            fen_board += row_data + "/"

        # Both kings are required for a legal position.
        if "K" not in fen_board or "k" not in fen_board:
            raise ValueError("You must place both a white king (K) and a black king (k).")

        active = self.mover_var.get()
        return fen_board[:-1] + f" {active} - - 0 1"

    def run_stockfish(self, fen):
        """Run Stockfish via the UCI protocol and return the best move."""
        engine = subprocess.Popen(
            [resolve_stockfish_path()],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            universal_newlines=True,
        )

        engine.stdin.write("uci\n")
        engine.stdin.write(f"position fen {fen}\n")
        engine.stdin.write("go depth 12\n")
        engine.stdin.flush()

        best_move = None
        for line in engine.stdout:
            line = line.strip()
            if line.startswith("bestmove"):
                parts = line.split()
                if len(parts) >= 2 and parts[1] != "(none)":
                    best_move = parts[1]
                break

        engine.kill()
        if best_move is None:
            raise RuntimeError("Stockfish did not return a move.")
        return best_move


if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = ZiffieChessTool(root)
        root.mainloop()
    except FileNotFoundError as e:
        messagebox.showerror("Setup error", str(e))
