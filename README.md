# ♟️ Chess: Minimax & Alpha-Beta Pruning Chess Bot

A lightweight Python chess engine built using adversarial search techniques, including Minimax and Alpha-Beta pruning. The bot evaluates board states using a handcrafted evaluation function and visualizes game tree decisions.

---

## 🚀 Project Overview

This project implements a terminal-based chess engine that:

- Evaluates board states using **material, mobility**, and **center control**
- Applies **Minimax** and **Alpha-Beta pruning** for intelligent move selection
- Includes **game tree visualizations** using NetworkX and custom heuristics
- Supports CLI play and integration with GUI chess engines (e.g. Easy Chess GUI)

---

## 🧠 Core Algorithms

- `minimax(board, depth, alpha, beta, maximizing_player)`  
- `iterative_deepening(board, max_depth, time_limit)`
- `evaluate_board(board)` for score calculation
- `build_minimax_tree()` for alpha-beta tree structure visualization

---

## 📐 Evaluation Strategy

We use the following evaluation function to score positions from White’s perspective:

$$
f(\text{board}) = \text{Material} + 0.1 \times (\text{Mobility}_W - \text{Mobility}_B) + \text{CenterControl}
$$

### Weights:

- **Material**: Sum of all piece values (+ for White, – for Black)
- **Mobility**: Legal move count difference
- **Center Control**: Bonus for knights/pawns on central squares

| Piece   | Value |
|---------|-------|
| Pawn    | 100   |
| Knight  | 320   |
| Bishop  | 330   |
| Rook    | 500   |
| Queen   | 900   |
| King    | 0     |

---

## 📊 Tree Visualizations

### Minimax Tree

![Minimax](./output/minimax_tree.png)

### Alpha-Beta Pruning

![AlphaBeta](./output/AlphaBeta_Result.png)

---

## 🕹️ How to Play / Visualize

```bash
# Install requirements
pip install chess pydot networkx pyinstaller

# Run the bot against UCI commands
python boba_slayer.py

# Visualize a Minimax tree from a fixed opening
python boba_slayer.py draw
