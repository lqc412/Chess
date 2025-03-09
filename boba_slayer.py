#!/usr/bin/env python
import chess
import random
import sys
import time

try:
    import pydot  # If missing, install via: pip install pydot
    HAS_PYDOT = True
except ImportError:
    HAS_PYDOT = False

# Global board object
board = chess.Board()

def evaluate_board(board):
    """
    Evaluate a given chess board position from the perspective of White.
    Returns a positive score if White is favored, or a negative score if Black is favored.
    """
    if board.is_checkmate():
        return -10000 if board.turn else 10000
    if board.is_stalemate() or board.is_insufficient_material():
        return 0

    piece_values = {
        chess.PAWN: 100,
        chess.KNIGHT: 320,
        chess.BISHOP: 330,
        chess.ROOK: 500,
        chess.QUEEN: 900,
        chess.KING: 0  # We do not score the king in material calculations.
    }

    # Material (piece) score
    material_score = 0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            value = piece_values[piece.piece_type]
            material_score += value if piece.color == chess.WHITE else -value

    # Mobility: difference in the number of legal moves
    original_turn = board.turn
    board.turn = chess.WHITE
    white_mobility = len(list(board.legal_moves))
    board.turn = chess.BLACK
    black_mobility = len(list(board.legal_moves))
    board.turn = original_turn
    mobility_score = 0.1 * (white_mobility - black_mobility)

    # Basic center control: pawns and knights on central squares
    center_squares = [chess.E4, chess.D4, chess.E5, chess.D5]
    center_control = 0
    for csq in center_squares:
        pc = board.piece_at(csq)
        if pc and pc.piece_type in [chess.PAWN, chess.KNIGHT]:
            center_control += 20 if pc.color == chess.WHITE else -20

    total_score = material_score + mobility_score + center_control
    return total_score

def order_moves(board, limit_top_moves=False):
    """
    Order moves heuristically to improve alpha-beta pruning efficiency.
    If limit_top_moves=True, only return a limited subset (e.g., top 7) after sorting.
    """
    moves = list(board.legal_moves)
    scored_moves = []

    for move in moves:
        score = 0

        # Capture moves first
        if board.is_capture(move):
            score += 50
            target_piece = board.piece_at(move.to_square)
            if target_piece:
                # Reward capturing higher-value pieces
                if target_piece.piece_type == chess.QUEEN:
                    score += 900
                elif target_piece.piece_type == chess.ROOK:
                    score += 500
                elif target_piece.piece_type in [chess.BISHOP, chess.KNIGHT]:
                    score += 330
                elif target_piece.piece_type == chess.PAWN:
                    score += 100

        # Check moves
        board.push(move)
        if board.is_check():
            score += 25
        board.pop()

        # Promotion bonus
        if move.promotion:
            score += 800

        scored_moves.append((move, score))

    # Sort by descending score
    scored_moves.sort(key=lambda x: x[1], reverse=True)
    if limit_top_moves and len(scored_moves) > 7:
        scored_moves = scored_moves[:7]

    return [mv for mv, sc in scored_moves]

def minimax(board, depth, alpha=float('-inf'), beta=float('inf'), maximizing_player=True):
    """
    Minimax algorithm with alpha-beta pruning.
    Returns (best_score, best_move).
    """
    if depth == 0 or board.is_game_over():
        return evaluate_board(board), None

    best_move = None
    if maximizing_player:
        max_eval = float('-inf')
        moves = order_moves(board, limit_top_moves=(depth < 3))
        for move in moves:
            board.push(move)
            eval_child, _ = minimax(board, depth - 1, alpha, beta, False)
            board.pop()

            if eval_child > max_eval:
                max_eval = eval_child
                best_move = move
            alpha = max(alpha, max_eval)
            if beta <= alpha:
                # Beta pruning
                break
        return max_eval, best_move
    else:
        min_eval = float('inf')
        moves = order_moves(board, limit_top_moves=(depth < 3))
        for move in moves:
            board.push(move)
            eval_child, _ = minimax(board, depth - 1, alpha, beta, True)
            board.pop()

            if eval_child < min_eval:
                min_eval = eval_child
                best_move = move
            beta = min(beta, min_eval)
            if beta <= alpha:
                # Alpha pruning
                break
        return min_eval, best_move

# ======== Visualization and Tree-Building Code ========

def build_minimax_tree(board, depth, alpha, beta, maximizing_player, move_uci=None):
    """
    Build the search tree structure for visualization. Returns a dict:
        {
            'id': unique ID for the node,
            'move': e.g., 'e2e4' or None if root,
            'score': final minimax score at this node,
            'alpha': alpha value upon entering this node,
            'beta': beta value upon entering this node,
            'pruned': boolean, whether this node was pruned,
            'children': list of child nodes
        }
    """
    node = {
        'id': id(board) ^ random.getrandbits(16),
        'move': move_uci,
        'alpha': alpha,
        'beta': beta,
        'pruned': False,
        'children': [],
        'score': None
    }

    if depth == 0 or board.is_game_over():
        node['score'] = evaluate_board(board)
        return node

    if maximizing_player:
        max_eval = float('-inf')
        moves = order_moves(board, limit_top_moves=True)[:3]
        for move in moves:
            board.push(move)
            child = build_minimax_tree(board, depth - 1, alpha, beta, False, move_uci=move.uci())
            board.pop()

            val = child['score']
            if val > max_eval:
                max_eval = val
            alpha = max(alpha, val)

            node['children'].append(child)

            if beta <= alpha:
                # A cutoff occurred. Mark the rest of the siblings as pruned.
                remaining_moves = moves[moves.index(move)+1:]
                for sibling_move in remaining_moves:
                    pruned_node = {
                        'id': id(board) ^ random.getrandbits(16),
                        'move': sibling_move.uci(),
                        'score': None,
                        'alpha': alpha,
                        'beta': beta,
                        'pruned': True,  # this sibling is never searched
                        'children': []
                    }
                    node['children'].append(pruned_node)
                break

        node['score'] = max_eval
        return node
    else:
        min_eval = float('inf')
        moves = order_moves(board, limit_top_moves=True)[:3]
        for move in moves:
            board.push(move)
            child = build_minimax_tree(board, depth - 1, alpha, beta, True, move_uci=move.uci())
            board.pop()

            val = child['score']
            if val < min_eval:
                min_eval = val
            beta = min(beta, val)

            node['children'].append(child)

            if beta <= alpha:
                # A cutoff occurred. Mark the rest of the siblings as pruned.
                remaining_moves = moves[moves.index(move)+1:]
                for sibling_move in remaining_moves:
                    pruned_node = {
                        'id': id(board) ^ random.getrandbits(16),
                        'move': sibling_move.uci(),
                        'score': None,
                        'alpha': alpha,
                        'beta': beta,
                        'pruned': True,
                        'children': []
                    }
                    node['children'].append(pruned_node)
                break

        node['score'] = min_eval
        return node

def draw_tree_dot(root_node, graph=None, parent_id=None, edge_label=None):
    """
    Recursively add nodes to a pydot graph for visualization.
    If a node is pruned=True, color it lightgray and its edge red.
    """
    if graph is None:
        graph = pydot.Dot("MinimaxTree", graph_type='digraph', rankdir="TB")

    label_parts = []
    if root_node['move']:
        label_parts.append(f"move={root_node['move']}")
    if root_node['score'] is not None:
        label_parts.append(f"score={root_node['score']}")
    label_parts.append(f"alpha={root_node['alpha']}, beta={root_node['beta']}")

    node_color = "white"
    if root_node['pruned']:
        node_color = "lightgray"

    this_node = pydot.Node(
        str(root_node['id']),
        label="\n".join(label_parts),
        style="filled",
        fillcolor=node_color,
        shape="box"
    )
    graph.add_node(this_node)

    if parent_id is not None:
        edge = pydot.Edge(str(parent_id), str(root_node['id']), label=edge_label)
        if root_node['pruned']:
            edge.set_color("red")
        graph.add_edge(edge)

    for child in root_node['children']:
        move_label = child['move']
        draw_tree_dot(child, graph, parent_id=root_node['id'], edge_label=move_label)

    return graph

def generate_minimax_visualization():
    """
    1) Set the board to a given FEN (by default, QGD).
    2) Build a 4-ply minimax tree, expanding only top 3 moves.
    3) Perform alpha-beta pruning; pruned siblings become red edges and gray nodes.
    4) Output a dot file for Graphviz visualization.
    """
    fen_qgd = "rnbqkbnr/pppp1ppp/4p3/3p4/2PP4/8/PP2PPPP/RNBQKBNR w KQkq - 0 3"
    board.set_fen(fen_qgd)

    root = build_minimax_tree(board, depth=4, alpha=float('-inf'), beta=float('inf'), maximizing_player=True)
    final_score = root['score']

    # Identify best move at root
    best_move_child = None
    best_eval = float('-inf')
    for child in root['children']:
        if child['score'] is not None and child['score'] > best_eval:
            best_eval = child['score']
            best_move_child = child
    best_move_uci = best_move_child['move'] if best_move_child else "None"

    print("=== Minimax Tree Visualization ===")
    print(f"Root Evaluation: {final_score}")
    print(f"Chosen best move for White: {best_move_uci}")

    if not HAS_PYDOT:
        print("warning: pydot is not installed, cannot generate graph visualization.")
        return

    graph = draw_tree_dot(root)
    graph.set_label(f"Final Minimax Value: {final_score} | Best Move: {best_move_uci}")

    output_dot = "minimax_tree.dot"
    graph.write_raw(output_dot)
    print(f"Dot file created: {output_dot}")
    print("Use: dot -Tpng minimax_tree.dot -o minimax_tree.png")
    print("Then annotate alpha, beta, and pruning decisions on the image.")

def iterative_deepening(board, max_depth, time_limit=5.0):
    """
    Iterative deepening search with a time limit.
    """
    start_time = time.time()
    best_move = None
    try:
        best_move = random.choice(list(board.legal_moves))
    except IndexError:
        return None

    for depth in range(1, max_depth + 1):
        if time.time() - start_time > time_limit:
            break
        score, move = minimax(board, depth, float('-inf'), float('inf'), board.turn == chess.WHITE)
        if move:
            best_move = move
    return best_move

def make_best_move(board):
    """
    Find and return the best move for the current board position.
    """
    if board.is_game_over():
        return None
    return iterative_deepening(board, max_depth=3, time_limit=1.0)

def uci(msg: str):
    """
    Handle UCI protocol messages.
    """
    global board
    if msg == "uci":
        print("id name Boba Slayer")
        print("id author Quancheng Li")
        print("uciok")
        sys.stdout.flush()
    elif msg == "isready":
        print("readyok")
        sys.stdout.flush()
    elif msg.startswith("position startpos moves"):
        board.clear()
        board.set_fen(chess.STARTING_FEN)
        moves = msg.split()[3:]
        for mv in moves:
            board.push(chess.Move.from_uci(mv))
    elif msg.startswith("position fen"):
        parts = msg.split(" moves ")
        fen_str = parts[0].removeprefix("position fen ")
        board.set_fen(fen_str)
        if len(parts) > 1:
            moves = parts[1].split()
            for mv in moves:
                board.push(chess.Move.from_uci(mv))
    elif msg.startswith("go"):
        best_move = make_best_move(board)
        if best_move:
            print(f"bestmove {best_move}")
        else:
            print("bestmove 0000")
        sys.stdout.flush()
    elif msg == "quit":
        sys.exit(0)

def main():
    """
    Main entry point. If 'draw' is passed as an argument, generate the minimax visualization.
    Otherwise, run as a standard UCI engine.
    """
    if len(sys.argv) > 1 and sys.argv[1] == "draw":
        generate_minimax_visualization()
        sys.exit(0)

    try:
        while True:
            line = input()
            uci(line)
    except EOFError:
        pass
    except Exception as e:
        print(f"info string Error: {e}")
        sys.stdout.flush()
        sys.exit(1)

if __name__ == "__main__":
    main()
