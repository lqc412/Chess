import networkx as nx
import matplotlib.pyplot as plt

###
# 
# Add your Minimax and AlphaBeta search code here
#
# @Quancheng Li
####

def create_game_tree():
    """
    Creates a directional graph (DiGraph) with nodes 0..14:
      - 0: MAX
      - 1..2: MIN
      - 3..6: MAX
      - 7..14: LEAF with assigned static values
    """
    G = nx.DiGraph()

    # --- Add nodes ---
    # Root node (MAX)
    G.add_node(0, player="MAX", value=None)

    # Second level (MIN)
    G.add_node(1, player="MIN", value=None)
    G.add_node(2, player="MIN", value=None)

    # Third level (MAX)
    G.add_node(3, player="MAX", value=None)
    G.add_node(4, player="MAX", value=None)
    G.add_node(5, player="MAX", value=None)
    G.add_node(6, player="MAX", value=None)

    # Leaf nodes (fixed values)
    G.add_node(7, player="LEAF", value=3)
    G.add_node(8, player="LEAF", value=5)
    G.add_node(9, player="LEAF", value=6)
    G.add_node(10, player="LEAF", value=9)
    G.add_node(11, player="LEAF", value=1)
    G.add_node(12, player="LEAF", value=2)
    G.add_node(13, player="LEAF", value=0)
    G.add_node(14, player="LEAF", value=-1)

    # --- Add edges with labels ---
    G.add_edge(0, 1, label="L")
    G.add_edge(0, 2, label="R")

    G.add_edge(1, 3, label="L")
    G.add_edge(1, 4, label="R")
    G.add_edge(2, 5, label="L")
    G.add_edge(2, 6, label="R")

    G.add_edge(3, 7, label="L")
    G.add_edge(3, 8, label="R")
    G.add_edge(4, 9, label="L")
    G.add_edge(4, 10, label="R")
    G.add_edge(5, 11, label="L")
    G.add_edge(5, 12, label="R")
    G.add_edge(6, 13, label="L")
    G.add_edge(6, 14, label="R")

    return G

def visualize_tree(G, title="Game Tree",
                   highlight_path=None,  # list of edges to highlight
                   pruned_edges=None):   # list of edges to color red (pruned)
    """
    Draws the tree with node labels and edge labels.
    highlight_path: optional list of edges to highlight in blue
    pruned_edges: optional list of edges to show as pruned (red dashed)
    """
    plt.figure(figsize=(12, 8))

    # Hard-coded positions
    pos = {
        0: (0, 3),   # Root
        1: (-2, 2), 2: (2, 2),
        3: (-3, 1), 4: (-1, 1), 5: (1, 1), 6: (3, 1),
        7: (-3.5, 0), 8: (-2.5, 0), 9: (-1.5, 0), 10: (-0.5, 0),
        11: (0.5, 0), 12: (1.5, 0), 13: (2.5, 0), 14: (3.5, 0),
    }

    # Draw base nodes & edges
    nx.draw(G, pos, with_labels=False, node_size=700,
            node_color='lightblue', arrows=True)

    # Leaf nodes in green squares
    leaf_nodes = [n for n in G.nodes if G.nodes[n]['player'] == "LEAF"]
    nx.draw_networkx_nodes(G, pos, nodelist=leaf_nodes,
                           node_shape='s', node_color='lightgreen', node_size=700)

    # Write each node's numeric value (if known)
    for n in G.nodes:
        val = G.nodes[n]['value']
        if val is not None:
            plt.text(pos[n][0], pos[n][1], str(val),
                     fontsize=12, ha='center', va='center', color='black')

    # Label the player types
    for n in G.nodes:
        if G.nodes[n]['player'] != "LEAF":
            player_type = G.nodes[n]['player']  # "MAX" or "MIN"
            plt.text(pos[n][0], pos[n][1] + 0.3, player_type,
                     fontsize=10, ha='center', va='center')

    # Edge labels
    edge_labels = {(u, v): G[u][v]['label'] for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)

    # Highlight a path (in blue)
    if highlight_path:
        nx.draw_networkx_edges(G, pos, edgelist=highlight_path,
                               width=3.0, edge_color='blue')

    # Pruned edges (in red dashed)
    if pruned_edges:
        nx.draw_networkx_edges(G, pos, edgelist=pruned_edges,
                               width=2.0, edge_color='red', style='dashed')

    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    filename = f"{title.replace(' ', '_')}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Image saved as: {filename}")
    plt.show()

def minimax(G, node, maximizing):
    """
    Classic Minimax on the small tree G.
    G[node]['player'] should be "MAX", "MIN", or "LEAF".
    Returns the computed value for G[node], storing it in G.nodes[node]['value'].
    """
    # Leaf => return stored value
    if G.nodes[node]['player'] == "LEAF":
        return G.nodes[node]['value']

    children = list(G.successors(node))
    if maximizing:
        value = float('-inf')
        for c in children:
            child_val = minimax(G, c, False)
            value = max(value, child_val)
        G.nodes[node]['value'] = value
        return value
    else:
        value = float('inf')
        for c in children:
            child_val = minimax(G, c, True)
            value = min(value, child_val)
        G.nodes[node]['value'] = value
        return value

def get_best_move(G, node):
    """
    For the current node, returns an (node, child) edge leading
    to the child's value = parent's node value.
    """
    parent_val = G.nodes[node]['value']
    for child in G.successors(node):
        if G.nodes[child]['value'] == parent_val:
            return (node, child)
    return None

def alphabeta(G, node, alpha, beta, maximizing, pruned_edges=None):
    """
    AlphaBeta on the small toy tree G.
    Returns (value, pruned_edges).
    pruned_edges is a list of (u,v) edges that got pruned.
    """
    if pruned_edges is None:
        pruned_edges = []

    if G.nodes[node]['player'] == "LEAF":
        return G.nodes[node]['value'], pruned_edges

    children = list(G.successors(node))
    if maximizing:
        value = float('-inf')
        for i, c in enumerate(children):
            child_val, pruned_edges = alphabeta(G, c, alpha, beta, False, pruned_edges)
            value = max(value, child_val)
            alpha = max(alpha, value)

            if beta <= alpha:
                # We prune the *remaining siblings* not yet visited
                for sibling in children[i+1:]:
                    # Mark the (node, sibling) edge as pruned
                    if (node, sibling) not in pruned_edges:
                        pruned_edges.append((node, sibling))
                break
        G.nodes[node]['value'] = value
        return value, pruned_edges
    else:
        value = float('inf')
        for i, c in enumerate(children):
            child_val, pruned_edges = alphabeta(G, c, alpha, beta, True, pruned_edges)
            value = min(value, child_val)
            beta = min(beta, value)

            if beta <= alpha:
                # Prune the remaining siblings
                for sibling in children[i+1:]:
                    if (node, sibling) not in pruned_edges:
                        pruned_edges.append((node, sibling))
                break
        G.nodes[node]['value'] = value
        return value, pruned_edges

def get_best_path(G, start):
    """
    Walks from 'start' down to a leaf by always choosing
    a child whose value matches the parent's.
    Returns a list of edges (parent->child).
    """
    path_edges = []
    current = start
    while G.nodes[current]['player'] != "LEAF":
        parent_val = G.nodes[current]['value']
        # find child that matches
        next_c = None
        for c in G.successors(current):
            if G.nodes[c]['value'] == parent_val:
                next_c = c
                break
        if next_c is None:
            # no child matches => break
            break
        path_edges.append((current, next_c))
        current = next_c
    return path_edges

def main():
    # -- Step 1: Create & visualize the initial game tree
    G = create_game_tree()
    visualize_tree(G, title="Initial_Game_Tree")

    # -- Minimax steps
    G_minimax = create_game_tree()
    minimax_value = minimax(G_minimax, 0, True)
    print(f"Minimax value at root: {minimax_value}")

    # Build the path chosen by Minimax:
    minimax_path = get_best_path(G_minimax, 0)
    visualize_tree(G_minimax, 
                   title="Minimax_Result",
                   highlight_path=minimax_path)

    # -- Alpha-Beta steps
    G_alphabeta = create_game_tree()
    ab_value, pruned_edges = alphabeta(G_alphabeta, 0, float('-inf'), float('inf'), True)
    print(f"Alpha-Beta value at root: {ab_value}")
    print(f"Pruned edges: {pruned_edges}")

    # Build the path chosen by alpha-beta (could be same or different from minimax):
    alphabeta_path = get_best_path(G_alphabeta, 0)
    visualize_tree(G_alphabeta,
                   title="AlphaBeta_Result",
                   highlight_path=alphabeta_path,
                   pruned_edges=pruned_edges)

if __name__ == "__main__":
    main()
