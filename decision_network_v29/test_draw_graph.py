import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Ellipse, Polygon

# Configuration variables
font_size = 9  # Adjust this value to change the font size (e.g., 12, 14, etc.)

# Create a directed graph
G = nx.DiGraph()

# Add nodes (labels will be the node names by default)
G.add_nodes_from(["A", "B", "C", "D"])

# Add directed edges
G.add_edges_from([("A", "B"), ("B", "C"), ("A", "C"), ("C", "D")])

# Specify node positions as a dictionary: node -> (x, y)
pos = {
    "A": (0, 0),  # Node A at (0, 0)
    "B": (2, 1),  # Node B at (2, 1) for better spacing
    "C": (4, 0),  # Node C at (4, 0)
    "D": (6, 0.5)  # Node D at (6, 0.5)
}

# Create figure and axis
fig, ax = plt.subplots()

# Set axis limits to fit the graph (needed for accurate transformations)
ax.set_xlim(-1, 7)
ax.set_ylim(-1, 2)

# Calculate points per data unit (for dynamic margin in points)
pt_per_unit = ax.transData.transform((1, 0))[0] - ax.transData.transform((0, 0))[0]

# Desired shrink distance in data units (approximate node "radius" to pull arrowhead back)
desired_shrink_data = 0.4
margin = pt_per_unit * desired_shrink_data

# Draw the edges with arrows, applying margin to pull back arrowheads
nx.draw_networkx_edges(G, pos, ax=ax, arrows=True, arrowstyle='->', arrowsize=20,
                       node_size=0, min_target_margin=margin)

# Draw custom node shapes using patches with white interior
# Node A: rectangle
rect_width = 1.0
rect_height = 0.5
ax.add_patch(Rectangle((pos["A"][0] - rect_width/2, pos["A"][1] - rect_height/2),
                       rect_width, rect_height, facecolor='white', edgecolor='black'))

# Node B: oval (ellipse)
ell_width = 1.2
ell_height = 0.6
ax.add_patch(Ellipse((pos["B"][0], pos["B"][1]),
                     ell_width, ell_height, facecolor='white', edgecolor='black'))

# Node C: rectangle
ax.add_patch(Rectangle((pos["C"][0] - rect_width/2, pos["C"][1] - rect_height/2),
                       rect_width, rect_height, facecolor='white', edgecolor='black'))

# Node D: diamond (using Polygon, with equal axes)
diamond_size = 0.5  # half-width/height (equal for square diamond)
diamond_points = [
    (pos["D"][0], pos["D"][1] + diamond_size),  # top
    (pos["D"][0] + diamond_size, pos["D"][1]),  # right
    (pos["D"][0], pos["D"][1] - diamond_size),  # bottom
    (pos["D"][0] - diamond_size, pos["D"][1])   # left
]
ax.add_patch(Polygon(diamond_points, facecolor='white', edgecolor='black'))

# Draw node labels (centered on positions, with normal font weight and custom font size)
nx.draw_networkx_labels(G, pos, ax=ax, font_weight='normal', font_size=font_size)

# Turn off axis
plt.axis('off')

# Save the figure as a PNG file
plt.savefig("directed_graph_custom_shapes_updated.png")

# Optional: Close the plot to free up memory
plt.close()