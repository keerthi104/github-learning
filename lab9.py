
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# STEP 1: Load the Facebook Dataset
file_path = "facebook_combined.txt"

try:
    # Read the edge list
    df = pd.read_csv(
        file_path,
        sep=" ",
        names=["user_1", "user_2"]
    )

    print("Dataset loaded successfully!")

except FileNotFoundError:
    print("Error: Dataset file not found.")
    print("Place 'facebook_combined.txt' in the same folder as the notebook.")
    raise

# Display first five rows
print("\nFirst 5 rows of the dataset:")
print(df.head())

# STEP 2: Create the Social Network Graph
# Create an undirected graph from the dataframe
G = nx.from_pandas_edgelist(
    df,
    source="user_1",
    target="user_2"
)

print("\nGraph Created Successfully!")
print("Number of Users (Nodes):", G.number_of_nodes())
print("Number of Friendships (Edges):", G.number_of_edges())

# STEP 3: Visualize a Small Portion of the Graph
# Visualize only the first 50 nodes
sample_nodes = list(G.nodes())[:50]
subgraph = G.subgraph(sample_nodes)

plt.figure(figsize=(8, 6))

nx.draw(
    subgraph,
    with_labels=True,
    node_color="lightblue",
    edge_color="gray",
    node_size=350,
    font_size=8
)

plt.title("Sample Facebook Social Network")
plt.show()

# STEP 4: Degree Centrality
# Degree = Number of direct friends
degree = dict(G.degree())

top_degree = sorted(
    degree.items(),
    key=lambda x: x[1],
    reverse=True
)[:5]

# STEP 5: Betweenness Centrality
# Approximation used for faster execution
print("\nCalculating Betweenness Centrality...")

betweenness = nx.betweenness_centrality(
    G,
    k=200,
    seed=42
)

top_betweenness = sorted(
    betweenness.items(),
    key=lambda x: x[1],
    reverse=True
)[:5]

# STEP 6: Closeness Centrality
print("Calculating Closeness Centrality...")

closeness = nx.closeness_centrality(G)

top_closeness = sorted(
    closeness.items(),
    key=lambda x: x[1],
    reverse=True
)[:5]

# STEP 7: Display Results
print("\n==========================================")
print("Top 5 Users based on Degree Centrality")
print("==========================================")

for user, value in top_degree:
    print(f"User {user} --> {value} friends")

print("\n==========================================")
print("Top 5 Users based on Betweenness Centrality")
print("==========================================")

for user, value in top_betweenness:
    print(f"User {user} --> {value:.4f}")

print("\n==========================================")
print("Top 5 Users based on Closeness Centrality")
print("==========================================")

for user, value in top_closeness:
    print(f"User {user} --> {value:.4f}")

print("\nProgram Executed Successfully!")