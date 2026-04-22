import streamlit as st
import networkx as nx
import itertools
import matplotlib.pyplot as plt


# Graph Initialization
def create_graph():

    G = nx.Graph()

    # Node types
    normal_nodes = ["A","B","C","D","E"]
    refugee_nodes = ["R1","R2","R3"]
    camp_nodes = ["C1","C2"]

    for n in normal_nodes:
        G.add_node(n, type="normal")

    for r in refugee_nodes:
        G.add_node(r, type="refugee")

    for c in camp_nodes:
        G.add_node(c, type="camp")

    edges = [
        ("A","B",5),
        ("A","C",4),
        ("B","D",6),
        ("C","D",3),
        ("C","E",6),
        ("D","R1",4),
        ("E","R2",5),
        ("B","R3",7),
        ("R1","C1",6),
        ("R2","C1",5),
        ("R3","C2",4),
        ("D","C2",8)
    ]

    for u,v,w in edges:
        G.add_edge(u,v,
                   base_time=w,
                   risk=1,
                   state="intact",
                   weight=w)

    return G


# Edge Damage Implementation
def update_edge_weights(G, alpha):

    beta = 1 - alpha

    max_time = max(nx.get_edge_attributes(G,"base_time").values())
    max_risk = 5

    for u,v,data in G.edges(data=True):

        time = data["base_time"]
        risk = data["risk"]

        state = data["state"]

        if state == "light":
            time *= 1.5
            risk *= 1.2

        elif state == "heavy":
            time *= 3
            risk *= 2

        elif state == "blocked":
            data["weight"] = float("inf")
            continue

        norm_time = time / max_time
        norm_risk = risk / max_risk

        data["weight"] = alpha*norm_time + beta*norm_risk


# Distance Matrix
def compute_distance_matrix(G, nodes):

    matrix = {}

    for n in nodes:
        dist = nx.single_source_dijkstra_path_length(G, n, weight="weight")
        matrix[n] = dist

    return matrix


# Refugee Evacuation Optimizer
# (Currently Brute Force TSP)
def solve_refugee_order(start, refugees, camps, dist_matrix):

    best_cost = float("inf")
    best_route = None
    best_camp = None

    for perm in itertools.permutations(refugees):

        cost = dist_matrix[start][perm[0]]

        for i in range(len(perm)-1):
            cost += dist_matrix[perm[i]][perm[i+1]]

        last = perm[-1]

        for camp in camps:
            total = cost + dist_matrix[last][camp]

            if total < best_cost:
                best_cost = total
                best_route = perm
                best_camp = camp

    return best_route, best_camp, best_cost


# Path Reconstruction
def reconstruct_path(G, start, route, camp):

    full_path = []
    current = start

    for r in route:

        path = nx.shortest_path(G, current, r, weight="weight")
        full_path += path[:-1]
        current = r

    path = nx.shortest_path(G, current, camp, weight="weight")
    full_path += path

    return full_path


# Graph Visualization
def draw_graph(G, route=None):

    pos = nx.spring_layout(G, seed=42)

    colors = []

    for n,data in G.nodes(data=True):

        if data["type"] == "refugee":
            colors.append("red")

        elif data["type"] == "camp":
            colors.append("green")

        else:
            colors.append("gray")

    nx.draw(G,pos,
            with_labels=True,
            node_color=colors,
            node_size=900)

    if route:

        edges = list(zip(route, route[1:]))

        nx.draw_networkx_edges(
            G,pos,
            edgelist=edges,
            edge_color="blue",
            width=3
        )

    st.pyplot(plt.gcf())
    plt.clf()


# Streamlit App
def main():

    st.title("Evacuation Route Optimizer")

    if "graph" not in st.session_state:
        st.session_state.graph = create_graph()

    G = st.session_state.graph

    nodes = list(G.nodes())

    refugees = [n for n,d in G.nodes(data=True) if d["type"]=="refugee"]
    camps = [n for n,d in G.nodes(data=True) if d["type"]=="camp"]
    normals = [n for n,d in G.nodes(data=True) if d["type"]=="normal"]

    start = st.selectbox("Select Starting Node", normals)

    alpha = st.slider("Speed Priority (vs Safety)",0.0,1.0,0.5)

    st.subheader("Road Damage Control")

    edges = list(G.edges())

    for u,v in edges:

        state = st.selectbox(
            f"{u}-{v}",
            ["intact","light","heavy","blocked"],
            key=f"{u}-{v}"
        )

        G[u][v]["state"] = state

    update_edge_weights(G, alpha)

    if st.button("Compute Optimal Route"):

        important = [start] + refugees + camps

        dist_matrix = compute_distance_matrix(G, important)

        route, camp, cost = solve_refugee_order(
            start, refugees, camps, dist_matrix
        )

        path = reconstruct_path(G, start, route, camp)

        st.write("Pickup Order:", route)
        st.write("Camp:", camp)
        st.write("Total Cost:", round(cost,3))

        draw_graph(G, path)

    else:
        draw_graph(G)


if __name__ == "__main__":
    main()
