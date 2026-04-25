import streamlit as st
import networkx as nx
import itertools
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from io import BytesIO
import base64
import math


# Graph Initialization
def create_graph():

    G = nx.Graph()

    # Node types
    normal_nodes = ["A", "B", "C", "D", "E"]
    refugee_nodes = ["R1", "R2", "R3"]
    camp_nodes = ["C1", "C2"]

    for n in normal_nodes:
        G.add_node(n, type="normal")

    for r in refugee_nodes:
        G.add_node(r, type="refugee")

    for c in camp_nodes:
        G.add_node(c, type="camp")

    edges = [
        ("A", "B", 5),
        ("A", "C", 4),
        ("B", "D", 6),
        ("C", "D", 3),
        ("C", "E", 6),
        ("D", "R1", 4),
        ("E", "R2", 5),
        ("B", "R3", 7),
        ("R1", "C1", 6),
        ("R2", "C1", 5),
        ("R3", "C2", 4),
        ("D", "C2", 8)
    ]

    for u, v, w in edges:
        G.add_edge(
            u,
            v,
            base_time=w,
            risk=1,
            state="intact",
            weight=w
        )

    return G


# Edge Damage Implementation
def update_edge_weights(G, alpha):

    beta = 1 - alpha

    max_time = max(nx.get_edge_attributes(G, "base_time").values())
    max_risk = 5

    for u, v, data in G.edges(data=True):

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

        data["weight"] = alpha * norm_time + beta * norm_risk


# Distance Matrix
def compute_distance_matrix(G, nodes):

    matrix = {}

    for n in nodes:
        dist = nx.single_source_dijkstra_path_length(G, n, weight="weight")
        matrix[n] = dist

    return matrix


# Refugee Evacuation Optimizer
def solve_refugee_order(start, refugees, camps, dist_matrix):

    best_cost = float("inf")
    best_route = None
    best_camp = None

    for perm in itertools.permutations(refugees):

        if perm[0] not in dist_matrix[start]:
            continue

        cost = dist_matrix[start][perm[0]]
        valid = True

        for i in range(len(perm) - 1):
            if perm[i + 1] not in dist_matrix[perm[i]]:
                valid = False
                break

            cost += dist_matrix[perm[i]][perm[i + 1]]

        if not valid:
            continue

        last = perm[-1]

        for camp in camps:
            if camp not in dist_matrix[last]:
                continue

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
def draw_graph(G, route=None, start=None):

    pos = {
        "A": (5.0, 0.8),
        "B": (3.5, 0.8),
        "C": (5.8, 2.4),
        "D": (3.7, 2.4),
        "E": (7.2, 3.6),
        "R1": (3.8, 4.6),
        "R2": (8.2, 4.6),
        "R3": (2.1, 0.5),
        "C1": (6.1, 5.4),
        "C2": (2.4, 1.7)
    }

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=120)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Draw edges based on road damage state
    edge_style_map = {
        "intact": {"color": "gray", "width": 2, "style": "solid", "alpha": 0.8},
        "light": {"color": "orange", "width": 3, "style": "solid", "alpha": 0.9},
        "heavy": {"color": "red", "width": 4, "style": "solid", "alpha": 0.9},
        "blocked": {"color": "darkred", "width": 3, "style": "dashed", "alpha": 0.5},
    }

    for state, style in edge_style_map.items():
        state_edges = [
            (u, v)
            for u, v, data in G.edges(data=True)
            if data["state"] == state
        ]

        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=state_edges,
            edge_color=style["color"],
            width=style["width"],
            style=style["style"],
            alpha=style["alpha"],
            ax=ax
        )

    # Highlight optimal route if it exists
    if route:
        route_edges = list(zip(route, route[1:]))

        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=route_edges,
            edge_color="blue",
            width=5,
            ax=ax
        )

    # Node groups
    normal_nodes = [n for n, d in G.nodes(data=True) if d["type"] == "normal"]
    refugee_nodes = [n for n, d in G.nodes(data=True) if d["type"] == "refugee"]
    camp_nodes = [n for n, d in G.nodes(data=True) if d["type"] == "camp"]

    # Draw nodes
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=normal_nodes,
        node_color="gray",
        node_size=700,
        ax=ax
    )

    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=refugee_nodes,
        node_color="red",
        node_size=700,
        ax=ax
    )

    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=camp_nodes,
        node_color="green",
        node_size=700,
        ax=ax
    )

    # Highlight start node
    if start:
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=[start],
            node_color="gold",
            node_size=850,
            edgecolors="black",
            linewidths=2,
            ax=ax
        )

    # Node labels
    nx.draw_networkx_labels(
        G,
        pos,
        font_size=9,
        font_weight="bold",
        ax=ax
    )

    # Edge labels placed next to each road and rotated parallel to the line
    for u, v, data in G.edges(data=True):
        x1, y1 = pos[u]
        x2, y2 = pos[v]

        # Midpoint of the edge
        mx = (x1 + x2) / 2
        my = (y1 + y2) / 2

        # Direction vector of the edge
        dx = x2 - x1
        dy = y2 - y1
        length = math.sqrt(dx**2 + dy**2)

        if length == 0:
            continue

        # Offset perpendicular to the line so the text sits beside the line
        offset = 0.18
        ox = -dy / length * offset
        oy = dx / length * offset

        # Compute line angle
        angle = math.degrees(math.atan2(dy, dx))

        # Keep text upright
        if angle > 90:
            angle -= 180
        elif angle < -90:
            angle += 180

        label = f"T:{data['base_time']} | R:{data['risk']}"

        ax.text(
            mx + ox,
            my + oy,
            label,
            fontsize=7,
            ha="center",
            va="center",
            rotation=angle,
            rotation_mode="anchor",
            color="black",
            bbox=dict(
                facecolor="white",
                edgecolor="none",
                alpha=0.8,
                pad=0.2
            )
        )

    # Legend
    legend_items = [
        Line2D([0], [0], marker="o", color="w", label="Normal Node",
               markerfacecolor="gray", markersize=8),
        Line2D([0], [0], marker="o", color="w", label="Refugee Node",
               markerfacecolor="red", markersize=8),
        Line2D([0], [0], marker="o", color="w", label="Camp Node",
               markerfacecolor="green", markersize=8),
        Line2D([0], [0], marker="o", color="w", label="Start Node",
               markerfacecolor="gold", markeredgecolor="black", markersize=8),
        Line2D([0], [0], color="gray", lw=2, label="Intact Road"),
        Line2D([0], [0], color="orange", lw=3, label="Light Damage"),
        Line2D([0], [0], color="red", lw=4, label="Heavy Damage"),
        Line2D([0], [0], color="darkred", lw=3, linestyle="dashed", label="Blocked Road"),
        Line2D([0], [0], color="blue", lw=4, label="Optimal Route")
    ]

    ax.legend(
        handles=legend_items,
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        borderaxespad=0.0,
        fontsize=8
    )

    ax.set_title("Evacuation Map", fontsize=14)
    ax.axis("off")
    plt.tight_layout()

    # Save image to buffer
    buffer = BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight", dpi=120)
    buffer.seek(0)

    # Center the map image using HTML
    image_base64 = base64.b64encode(buffer.getvalue()).decode()

    st.markdown(
        f"""
        <div style="display: flex; justify-content: center; align-items: center; width: 100%;">
            <img src="data:image/png;base64,{image_base64}"
                 style="width: 650px; max-width: 100%; border-radius: 6px;">
        </div>
        """,
        unsafe_allow_html=True
    )

    plt.close(fig)


# Streamlit App
def main():

    st.set_page_config(
        page_title="Evacuation Route Optimizer",
        page_icon="🗺️",
        layout="wide"
    )

    st.markdown(
        "<h1 style='text-align: center;'>Evacuation Route Optimizer</h1>",
        unsafe_allow_html=True
    )

    if "graph" not in st.session_state:
        st.session_state.graph = create_graph()

    G = st.session_state.graph

    refugees = [n for n, d in G.nodes(data=True) if d["type"] == "refugee"]
    camps = [n for n, d in G.nodes(data=True) if d["type"] == "camp"]
    normals = [n for n, d in G.nodes(data=True) if d["type"] == "normal"]

    # Mission setup
    st.subheader("Mission Setup")

    setup_col1, setup_col2 = st.columns(2)

    with setup_col1:
        start = st.selectbox("Select starting node", normals)

    with setup_col2:
        alpha = st.slider("Speed priority vs. safety", 0.0, 1.0, 0.5)

    # Road damage controls
    st.subheader("Road Damage Control")

    edges = list(G.edges())
    road_col1, road_col2 = st.columns(2)

    for index, (u, v) in enumerate(edges):
        target_col = road_col1 if index % 2 == 0 else road_col2

        with target_col:
            state = st.selectbox(
                f"{u}-{v}",
                ["intact", "light", "heavy", "blocked"],
                key=f"{u}-{v}"
            )
            G[u][v]["state"] = state

    update_edge_weights(G, alpha)

    st.markdown("---")

    compute_clicked = st.button(
        "Compute Optimal Route",
        use_container_width=True
    )

    st.markdown(
        "<h2 style='text-align: center;'>Evacuation Map</h2>",
        unsafe_allow_html=True
    )

    if compute_clicked:
        try:
            important = [start] + refugees + camps
            dist_matrix = compute_distance_matrix(G, important)

            route, camp, cost = solve_refugee_order(
                start,
                refugees,
                camps,
                dist_matrix
            )

            if route is None or camp is None:
                st.error("No valid route found with the current road settings.")
                draw_graph(G, start=start)

            else:
                path = reconstruct_path(G, start, route, camp)

                result_col1, result_col2, result_col3 = st.columns(3)

                with result_col1:
                    st.metric("Selected Camp", camp)

                with result_col2:
                    st.metric("Total Cost", round(cost, 3))

                with result_col3:
                    st.metric("Refugee Stops", len(route))

                st.success("Optimal evacuation route computed.")

                st.markdown(
                    f"<p style='text-align: center; font-size: 18px;'><b>Pickup Order:</b> {' → '.join(route)}</p>",
                    unsafe_allow_html=True
                )

                st.markdown(
                    f"<p style='text-align: center; font-size: 18px;'><b>Full Path:</b> {' → '.join(path)}</p>",
                    unsafe_allow_html=True
                )

                draw_graph(G, path, start=start)

        except Exception:
            st.error("No feasible evacuation route exists under the current road settings.")
            draw_graph(G, start=start)

    else:
        draw_graph(G, start=start)


if __name__ == "__main__":
    main()
