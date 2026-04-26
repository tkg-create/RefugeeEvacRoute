import streamlit as st
import networkx as nx
import random
import math


# Graph Initialization
def create_graph(size):

    G = nx.Graph()

    # Config
    num_clusters = 4

    all_nodes = []
    cluster_nodes = {}

    grid_side = int(math.sqrt(num_clusters)) + 1
    cluster_centers = {}

    for c in range(num_clusters):
        gx = c % grid_side
        gy = c // grid_side

        cluster_centers[c] = (
            gx * random.uniform(4, 6),
            gy * random.uniform(4, 6)
        )

    # Distribute nodes across clusters so demo mode can use exactly 15 nodes
    base_nodes_per_cluster = size // num_clusters
    extra_nodes = size % num_clusters

    node_id = 0

    for c in range(num_clusters):

        cluster_nodes[c] = []

        target_count = base_nodes_per_cluster
        if c < extra_nodes:
            target_count += 1

        side = int(math.sqrt(target_count)) + 1

        for i in range(side):
            for j in range(side):

                if len(cluster_nodes[c]) >= target_count:
                    break

                name = f"N{node_id}"
                node_id += 1

                cx, cy = cluster_centers[c]

                pos = (
                    cx + i * 0.8 + random.uniform(-0.35, 0.35),
                    cy + j * 0.8 + random.uniform(-0.35, 0.35)
                )

                G.add_node(
                    name,
                    type="normal",
                    cluster=c,
                    pos=pos
                )

                cluster_nodes[c].append(name)
                all_nodes.append(name)

            if len(cluster_nodes[c]) >= target_count:
                break

    # Add Refugees
    num_refugee = max(2, int(0.25 * size))
    refugees = random.sample(all_nodes, min(num_refugee, len(all_nodes)))

    for r in refugees:
        G.nodes[r]["type"] = "refugee"

    # Add Camps
    num_camp = max(2, int(0.15 * size))

    outer_nodes = [
        n for n in all_nodes
        if G.nodes[n]["cluster"] in [0, num_clusters - 1]
        and G.nodes[n]["type"] != "refugee"
    ]

    if len(outer_nodes) < num_camp:
        outer_nodes = [
            n for n in all_nodes
            if G.nodes[n]["type"] != "refugee"
        ]

    camps = random.sample(outer_nodes, min(num_camp, len(outer_nodes)))

    for c in camps:
        G.nodes[c]["type"] = "camp"

    # Local Edges
    for c in cluster_nodes:

        nodes = cluster_nodes[c]

        for _ in range(len(nodes) * 2):

            if len(nodes) < 2:
                continue

            u, v = random.sample(nodes, 2)

            if not G.has_edge(u, v):

                base_time = random.randint(2, 6)

                G.add_edge(
                    u,
                    v,
                    base_time=base_time,
                    risk=random.randint(1, 3),
                    state="intact",
                    weight=base_time
                )

    # Highway Edges Between Clusters
    for _ in range(max(4, size // 2)):

        u, v = random.sample(all_nodes, 2)

        if G.nodes[u]["cluster"] != G.nodes[v]["cluster"]:

            if not G.has_edge(u, v):

                base_time = random.randint(5, 12)

                G.add_edge(
                    u,
                    v,
                    base_time=base_time,
                    risk=random.randint(2, 5),
                    state="intact",
                    weight=base_time
                )

    # Ensure each node has at least one connection
    for n in all_nodes:
        if G.degree(n) == 0:

            target = random.choice([x for x in all_nodes if x != n])
            base_time = random.randint(2, 10)

            G.add_edge(
                n,
                target,
                base_time=base_time,
                risk=random.randint(1, 5),
                state="intact",
                weight=base_time
            )

    # Ensure clusters are connected to each other
    for c in range(num_clusters - 1):

        u = random.choice(cluster_nodes[c])
        v = random.choice(cluster_nodes[c + 1])

        if not G.has_edge(u, v):

            base_time = random.randint(5, 12)

            G.add_edge(
                u,
                v,
                base_time=base_time,
                risk=random.randint(2, 5),
                state="intact",
                weight=base_time
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

    # Step 1 - Greedy Nearest Neighbor
    unvisited = set(refugees)
    current = start
    route = []

    while unvisited:
        nearest = None
        best_dist = float("inf")

        for r in unvisited:
            if r in dist_matrix[current]:
                d = dist_matrix[current][r]
                if d < best_dist:
                    best_dist = d
                    nearest = r

        if nearest is None:
            return None, None, float("inf")

        route.append(nearest)
        unvisited.remove(nearest)
        current = nearest

    # Step 2 - 2-opt Optimization
    def route_cost(candidate_route):

        if not candidate_route:
            return 0

        if candidate_route[0] not in dist_matrix[start]:
            return float("inf")

        cost = dist_matrix[start][candidate_route[0]]

        for i in range(len(candidate_route) - 1):

            if candidate_route[i + 1] not in dist_matrix[candidate_route[i]]:
                return float("inf")

            cost += dist_matrix[candidate_route[i]][candidate_route[i + 1]]

        return cost

    improved = True

    while improved:
        improved = False

        for i in range(len(route)):
            for j in range(i + 1, len(route)):

                new_route = route[:i] + route[i:j + 1][::-1] + route[j + 1:]

                if route_cost(new_route) < route_cost(route):
                    route = new_route
                    improved = True

    # Step 3 - Choose Best Camp
    best_cost = float("inf")
    best_camp = None

    last = route[-1]

    for camp in camps:
        if camp in dist_matrix[last]:
            total = route_cost(route) + dist_matrix[last][camp]

            if total < best_cost:
                best_cost = total
                best_camp = camp

    return route, best_camp, best_cost


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
def draw_interactive_graph(G, route=None, start=None):

    from pyvis.network import Network

    net = Network(height="650px", width="100%", bgcolor="white")

    pos = {
        n: G.nodes[n]["pos"]
        for n in G.nodes
    }

    route_edges = set(zip(route, route[1:])) if route else set()

    # Nodes
    for n, data in G.nodes(data=True):

        color = "gray"
        size = 6

        if data["type"] == "refugee":
            color = "red"
            size = 8

        elif data["type"] == "camp":
            color = "green"
            size = 8

        if start == n:
            color = "gold"
            size = 10

        x, y = pos[n]

        net.add_node(
            n,
            label=n if len(G.nodes) <= 20 else "",
            color=color,
            size=size,
            x=float(x) * 100,
            y=float(y) * 100,
            physics=False,
            title=f"{n} ({data['type']})"
        )

    # Edges
    for u, v, data in G.edges(data=True):

        color = "gray"
        width = 2
        dashes = False

        if data["state"] == "light":
            color = "orange"

        elif data["state"] == "heavy":
            color = "red"

        elif data["state"] == "blocked":
            color = "darkred"
            dashes = True

        if route and ((u, v) in route_edges or (v, u) in route_edges):
            color = "blue"
            width = 5
            dashes = False

        net.add_edge(
            u,
            v,
            color=color,
            width=width,
            dashes=dashes,
            title=f"Time: {data['base_time']}\nRisk: {data['risk']}\nState: {data['state']}"
        )

    net.toggle_physics(False)

    st.components.v1.html(net.generate_html(), height=650)


# Streamlit App
def main():

    if "graph" not in st.session_state:
        st.session_state.graph = None

    if "graph_size" not in st.session_state:
        st.session_state.graph_size = None

    st.set_page_config(
        page_title="Evacuation Route Optimizer",
        page_icon="🗺️",
        layout="wide"
    )

    st.markdown(
        "<h1 style='text-align: center;'>Evacuation Route Optimizer</h1>",
        unsafe_allow_html=True
    )

    st.subheader("Map Selection")

    map_size = st.selectbox(
        "Choose map size",
        [
            "Demo Mode (15 nodes)",
            "Small (75 nodes)",
            "Medium (150 nodes)",
            "Large (250 nodes)"
        ]
    )

    size_map = {
        "Demo Mode (15 nodes)": 15,
        "Small (75 nodes)": 75,
        "Medium (150 nodes)": 150,
        "Large (250 nodes)": 250
    }

    if (
        st.session_state.graph is None
        or st.session_state.graph_size != map_size
    ):
        G = create_graph(size_map[map_size])

        st.session_state.graph = G
        st.session_state.graph_size = map_size

    if st.session_state.graph is None:
        st.stop()

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
    with st.expander("Road Damage Control", expanded=False):

        st.caption(
            "Use the buttons below to set each road condition. "
            "Blocked roads are removed from possible routes."
        )

        damage_options = ["intact", "light", "heavy", "blocked"]

        display_names = {
            "intact": "Intact",
            "light": "Light",
            "heavy": "Heavy",
            "blocked": "Blocked"
        }

        st.markdown(
            """
            <div style='font-size: 14px; margin-bottom: 10px;'>
                <b>Road condition key:</b>
                Intact = normal road,
                Light = moderately damaged,
                Heavy = severely damaged,
                Blocked = unusable
            </div>
            """,
            unsafe_allow_html=True
        )

        edges = list(G.edges())
        road_col1, road_col2 = st.columns(2)

        for index, (u, v) in enumerate(edges):

            target_col = road_col1 if index % 2 == 0 else road_col2

            with target_col:
                current_state = G[u][v].get("state", "intact")
                current_index = damage_options.index(current_state)

                state = st.radio(
                    f"{u}-{v}",
                    damage_options,
                    index=current_index,
                    format_func=lambda option: display_names[option],
                    horizontal=True,
                    key=f"road_state_{st.session_state.graph_size}_{u}_{v}"
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
                draw_interactive_graph(G, start=start)

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

                draw_interactive_graph(G, path, start=start)

        except Exception:
            st.error("No feasible evacuation route exists under the current road settings.")
            draw_interactive_graph(G, start=start)

    else:
        draw_interactive_graph(G, start=start)


if __name__ == "__main__":
    main()
