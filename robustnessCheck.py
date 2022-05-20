import networkx as nx
import ast
import os
import math


def lcc_length(G):
    return len(max(nx.strongly_connected_components(G), key=len))


def lcc(G):
    return max(nx.strongly_connected_components(G), key=len)


def shortest_average_path_length(G):
    return nx.average_shortest_path_length(G)


def average_dijkstra_path(G, w):
    d = dict(nx.all_pairs_dijkstra_path_length(G, weight=w))
    lens = []
    for i in d:
        if len(d[i]) > 2:
            avg_length = sum(d[i].values()) / (len(d[i]) - 1)
            lens.append(avg_length)
    return sum(lens)/len(lens)


def shortest_average_dijkstra_path(G, w):
    d = dict(nx.all_pairs_dijkstra_path_length(G, weight=w))
    lens = []
    ids = []
    paths = []
    for i in d:
        if len(d[i]) > 2:
            avg_length = sum(d[i].values()) / (len(d[i]) - 1)
            lens.append(avg_length)
            ids.append(i)
            paths.append(d[i].values())

    shortest = min(lens)
    return G[ids[lens.index(shortest)]], paths[lens.index(shortest)], shortest


def longest_average_dijkstra_path(G, w):
    d = dict(nx.all_pairs_dijkstra_path_length(G, weight=w))
    lens = []
    ids = []
    paths = []
    for i in d:
        if len(d[i]) > 2:
            avg_length = sum(d[i].values()) / (len(d[i]) - 1)
            lens.append(avg_length)
            ids.append(i)
            paths.append(d[i].values())

    longest = max(lens)
    return G[ids[lens.index(longest)]], paths[lens.index(longest)], longest


def shortest_path_dijkstra(G, s, d, w):
    return nx.dijkstra_path_length(G, s, d, weight=w)


def global_efficiency(G):
    d = dict(nx.all_pairs_shortest_path_length(G))
    s = 0
    for i in d:
        d[i].update((x, 0 if y == 0 else 1/y) for x, y in d[i].items())
        s = s + sum(d[i].values())
    return (1 / (G.number_of_nodes() * (G.number_of_nodes() - 1))) * s
    

def global_efficiency_dijkstra(G):
    d = dict(nx.all_pairs_dijkstra_path_length(G, weight='length'))
    s = 0
    for i in d:
        d[i].update((x, 0 if y == 0 else 1/y) for x, y in d[i].items())
        s = s + sum(d[i].values())
    return (1 / (G.number_of_nodes() * (G.number_of_nodes() - 1))) * s


def get_nearest_node(g, x, y):
    original = [x, y]
    closest_id = ''
    dist = 1000
    for x, data in g.nodes.data():
        p = [float(data["x"]), float(data["y"])]
        if math.dist(p, original) < dist:
            closest_id = x
            dist = math.dist(p, original)
    return g[closest_id], closest_id


def load_networkx (filepath : str) -> nx.DiGraph :
    g = nx.read_graphml(filepath)

    dtypes = {
        "bearing": float,
        "grade": float,
        "grade_abs": float,
        "length": float,
        "osmid": int,
        "speed_kph": float,
        "travel_time": float,
    }

    for _, _, data in g.edges(data = True):
        data.pop("id", None)

        for attr, value in data.items():
            if value.startswith("[") and value.endswith("]"):
                try:
                    data[attr] = ast.literal_eval(value)
                except (SyntaxError, ValueError):
                    pass

        for attr in data.keys() & dtypes.keys():
            if isinstance(data[attr], list):
                data[attr] = [dtypes[attr](item) for item in data[attr]]
            else:
                data[attr] = dtypes[attr](data[attr])

    return nx.DiGraph(g)


g = load_networkx(os.path.join('osm-network', 'slovenia.osm'))
print(g)

print(get_nearest_node(g, 13.592013, 46.269776))
print(get_nearest_node(g, 14.486538, 46.050842))
print(shortest_path_dijkstra(g, '277887900', '867596497', 'travel_time')) # koper do maribor
print(shortest_path_dijkstra(g, '4723856618', '26458504', 'travel_time')) # magozd do ljubljana