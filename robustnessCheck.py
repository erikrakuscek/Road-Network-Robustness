import networkx as nx
import matplotlib.pyplot as plt
import powerlaw
import math
import numpy as np
import random
from statistics import mean


def degreeDistribution(G, color):
    degree_hist = nx.degree_histogram(G)
    degree_prob = list(map(lambda x: x / G.number_of_nodes(), degree_hist))
    plt.loglog(degree_prob, '.', markersize=5, color=color)
    fit = powerlaw.Fit(list(map(lambda x: G.degree(x), G.nodes())), xmin=1, verbose=False, estimate_discrete=True)
    print(fit.alpha)
    print("k-max", np.max(degree_hist))
    print("k-min * n^(1/gama-1)", math.pow(G.number_of_nodes(), 1 / (fit.alpha - 1)))
    plt.xlabel("log k")
    plt.ylabel("log P(k)")
    plt.show()


def lccLength(G):
    return len(max(nx.connected_components(G), key=len))


def lcc(G):
    return max(nx.connected_components(G), key=len)


def shortestAveragePathLength(G):
    for component in nx.connected_components(G):
        component_ = G.subgraph(component)
        nodes = component_.nodes()
        lengths = []
        for _ in range(10):
            n1, n2 = random.choices(list(nodes), k=2)
            length = nx.shortest_path_length(component_, source=n1, target=n2)
            lengths.append(length)
        print(f'Nodes num: {len(nodes)}, shortest path mean: {mean(lengths)} \n')
    return nx.shortest_path_length(G)


# G = nx.read_adjlist('road-luxembourg-osm.mtx')