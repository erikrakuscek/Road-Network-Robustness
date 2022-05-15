from collections import deque, Counter
from itertools import combinations
import networkx as nx
import matplotlib.pyplot as plt
import powerlaw
import math
import numpy as np
import random
from statistics import mean
from tqdm import tqdm
import time

def read_net(file_name):
    """Read network"""
    G = nx.Graph(name=file_name)  # define empty graph
    with open(file_name, 'r') as f:
        for line in f:
            if not line.startswith("%"):
                break

        num_of_nodes = int(line.split()[0])

        # add nodes
        for i in range(num_of_nodes):
            G.add_node(i, label=i+1)

        # add edges
        for line in f:
            node1_str, node2_str = line.split()[:2]
            G.add_edge(int(node1_str) - 1, int(node2_str) - 1)
    return G

def get_link_triads(G, i, j):
    """Return link triads containing nodes i and j"""
    return len(set(G[i]).intersection(G[j]))

def get_node_triads(G, i):
    """Return triads containing node i"""
    t = 0
    for neighbor in G[i]:
        if G.degree[i] <= G.degree[neighbor]:
            # it's just to optimize the set intersection,
            # if the first set is smaller, we use less resources
            t += get_link_triads(G, i, neighbor) / 2
        else:
            t += get_link_triads(G, neighbor, i) / 2
    return t


def get_node_clustering_coef(G, i):
    """Return node clustering coef for node i"""
    k_i = G.degree[i]
    if k_i <= 1:
        return 0
    return get_node_triads(G, i) * 2 / (k_i ** 2 - k_i)


def sort_node_centrality_list(node_centrality):
    return sorted([(i, c) for i, c in node_centrality], key=lambda x: x[1], reverse=True)


def calc_degree_centrality(G):
    """
    Calcaulates degree centrality for each node

    Return
    ------
    sorted list of pairs (node, degree centrality value) in descending order
    """
    node_centrality = [(i, G.degree(i) / (G.number_of_nodes()-1)) for i in tqdm(G.nodes())]
    return sort_node_centrality_list(node_centrality)


def calc_cluster_centrality(G):
    """
    Calcaulates cluster coeficient for each node

    Return
    ------
    sorted list of pairs (node, cluster coeficient) in descending order
    """
    node_centrality = [(i, get_node_clustering_coef(G,i)) for i in tqdm(G.nodes())]
    return sort_node_centrality_list(node_centrality)

def calc_mu_cluster_centrality(G):
    """
    Calcaulates µ-corrected cluster coeficient for each node

    Return
    ------
    sorted list of pairs (node,  µ-corrected coeficient) in descending order
    """
    mu = max([get_link_triads(G,i,j) for i,j in G.edges()])
    node_centrality = [(i, get_node_clustering_coef(G,i) * (G.degree(i) - 1) / mu) for i in tqdm(G.nodes())]
    return sort_node_centrality_list(node_centrality)

def get_distance_for_node(G, i):
    """Return list of distances between node i and others"""
    # empty array
    D = [-1] * len(G)
    Q = deque()
    D[i] = 0
    Q.append(i)

    # main algorithm
    while Q:
        i = Q.popleft()
        for j in G[i]:  # neighbors
            if D[j] == -1:
                D[j] = D[i] + 1
                Q.append(j)
    return [d for d in D if d > 0]

def calc_closeness_centrality(G):
    """
    Calcaulates closeness centrality for each node

    Return
    ------
    sorted list of pairs (node,  closeness centrality) in descending order
    """
    node_centrality = [(i, sum([1/d for d in get_distance_for_node(G,i)])/ (G.number_of_nodes()-1)) for i in tqdm(G.nodes())]
    return sort_node_centrality_list(node_centrality)

def calc_closeness_centrality_nx(G):
    """
    Calcaulates closeness centrality for each node

    Return
    ------
    sorted list of pairs (node,  closeness centrality) in descending order
    """
    node_to_centrality = nx.closeness_centrality(G)
    node_centrality = node_to_centrality.items()
    return sort_node_centrality_list(node_centrality)


def calc_betweenness_centrality(G):
    """
    Calcaulates betweenness centrality for each node

    Return
    ------
    sorted list of pairs (node,  betweenness centrality) in descending order
    """
    node_to_centrality = nx.betweenness_centrality(G)
    node_centrality= node_to_centrality.items()
    return sort_node_centrality_list(node_centrality)

def calc_eigenvector_centrality(G, eps=1e-6):
    #define list of default centrality values for each node (set value = 1）
    E = [1] * G.number_of_nodes()

    #define default difference between old and new values (set value > eps)
    diff = 1

    while diff > eps:

        # U = sum of neighboring centrality values for each node
        U = [sum(E[j] for j in G[i]) for i in G.nodes()]

        # calculate normalizing constant u_sum
        u_sum = sum(U)

        #update all nodes' values in U according to the algo
        U = [U[i] * G.number_of_nodes()/u_sum for i in G.nodes()]

        #update diff = sum of old value - new value over each node
        diff = sum([abs(E[i] - U[i]) for i in G.nodes])
        print(diff)
        # update E
        E = U

    node_centrality = [(i, E[i]) for i in range(len(E))]
    return sort_node_centrality_list(node_centrality)


def sort_edge_centrality_list(edge_centrality):
    """Sorts list of pairs by the second element in pairs """
    return sorted([(i, c) for i, c in edge_centrality],
                    key=lambda x: x[1], reverse=True)


def get_edges_statistics_in_paths(G):
    """Returns dict {
        (source, target): ({edge_ij: g_{st}^{ij}}, g_{st} - number_of_paths from source to target)
    }
    """
    edges_to_paths_info = dict()

    # iterate over all pairs of nodes
    for source, target in tqdm(list(combinations(G.nodes(), 2))):
        try:
            # find all shortest paths from source to target (use nx.all_shortest_paths)
            shortest_paths = list(nx.all_shortest_paths(G, source, target))

            # number of paths
            number_of_paths = len(shortest_paths)

            # get all edges in the paths
            edges = []
            for path in shortest_paths:
                edges.extend([tuple(sorted((path[i], path[i + 1]))) for i in range(len(path) - 1)])

            # calculate number of edge occurences
            edge_to_cnt = dict(Counter(edges))
            edges_to_paths_info[(source, target)] = (edge_to_cnt, number_of_paths)
        except nx.NetworkXNoPath:
            pass
    return edges_to_paths_info


def calc_simple_edge_betweenness_centrality(G, normalized=True):
    """Calculates edge_betweenness_centrality in a simple way"""
    nodes_to_paths_info = get_edges_statistics_in_paths(G)

    # calculate sigma_ij
    edge_to_centrality = dict()

    for edge_to_cnt, number_of_paths in nodes_to_paths_info.values():
        for edge, edge_cnt in edge_to_cnt.items():
            if edge in edge_to_centrality:
                edge_to_centrality[edge] += edge_cnt / number_of_paths
            else:
                edge_to_centrality[edge] = edge_cnt / number_of_paths

    # normalization (as it is done in networkx)
    if normalized:
        normalized_factor = 2 / (len(G.nodes()) - 1) / (len(G.nodes()) - 2)
        for edge in edge_to_centrality:
            edge_to_centrality[edge] *= normalized_factor

    return edge_to_centrality


def calc_edge_betweenness_centrality(G, normalized=True, method='networkx'):
    """Calculates edge_betweenness_centrality via given method"""
    if method == 'networkx':
        edge_to_centrality = nx.edge_betweenness_centrality(G, normalized=normalized)
    elif method == 'simple':
        edge_to_centrality = calc_simple_edge_betweenness_centrality(G, normalized)
    else:
        raise NotImplementedError(f'Method {method} is not implemented')

    # return transformed dict to sorted list of pairs
    edge_centrality_list = list(edge_to_centrality.items())
    return sort_edge_centrality_list(edge_centrality_list)

def calc_edge_embeddedness_centrality(G):

    #calculate theta
    edge_to_centrality = dict()

    for edge in G.edges():
        i = edge[0]
        j = edge[1]
        if edge not in edge_to_centrality:
            size_of_set = len(set(G[i]).intersection(G[j]))
            if G.degree(i) == 1 and G.degree(j) == 1:
                edge_to_centrality[edge] = 0
            else:
                edge_to_centrality[edge] = size_of_set / (G.degree(i) - 1 + G.degree(j) - 1 - size_of_set)

    # return transformed dict to sorted list of pairs
    edge_centrality_list = list(edge_to_centrality.items())
    return sort_edge_centrality_list(edge_centrality_list)


# luxembourg_test = nx.read_adjlist('luxembourg.txt')
# print(luxembourg_test.number_of_nodes())

luxembourg_G = read_net("road-luxembourg-osm.mtx")

# degree_node_centrality = calc_degree_centrality(luxembourg_G)
# print(degree_node_centrality[:10])
#
# cluster_node_centrality = calc_cluster_centrality(luxembourg_G)
# print(cluster_node_centrality[:500])
#
# mu_cluster_centrality = calc_mu_cluster_centrality(luxembourg_G)
# print(mu_cluster_centrality[:500])

# closeness_node_centrality = calc_closeness_centrality(luxembourg_G)
# print(closeness_node_centrality[:10])

# betweenness_node_centrality = calc_betweenness_centrality(luxembourg_G)
# print(betweenness_node_centrality[:10])

# eigenvector_node_centrality = calc_eigenvector_centrality(luxembourg_G)
# print(eigenvector_node_centrality[:10])

# betweenness_edge_centrality = calc_edge_betweenness_centrality(luxembourg_G,method='simple')
# print(betweenness_edge_centrality[:10])

# embeddedness_edge_centrality = calc_edge_embeddedness_centrality(luxembourg_G)
# print(embeddedness_edge_centrality[:10])