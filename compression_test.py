import sys
from pprint import pprint

import networkx as nx
from black.trans import defaultdict
import matplotlib.pyplot as plt
from networkx.classes import DiGraph


def draw_graph(graph: DiGraph):
    nx.draw(graph, with_labels=True)
    plt.title("Graph")
    plt.show()


if __name__ == "__main__":
    graph = nx.DiGraph()
    edges = [('a', 'AND-1'), ('AND-1', 'c'), ('c', 'AND-2'), ('AND-2', 'd'), ('d', 'f'), ('f', 'g'), ('AND-2', 'H'), ('AND-1', 'i'), ('i', 'j')]
    # simple_edges = [('z', 'c'), ('c', 'AND-2'), ('AND-2', 'd'), ('d', 'f'), ('f', 'g'), ('AND-2', 'H')]
    graph.add_edges_from(edges)

    node_containers = defaultdict(list)
    pivot_subjects = dict()
    pivot_nodes = {'AND-1', 'AND-2'}

    compressed_graph = nx.DiGraph()

    combinator_parents = dict()
    current_pivot = None
    current_parent = None
    current_node = None
    starting = True
    for subj, obj in nx.dfs_edges(graph):
        if starting:
            starting_node = subj
            node_containers[starting_node].append(subj)
            current_node = starting_node
            starting = False
        if obj in pivot_nodes or (subj in pivot_nodes and current_pivot != subj):
            if subj in pivot_nodes:
                parent_node = combinator_parents[subj]
            else:
                parent_node = subj
                combinator_parents[obj] = parent_node
            current_pivot = obj
            pivot_subjects[obj] = parent_node
            current_parent = parent_node
            current_node = None
        if obj not in pivot_nodes and subj in pivot_nodes:
            current_node = obj
            node_containers[current_node].append(obj)
            compressed_graph.add_edge(current_parent, current_node)
        if obj not in pivot_nodes and subj not in pivot_nodes:
            node_containers[current_node].append(obj)
        print(f"({subj}, {obj})", current_pivot, current_parent, current_node)

    draw_graph(compressed_graph)
    pprint(node_containers)
