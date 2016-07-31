from __future__ import print_function

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from networkx.algorithms import enumerate_all_cliques, is_isomorphic
from itertools import combinations, chain
from collections import defaultdict

def powerset(items):
    """ The itertools powerset recipe """
    s = list(items)
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s)+1))

def multi_compose(graphs):
    """ Recursively compose (union) a list of graphs together """
    if len(graphs) == 0:
        return nx.Graph() # Base case.
    else:
        return nx.compose(graphs[0], multi_compose(graphs[1:]))

class Container(object):
    pass

def get_labels(comparisons, rejections):
    """
    See https://en.wikipedia.org/wiki/Intersection_number(graph_theory)
    This algorithms finds the clique edge cover, or the smallest number
    of cliques (complete subgraphs) that covers all of a graph. In
    this case, the graph is made such that the vertices are 
    collections of measurements and edges are non-rejected null hypotheses.
    """

    graph = nx.Graph()
    for (x1, x2), reject in zip(comparisons, rejections):
        if not reject:
            graph.add_edge(x1, x2)

    all_cliques = list(enumerate_all_cliques(graph))
    clique_combinations = powerset(all_cliques) 

    click_edge_cover_graphs = None 
    min_cliques = np.inf
    for combination in clique_combinations:
        clique_list = map(graph.subgraph, combination)
        combined = multi_compose(clique_list) 
        if is_isomorphic(combined, graph) and len(combination) < min_cliques:
            min_cliques = len(combination)
            click_edge_cover_graphs = clique_list
            #break

    nodes = graph.nodes()
    node_significance_ids = defaultdict(list)
    significance_id = 0
    for g in click_edge_cover_graphs:
        for node in nodes:
            if node in g.nodes():
                node_significance_ids[node].append(significance_id) 
        significance_id += 1

    return node_significance_ids

if __name__ == '__main__':
    ab = (frozenset(['A', 'B']), False)
    ac = (frozenset(['A', 'C']), True)
    ad = (frozenset(['A', 'D']), True)
    bc = (frozenset(['B', 'C']), False)
    bd = (frozenset(['B', 'D']), False)
    cd = (frozenset(['C', 'D']), False)

    result = get_labels(*zip(*[ab, ac, ad, bc, bd, cd]))

    print(result)

    print('Done')

