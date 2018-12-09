import snap
import numpy as np
import util
import pickle
import os

def get_neighbors(graph, nodeId):
    n = set([])
    for i in range(graph.GetNI(nodeId).GetDeg()):
        nid = graph.GetNI(nodeId).GetNbrNId(i)
        n.add(nid)
    return n
 
def sim(x, y):
    if np.linalg.norm(x) == 0 or np.linalg.norm(y) == 0:
        return 0
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))


def collect_features(graph):
    print '\nCollecting features'
    features = {}
    for node in graph.Nodes():
        deg = node.GetDeg()
        in_edges = set([])
        out_edges = set([])
        neighbors = get_neighbors(graph, node.GetId())
        for i in neighbors:
            for j in get_neighbors(graph, i):
                if j in neighbors and (i, j) not in in_edges and (j, i) not in in_edges:
                    in_edges.add((i, j))
                if j != node.GetId() and j not in neighbors and (i, j) not in out_edges and (j, i) not in out_edges:
                    if node.GetId() == 9:
                        print j
                    out_edges.add((i, j))
        features[node.GetId()] = np.array([deg, len(neighbors) + len(in_edges), len(out_edges)])

    return features

def collect_recursive_features(graph, features, k=2):
    for i in range(k):
        n_starting_features = len(features[graph.GetRndNId()])
        for node in graph.Nodes():
            s = np.zeros(n_starting_features)
            nbrs = get_neighbors(graph, node.GetId())
            for n in nbrs:
                s += features[n][:n_starting_features]
            new_features = np.zeros(3 * len(s))
            new_features[:len(s)] = features[node.GetId()]
            if len(nbrs) > 0:
                new_features[len(s):2*len(s)] = s
                new_features[2*len(s):] = s / len(nbrs)
            features[node.GetId()] = new_features
            
    return features
    

# be in top directory, not in src
assert(os.path.isdir('data/alliance_features'))
print(os.getcwd())

for year in range(1816, 2012 + 1):
    graph = util.load_alliance_graph(year)
    features = collect_features(graph)
    features = collect_recursive_features(graph, features, k=1)
    for key in features:
        features[key] = [features[key][i] for i in range(len(features[key]))]
    with open('data/alliance_features/features{}.pickle'.format(year), 'wb') as of:
        pickle.dump(features, of)
   
