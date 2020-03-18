import base64
import random
import logging
from igraph import *
import numpy as np
import time


class Detector(object):
    instance = None
    graphs = dict()
    log = logging.getLogger(__name__)
    def __new__(cls):
        if cls.instance is not None:
            return cls.instance
        else:
            inst = cls.instance = super(Detector, cls).__new__(cls)
            return inst

    def registerGraph(self, file, nameG):

        random.seed(123)
        plt = Plot()
        name = "database/" + nameG
        #read file
        print("registering graph: "+nameG)
        text_file = open(name, "w+")
        text_file.write(file)
        text_file.close()
        g = Graph.Read_Ncol(name, directed=False)
        os.remove(name)


        g = g.simplify(g)
        g.vs["vertex_size"] = 1
        # g.layout_drl()
        # plot(g,layout=g.layout_drl())
        # subgraph_vertex_list = [v.index for v in random.sample(list(g.vs), 40000)]
        ss =self._determine_best_sampling_size(len(g.vs))
        newG,key =self._strong_simplify(g,factor=ss)

        plt.add(newG,layout='auto')
        plt.save(name + "-graph.png")

        with open(name + "-graph.png", 'rb') as imageFile:
            graph = base64.b64encode(imageFile.read()).decode()
        os.remove(name + "-graph.png")

        response = {
            'graph': graph,
            'edges' : len(g.es),
            'vertices': len(g.vs)
        }
        ##add to dict
        self.graphs[nameG] = GraphInfo(g,True,key)

        print("registering graph complete: "+nameG)

        return response

    def unregister(self, name):
        self.graphs[name] = None

    def _edge_betweenness(self, file):

        # Create the graph
        # vertices = [i for i in range(7)]
        # edges = [(0, 2), (0, 1), (0, 3), (1, 0), (1, 2), (1, 3), (2, 0), (2, 1), (2, 3), (3, 0), (3, 1), (3, 2), (2, 4),
        #          (4, 5), (4, 6), (5, 4), (5, 6), (6, 4), (6, 5)]
        name = "database/random-" + str(random())

        plt = Plot()

        os.mknod(name)
        text_file = open(name, "w")
        n = text_file.write(file)
        text_file.close()

        g = Graph.Read_Ncol(name, directed=False)
        plt.add(g)
        plt.save(name + "jpg")

        # gg = g.induced_subgraph([0,1,2], implementation="auto")

        # save plot to file
        # g.write_svg(name+".svg")
        # out.save(name+".jpg")

        visual_style = {}
        g = g.simplify(g)
        g.layout_auto()
        plot(g,layout = "layout_drl")
        # Scale vertices based on degree
        outdegree = g.outdegree()
        visual_style["vertex_size"] = [x / max(outdegree) * 25 + 50 for x in outdegree]

        # Set bbox and margin
        visual_style["bbox"] = (800, 800)
        visual_style["margin"] = 100

        # Define colors used for outdegree visualization
        colours = ['#fecc5c', '#a31a1c']

        # Order vertices in bins based on outdegree
        bins = np.linspace(0, max(outdegree), len(colours))
        digitized_degrees = np.digitize(outdegree, bins)

        # Set colors according to bins
        g.vs["color"] = [colours[x - 1] for x in digitized_degrees]

        # Also color the edges
        for ind, color in enumerate(g.vs["color"]):
            edges = g.es.select(_source=ind)
            edges["color"] = [color]

        # Don't curve the edges
        visual_style["edge_curved"] = False

        # Community detection
        communities = g.community_edge_betweenness()
        done = time.time()

        clusters = communities.as_clustering()
        clusters.write_svg(name + ".svg")

        # plot(clusters)

        # Set edge weights based on communities
        print(clusters)
        weights = {v: len(c) for c in clusters for v in c}
        g.es["weight"] = [weights[e.tuple[0]] + weights[e.tuple[1]] for e in g.es]

        # Choose the layout
        # N = len(vertices)
        # visual_style["layout"] = g.layout_fruchterman_reingold(weights=g.es["weight"], maxiter=1000, area=N ** 3,
        #                                                        repulserad=N ** 3)
        array = [None] * len(clusters)
        for idx, cluster in enumerate(clusters):
            for memeber in cluster:
                if array[idx] == None:
                    array[idx] = [memeber]
                else:
                    array[idx].append(memeber)
                print(str(memeber))

        # Plot the graph
        plot(g, **visual_style)
        return array

    def edge_betweenness(self, nameG):
        # process graph
        graph_info = self.graphs[nameG]

        # community
        print("performing EDGE_BETWENNESS graph: "+nameG)

        start = time.time()
        communities = graph_info.graph.community_edge_betweenness()
        done = time.time()

        (cluster, clusters) = self._saveCluster(name + "clusters.png", communities, graph_info)

        response = {
            'cluster': cluster,
            'modularity': clusters.q,
            'time': (done - start)
        }
        return response

    def fast_greedy(self, nameG):
        # process graph
        graph_info = self.graphs[nameG]
        # community
        print("performing FAST GREEDY graph: "+nameG)

        start = time.time()
        communities = graph_info.graph.community_fastgreedy()
        done = time.time()

        (cluster, clusters) = self._saveCluster(name + "clusters.png", communities, graph_info)

        response = {
            'cluster': cluster,
            'modularity': clusters.q,
            'time': (done - start)
        }
        return response

    def walk_trap(self, nameG, steps=4):
        # process graph
        graph_info = self.graphs[nameG]
        # community
        print("performing WALK TRAP community detection: "+nameG)

        start = time.time()
        communities = graph_info.graph.community_walktrap(steps=steps)
        done = time.time()

        (cluster, clusters) = self._saveCluster(name + "clusters.png", communities, graph_info)

        response = {
            'cluster': cluster,
            'modularity': clusters.q,
            'time': (done - start)
        }
        return response

    def _saveCluster(self, name, communities, graph_info):
        clusters = communities.as_clustering()
        print("saving cluster :"+name)
        if (graph_info.isHuge):
            newGraph = clusters._graph.subgraph(graph_info.random_list)
            clusters._graph.layout_auto()

        newm = []
        for i in newGraph.vs.indices:
            a = clusters.membership[graph_info.random_list[i]]
            newm.append(a)

        newClustering = VertexClustering(newGraph,membership=newm)
        random.randint(0, 9)
        random.seed(123)
        plt = Plot()
        plt.add(newClustering,vertex_size=8,layout='auto')
        plt.save(name + "clusters.png")

        with open(name + "clusters.png", 'rb') as imageFile:
            cluster = base64.b64encode(imageFile.read()).decode()
        os.remove(name + "clusters.png")
        return (cluster, clusters)

    def _strong_simplify(self,graph,factor=1000):
        vs = VertexSeq(graph)
        degree_list = vs.degree()

        degree_list = {i: degree_list[i] for i in range(0, len(degree_list))}
        degree_list = {k: v for k, v in sorted(degree_list.items(), reverse=True, key=lambda item: item[1])}
        key = list(degree_list.keys())[0:factor]
        key.sort()
        # newG = graph.subgraph(key)
        # nodes = [i for i, e in enumerate(newG.vs.degree()) if e != 0]
        # key = list(set(nodes) & set(key))
        newG = graph.subgraph(key)
        newG.layout_auto()
        return newG,key

    def _determine_best_sampling_size(self,sampleSize):
        if(sampleSize <1000):
            return sampleSize
        if(sampleSize<5000):
            return sampleSize
        if (sampleSize < 10000):
            return sampleSize/10
        else:
            return 900

class GraphInfo:
    def __init__(self, graph, isHuge, random_list=None):
        self.graph = graph
        self.isHuge = isHuge
        self.random_list = random_list


detector = Detector()


def createDetectorInstanse():
    detector = Detector()
    return detector
