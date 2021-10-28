from abc import ABC, abstractmethod
import networkx as nx


def unpack_node(rag, region):
	return list(region['centroid']) + list(rag._node[region['label']]['mean color'])


def colour_intensity(feature):
	return feature[2] ** 2 + feature[3] ** 2 + feature[4] ** 2


# Transforms a graph into a simplicial complex
class MakeSC(ABC):

	@staticmethod
	@abstractmethod
	def convert_graph(self, rag, regions):
		pass


class RAGSC(MakeSC):

	@staticmethod
	def convert_graph(rag, regions):
		node_features = [unpack_node(rag, region) for region in regions]
		triangles = [*filter(lambda x: len(x) == 3, nx.enumerate_all_cliques(rag))]
		return rag.nodes(), rag.edges(), triangles, node_features


class EdgeFlowSC(MakeSC):

	@staticmethod
	def convert_graph(rag, regions):
		node_features = [unpack_node(rag, region) for region in regions]
		edges = []
		for x, y in rag.edges:
			node_x, node_y = node_features[x-1], node_features[y-1]
			if colour_intensity(node_y) > colour_intensity(node_x):
				edges.append((y,x))
			else:
				edges.append((x,y))

		triangles = [*filter(lambda x: len(x) == 3, nx.enumerate_all_cliques(rag))]
		return rag.nodes(), edges, triangles, node_features
