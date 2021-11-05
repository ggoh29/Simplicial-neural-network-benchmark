from abc import ABC, abstractmethod
import networkx as nx
from skimage.future import graph
from skimage.measure import regionprops


def unpack_node(rag, region):
	return list(rag._node[region['label']]['mean color'])+ list(region['centroid'])


def colour_intensity(feature):
	return feature[0]**2 + feature[1]**2 + feature[2]**2


def make_triangles(triangle, features):
	tri = list(triangle)
	tri = [*sorted(tri, key = lambda x : colour_intensity(features[x-1]), reverse=True)]
	return tri

# Transforms a graph into a simplicial complex
class EdgeFlow(ABC):

	@staticmethod
	@abstractmethod
	def convert_graph(image, superpixel):
		pass


class RAGBasedEdgeFlow(EdgeFlow):

	"Edges flow from based on those computed in graph.rag_mean_color"

	@staticmethod
	def convert_graph(image, superpixel):
		rag = graph.rag_mean_color(image, superpixel)
		regions = regionprops(superpixel)
		node_features = [unpack_node(rag, region) for region in regions]
		triangles = [*filter(lambda x: len(x) == 3, nx.enumerate_all_cliques(rag))]
		return rag.nodes(), rag.edges(), triangles, node_features


class PixelBasedEdgeFlowSC(EdgeFlow):

	"Edges flow from high pixel values to low pixel values"

	@staticmethod
	def convert_graph(image, superpixel):
		rag = graph.rag_mean_color(image, superpixel)
		regions = regionprops(superpixel)
		node_features = [unpack_node(rag, region) for region in regions]
		edges = []
		for x, y in rag.edges:
			node_x, node_y = node_features[x-1], node_features[y-1]
			if colour_intensity(node_y) > colour_intensity(node_x):
				edges.append((y,x))
			else:
				edges.append((x,y))

		def _make_tri(triangles):
			return make_triangles(triangles, node_features)

		triangles = [*filter(lambda x: len(x) == 3, nx.enumerate_all_cliques(rag))]
		triangles = [*map(_make_tri, triangles)]
		return rag.nodes(), edges, triangles, node_features

class RandomBasedEdgeFlowSC(EdgeFlow):

	"Class to test permutation invariance. Edges are randomly permuted"

	@staticmethod
	def convert_graph(rag, regions):
		raise NotImplementedError()
