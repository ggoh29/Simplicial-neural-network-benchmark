import torch
import networkx as nx
from models.SCData import SCData
from utils import edge_to_node_matrix, triangle_to_edge_matrix
import itertools
import numpy as np
from scipy.spatial import Delaunay
import random
from joblib import Parallel, delayed
from tqdm import tqdm

######################################################################################################
# This section is adopted from https://github.com/twitter-research/cwn/blob/main/data/datasets/flow_utils.py
######################################################################################################


def is_inside_rectangle(x, rect):
    return rect[0, 0] <= x[0] <= rect[1, 0] and rect[0, 1] <= x[1] <= rect[1, 1]


def sample_point_from_rect(points, rect):
    samples = []
    for i in range(len(points)):
        if is_inside_rectangle(points[i], rect):
            samples.append(i)

    return random.choice(samples)


def create_hole(points, triangles, hole):
    kept_triangles = []
    removed_vertices = set()

    # Find the points and triangles to remove
    for i in range(len(triangles)):
        simplex = triangles[i]
        assert len(simplex) == 3
        xs = points[simplex]

        remove_triangle = False
        for j in range(3):
            vertex = simplex[j]
            if is_inside_rectangle(xs[j], hole):
                remove_triangle = True
                removed_vertices.add(vertex)

        if not remove_triangle:
            kept_triangles.append(i)

    # Remove the triangles and points inside the holes
    triangles = triangles[np.array(kept_triangles)]

    # Remove the points that are not part of any triangles anymore.
    # This can happen in some very rare cases
    for i in range(len(points)):
        if np.sum(triangles == i) == 0:
            removed_vertices.add(i)

    points = np.delete(points, list(removed_vertices), axis=0)

    # Renumber the indices of the triangles' vertices
    for vertex in sorted(removed_vertices, reverse=True):
        triangles[triangles >= vertex] -= 1

    return points, triangles


def create_graph_from_triangulation(points, triangles):
    # Create a graph from from this containing only the non-removed triangles
    G = nx.Graph()
    edge_idx = 0
    edge_to_tuple = {}
    tuple_to_edge = {}

    for i in range(len(triangles)):
        vertices = triangles[i]
        for j in range(3):
            if vertices[j] not in G:
                G.add_node(vertices[j], point=points[vertices[j]])

            for v1, v2 in itertools.combinations(vertices, 2):
                if not G.has_edge(v1, v2):
                    G.add_edge(v1, v2, index=edge_idx)
                    edge_to_tuple[edge_idx] = (min(v1, v2), max(v1, v2))
                    tuple_to_edge[(min(v1, v2), max(v1, v2))] = edge_idx
                    edge_idx += 1
                assert G.has_edge(v2, v1)

    G.graph['edge_to_tuple'] = edge_to_tuple
    G.graph['tuple_to_edge'] = tuple_to_edge
    G.graph['points'] = points
    G.graph['triangles'] = triangles
    return G


def get_orient_matrix(size, orientation):
    """Creates a change of orientation operator of the specified size."""
    if orientation == 'default':
        return np.identity(size)
    elif orientation == 'random':
        diag = 2 * np.random.randint(0, 2, size=size) - 1
        return np.diag(diag).astype(np.long)
    else:
        raise ValueError(f'Unsupported orientation {orientation}')


def generate_trajectory(start_rect, end_rect, ckpt_rect, G: nx.Graph):
    points = G.graph['points']
    tuple_to_edge = G.graph['tuple_to_edge']

    start_vertex = sample_point_from_rect(points, start_rect)
    end_vertex = sample_point_from_rect(points, end_rect)
    ckpt_vertex = sample_point_from_rect(points, ckpt_rect)

    x = np.zeros((len(tuple_to_edge), 1))

    vertex = start_vertex
    end_point = points[end_vertex]
    ckpt_point = points[ckpt_vertex]

    path = [vertex]
    explored = set()

    ckpt_reached = False

    while vertex != end_vertex:
        explored.add(vertex)
        if vertex == ckpt_vertex:
            ckpt_reached = True

        nv = np.array([nghb for nghb in G.neighbors(vertex)
                       if nghb not in explored])
        if len(nv) == 0:
            # If we get stuck because everything around was explored
            # Then just try to generate another trajectory.
            return generate_trajectory(start_rect, end_rect, ckpt_rect, G)
        npoints = points[nv]

        if ckpt_reached:
            dist = np.sum((npoints - end_point[None, :]) ** 2, axis=-1)
        else:
            dist = np.sum((npoints - ckpt_point[None, :]) ** 2, axis=-1)

        # prob = softmax(-dist**2)
        # vertex = nv[np.random.choice(len(prob), p=prob)]
        coin_toss = np.random.uniform()

        if coin_toss < 0.1:
            vertex = nv[np.random.choice(len(dist))]
        else:
            vertex = nv[np.argmin(dist)]

        path.append(vertex)

        # Set the flow value according to the orientation
        if path[-2] < path[-1]:
            x[tuple_to_edge[(path[-2], path[-1])], 0] = 1
        else:
            x[tuple_to_edge[(path[-1], path[-2])], 0] = -1
    return x, path


def generate_flow_cochain(class_id, G, B1, B2, T2):
    assert 0 <= class_id <= 1

    # Define the start, midpoint and and stop regions for the trajectories.
    start_rect = np.array([[0.0, 0.8], [0.2, 1.0]])
    end_rect = np.array([[0.8, 0.0], [1.0, 0.2]])
    bot_ckpt_rect = np.array([[0.0, 0.0], [0.2, 0.2]])
    top_ckpt_rect = np.array([[0.8, 0.8], [1.0, 1.0]])
    ckpts = [bot_ckpt_rect, top_ckpt_rect]

    # Generate flow
    X1, _ = generate_trajectory(start_rect, end_rect, ckpts[class_id], G)

    X1 = torch.tensor(X1, dtype=torch.float)
    T2 = torch.tensor(T2, dtype=torch.float)

    B1 = (B1 @ T2).to_sparse()
    B2 = (T2 @ B2).to_sparse()
    return SCData(torch.zeros((B1.shape[0], 1)), X1, torch.zeros((B2.shape[1], 1)), B1, B2, torch.tensor([class_id]))


def gen_orientation_graph(num_points=1000, num_train=1000, num_test=200,
                          train_orientation='default', test_orientation='default', n_jobs=4):
    points = np.random.uniform(low=-0.05, high=1.05, size=(num_points, 2))
    tri = Delaunay(points)

    # Double check each point appears in some triangle.
    for i in range(len(points)):
        assert np.sum(tri.simplices == i) > 0

    hole1 = np.array([[0.2, 0.2], [0.4, 0.4]])
    hole2 = np.array([[0.6, 0.6], [0.8, 0.8]])

    points, triangles = create_hole(points, tri.simplices, hole1)

    # Double check each point appears in some triangle.
    for i in range(len(points)):
        assert np.sum(triangles == i) > 0

    points, triangles = create_hole(points, triangles, hole2)

    # Double check each point appears in some triangle.
    for i in range(len(points)):
        assert np.sum(triangles == i) > 0

    assert np.min(triangles) == 0
    assert np.max(triangles) == len(points) - 1

    G = create_graph_from_triangulation(points, triangles)
    print(G)
    assert G.number_of_nodes() == len(points)

    nodes, edges = G.nodes(), G.edges()

    B1 = edge_to_node_matrix(edges, nodes)
    B2 = triangle_to_edge_matrix(triangles, edges)

    classes = 2

    assert B1.shape[1] == B2.shape[0]
    num_edges = B1.shape[1]

    # Process these in parallel because it's slow
    samples_per_class = num_train // classes
    train_samples = Parallel(n_jobs=n_jobs, prefer="threads")(delayed(generate_flow_cochain)\
                            (class_id=min(i // samples_per_class, 1), G=G, B1=B1, B2=B2,
                            T2=get_orient_matrix(num_edges, train_orientation)) for i in tqdm(range(num_train)))

    samples_per_class = num_test // classes
    test_samples = Parallel(n_jobs=n_jobs, prefer="threads")(delayed(generate_flow_cochain)\
                            (class_id=min(i // samples_per_class, 1), G=G, B1=B1, B2=B2,
                            T2=get_orient_matrix(num_edges, test_orientation)) for i in tqdm(range(num_test)))

    return train_samples, test_samples


if __name__ == "__main__":
    gen_orientation_graph()
