from .tangram import *
from .utils import *
from .MCTS_Tangram import *
from .create_tree_layout import *

import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon, LineString
from shapely.ops import cascaded_union
from shapely import affinity
import networkx as nx
import imageio

def getDisplacedPoints(points, x, y, angle):

    # we defined the shapes such that the first coords tuple of the shape; we rotate around corner because then we are guaranteed to not intersect the grid lines when orthogonally rotating (while in case of centroid or center rotations we can end up intersecting and we would need an adjustment to 'click' the shape in place on the grid)
    bottom_left_corner = points[0]
    rotated = affinity.rotate(geom=Polygon(points), angle=-angle, origin=bottom_left_corner)  # shapely.affine.rotate() documentation: "Positive angles are counter-clockwise and negative are clockwise rotations."
    rotated_xy = rotated.exterior.coords.xy
    rotated_xy = list(tuple(zip(rotated_xy[0], rotated_xy[1])))

    new_points = []
    for vertex in rotated_xy:
        new_points.append((vertex[0]+x, vertex[1]+y))

    return new_points

poly_mapping = {"default":

                {'shape_1':{'points':[(0, 0),(1,0),(1,2),(2,2),(2,3),(0,3),(0,0)], 'color':'#D81B60'},
                'shape_2':{'points':[(0, 0),(2,0),(2,2),(1,2),(1,1),(0,1),(0,0)], 'color':'#1E88E5'},
                'shape_3':{'points':[(1,0),(2,0),(2,3),(0,3),(0,1),(1,1),(1,0)], 'color':'#7F582E'},
                'shape_4':{'points':[(0,0),(1,0),(1,1),(3,1),(3,2),(2,2),(2,3),(1,3),(1,2),(0,2),(0,0)], 'color':'#004D40'},
                'shape_5':{'points':[(0,0),(3,0),(3,1),(2,1),(2,2),(1,2),(1,1),(0,1),(0,0)], 'color':'#417C36'},
                'shape_6':{'points':[(1,0),(2,0),(2,2),(1,2),(1,3),(0,3),(0,1),(1,1),(1,0)], 'color':'#1b3316'},
                'shape_7':{'points':[(0,0),(1,0),(1,1),(2,1),(2,0),(3,0),(3,2),(0,2),(0,0)], 'color':'goldenrod'}},

                "WM":
                {'shape_1':{'points':[(0, 0),(2,0),(2,1),(1,1),(1,2),(0,2),(0,0)], 'color':'#D81B60'},
                'shape_2':{'points':[(0, 0),(1,0),(1,3),(0,3),(0,0)], 'color':'#1E88E5'},
                'shape_3':{'points':[(0,0),(3,0),(3,1),(2,1),(2,2),(1,2),(1,1),(0,1),(0,0)], 'color':'#7F582E'},
                'shape_4':{'points':[(0,0),(2,0),(2,1),(1,1),(1,3),(0,3),(0,0)], 'color':'#004D40'}
                }
                }

poly_mapping = poly_mapping['WM']

def pixels_to_shapely_polygon(pixels):
    where = np.nonzero(pixels)
    xs = where[1]
    ys = pixels.shape[0]-1 - where[0]
    pixel_polygons = [Polygon([(x,y), (x+1,y), (x+1, y+1), (x, y+1)]) for (x,y) in zip(xs,ys)]
    return cascaded_union(pixel_polygons)

def plot_state(state, ax=None, plot_inventory=True, save_name=False, show=False, xshift=0, yshift=5, x_nodeoffset=-3, y_nodeoffset=-3, scale=1):

    x_nodeoffset = x_nodeoffset*scale
    y_nodeoffset = y_nodeoffset*scale

    if ax is None:
        f, ax = plt.subplots()

    if plot_inventory:
        # plt.text(1, yshift-1, 'block inventory')
        for shape_name, shape in state.state['available_inventory'].items():
            xs, ys = zip(*poly_mapping[shape_name]['points'])
            xs, ys = np.array(xs), np.array(ys)
            ax.fill(xs*scale+state.world.shape_inventory_xshifts[shape_name], ys*scale, facecolor=poly_mapping[shape_name]['color'])

    # Plot silhouette
    silhouette_polygon = pixels_to_shapely_polygon(state.world.silhouette.pixels)
    xs, ys = zip(*silhouette_polygon.exterior.coords)
    xs, ys = np.array(xs), np.array(ys)
    ax.fill(xs*scale+xshift+x_nodeoffset, ys*scale+yshift+x_nodeoffset, facecolor=state.world.silhouette.color, zorder=10)

    # Plot shapes
    for action_chunk in state.state['action_history']:
        # if primitive
        if len(action_chunk)==1:
            action = action_chunk[0]
            shape_name = action['shape']

            points = getDisplacedPoints(poly_mapping[action['shape']]['points'], action['x'], action['y'], action['angle'])
            xs, ys = zip(*points)
            xs, ys = np.array(xs), np.array(ys)
            ax.fill(xs*scale+xshift+x_nodeoffset, ys*scale+yshift+x_nodeoffset, facecolor=poly_mapping[shape_name]['color'], zorder=10)
        # if action chunk
        else:
            perceptual_chunk = cascaded_union([Polygon(getDisplacedPoints(poly_mapping[action['shape']]['points'], action['x'], action['y'], action['angle'])) for action in action_chunk])
            xs, ys = perceptual_chunk.exterior.coords.xy
            xs, ys = np.array(xs), np.array(ys)
            ax.fill(xs*scale+xshift+x_nodeoffset, ys*scale+yshift+x_nodeoffset, facecolor='k', linewidth=3*scale, zorder=10)

    if save_name:
        plt.savefig(save_name, transparent=True)
        plt.close()
    elif show:
        if state.is_terminal():
            plt.show()
        else:
            plt.show(block=False)
            plt.pause(0.5)
            plt.close()

def plot_tree(tree, save_name=False, title=False, show=False, ax=None, figsize=(18,9), scale_of_state=0.01, baseline_node_size=500):
    G=nx.Graph()
    pairs_of_nodes = [(a,x) for (a,b) in tree.expanded_tree.items() for x in b]
    G.add_edges_from(pairs_of_nodes)
    nx.is_tree(G)
    pos = create_tree_layout(G=G, root=tree.start_node, width=1)

    if ax is None:
        f, ax = plt.subplots(figsize=figsize)

    # Node sizes/colors will be proportional to win ratios
    # node_size = [baseline_node_size + (node.tree.Q[node]/node.tree.N[node])*1000 for node in G.nodes]
    node_size = baseline_node_size

    node_color = [[1, 1-(node.tree.Q[node]/node.tree.N[node]), 1-(node.tree.Q[node]/node.tree.N[node])] for node in G.nodes]
    # node_color='k'

    # node_color = [[int((node.tree.Q[node]/node.tree.N[node])*255), 0, 0] for node in G.nodes]  # RGB value; we only use red and scale the win proportion to the red value
    # node_color = ['#%02x%02x%02x' % (c[0],c[1],c[2]) for c in node_color]

    if tree.path is not None:
        path_pairs = [(tree.path[i],tree.path[i+1]) for i in range(len(tree.path)-1)]
        edge_color = ['r' if pair in path_pairs else 'k' for pair in G.edges]
    else:
        edge_color = 'k'

    # Draws red nodes behind the base nodes, these will appear as outlines of the base nodes and will increase as a function of value
    # nx.draw(G, pos=pos, node_color='red', node_shape='s', node_size=node_size, edge_color=edge_color, with_labels=False, ax=ax)
    #
    # # Draws the nodes that simply mark the node location in the tree
    # nx.draw(G, pos=pos, node_color='k', node_shape='s', node_size=baseline_node_size, with_labels=False, ax=ax)

    nx.draw(G, pos=pos, node_color=node_color, edgecolors='k', node_shape='s', node_size=node_size, edge_color=edge_color, with_labels=False, ax=ax)

    for node, p in pos.items():
        plot_state(node, plot_inventory=False, xshift=p[0], yshift=p[1], scale=scale_of_state, ax=ax)

    if title:
        ax.set_title(title)

    if save_name:
        plt.savefig(save_name, dpi=600, transparent=False)
        plt.close()
    elif show:
        plt.show(block=False)
        plt.pause(0.5)
        plt.close()
    else:
        return f,ax


def create_tree_gif(tree, filename):

    frames = []
    # save axis dimensions of last tree so that the reference axis of the gif will be fixed
    f,ax = plot_tree(tree)
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    plt.close()

    all_trees = tree.trees + [tree]

    # now plot from first to last and concurrently read in the image files into a list
    for i, tree in enumerate(all_trees):
        tree_depth = max([len(node.state['action_history']) for node in tree.leaves['terminal'] + tree.leaves['TBD']])
        f,ax = plt.subplots()
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        plot_tree(tree, ax=ax, save_name = f'tree_{i}.png', title=f'Iteration {i}')

        image = imageio.v2.imread(f'tree_{i}.png')
        frames.append(image)

    # compile into gif
    imageio.mimsave(filename,      # output gif
                    frames,         # array of input frames
                    duration = 250)  # optional: duration in ms
