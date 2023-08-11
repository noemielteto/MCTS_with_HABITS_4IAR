from random import choice
import math
import numpy as np
import copy
from collections import defaultdict
import time
import ast

from .tangram import *
from .utils import *
from .plot import *
from HCRP_LM.ddHCRP_LM import *
from .shape_inventory import get_shape_inventory

NP_NAME = np.__name__
ANGLES_COSTS = {0:0, 90:0.5, 270:0.5, 180:1}


def get_init_element(chunk):
    return {k: chunk[0][k] for k in ('shape','angle')}

def get_predicted_elements(chunk):
    return chunk[1:]

class World():

    def __init__(self, silhouette, shape_inventory, angles):

        self.silhouette         = silhouette
        self.silhouette_y_range = np.where(silhouette.pixels)[0].max() - np.where(silhouette.pixels)[0].min()
        self.silhouette_x_range = np.where(silhouette.pixels)[1].max() - np.where(silhouette.pixels)[1].min()
        self.angles             = angles
        self.shape_inventory    = shape_inventory


        # max x and y should be the max shape width and height
        max_x, max_y = 0, 0
        for shape in shape_inventory.values():
            where = np.nonzero(shape.pixels)
            xs = where[1]
            ys = shape.pixels.shape[0]-1 - where[0]
            max_x = int(max(xs)) if max(xs)>max_x else max_x
            max_y = int(max(ys)) if max(ys)>max_y else max_y

        self.action_space = [{'shape':shape, 'angle':angle, 'x':x, 'y':y} for shape in shape_inventory for x in range(-max_x, max_x+1) for y in range(-max_y, max_y+1) for angle in angles]


class MCTS:
    "Monte Carlo tree searcher."

    def __init__(self, agent, world, start_node=None, c=1, gamma=1, node_budget=10, n_rollouts=1, max_total_rollouts=500, default_N=1):
        self.agent              = agent
        self.n_rollouts         = n_rollouts
        self.max_total_rollouts = max_total_rollouts
        self.total_rollouts     = 0
        self.node_budget        = node_budget
        self.gamma              = gamma      # discount rate
        self.c                  = c          # exploration_weight

        self.Q  	            = defaultdict(lambda:0)
        self.default_N          = default_N  # if 0, UCT will explore all potential nodes first; if 1, exhaustive exploration is not forced
        self.N  	            = defaultdict(lambda:default_N)

        self.known_dyn          = dict()  # we will memoize dynamics here

        self.path               = None

        if start_node is None:

            state = {'action_history'                           : [],
                        'relative_action_history'               : [],
                        'openloop_action_probabilities'         : [],
                        'context'                               : [],
                        'covered'                               : np.zeros((15,15)),  # TODO size should be a parameter
                        'available_inventory'                   : copy.deepcopy(world.shape_inventory)}

            start_node = Node(tree=self, world = copy.deepcopy(world), state = state)  # copy is important because the world properties will change

        self.start_node     = start_node
        self.expanded_tree  = dict()
        self.n_nodes_evaluated   = 0

        self.leaves    = {'TBD':[start_node], 'terminal':[]}
        self.n_iterations = 0

        self.halt=False

    #@profile
    def run_one_MCTS_iter(self, node, show_tree=False, tree_history_gif_name=False, BFS=False, halt_at_solution=True):

        if BFS:
            for leaf in self.leaves['TBD']:
                if self.N[leaf] == self.default_N:
                    self.n_nodes_evaluated += 1
                    # print(f'nodes expanded: {self.n_nodes_evaluated}')
                children = leaf.find_children()

                # terminal
                if children==[]:
                    self.leaves['TBD'].remove(leaf)
                    self.leaves['terminal'].append(leaf)
                    if halt_at_solution:
                        reward = self._rollout(leaf, terminal=True)
                        self.halt = bool(reward)

                # not terminal
                else:
                    self.expanded_tree[leaf] = children
                    self.leaves['TBD'].remove(leaf)
                    for child in self.expanded_tree[leaf]:
                        self.leaves['TBD'].append(child)

                if show_tree:
                    plot_tree(self, show=True)

        else:

            path = self._traverse_tree(node)
            # print(path)
            leaf = path[-1]
            # if we haven't done rollouts from this node before
            # print('-----------------------------------------------')
            # print(f'{self.N[leaf]} rollouts done from leaf {leaf}')
            # print(f'tree: {self.expanded_tree}')
            # print(f'default N is {self.default_N}')
            # print(f'rollout counts are {self.N}')
            if self.N[leaf] == self.default_N:
                self.n_nodes_evaluated += 1
                print(f'nodes evaluated: {self.n_nodes_evaluated}')
            # leaf_copy = copy.deepcopy(leaf)

            children = leaf.find_children()
            if children==[]:
                # print('terminal')
                if leaf in self.leaves['TBD']:
                    # if terminal and we haven't determined this in previous iterations
                    # print('terminal')
                    self.leaves['TBD'].remove(leaf)
                    self.leaves['terminal'].append(leaf)

                for i in range(self.n_rollouts):
                    reward = self._rollout(leaf, terminal=True)
                    self._backpropagate(path, reward)
                    self.halt = bool(reward)

            else:
                # print('not terminal')
                # if it's not terminal, we are expanding it now and it won't be a leaf anymore, its children become leaves
                self.expanded_tree[leaf] = children
                # print(f'leaves: {self.leaves}')
                self.leaves['TBD'].remove(leaf)
                for child in self.expanded_tree[leaf]:
                    self.leaves['TBD'].append(child)

                for i in range(self.n_rollouts):
                    reward = self._rollout(leaf)
                    self._backpropagate(path, reward)

            if show_tree:
                plot_tree(self, show=True)

        self.n_iterations += 1

    #@profile
    def _traverse_tree(self, node):
        "Find an unexplored descendent of `node`"

        path = []
        while True:
            path.append(node)
            # node is unexplored
            if node not in self.expanded_tree:
                return path

            node = self._select_with_tree_policy(node)  # descend a layer deeper

    #@profile
    def _select_with_tree_policy(self, node):

        objective = list(
                        self._uct_values_of_children(node)
                        + self._habit_values_of_children(node)
                        - self._simulation_costs_of_children(node)
                        )
        # print(node.find_children())
        # print(self._uct_values_of_children(node))
        # print(self._habit_values_of_children(node))
        # print(self._simulation_costs_of_children(node))
        # print(objective)
        # print('---------------------------------------------------------------')
        # choose first max value
        # max_index = objective.index(max(objective))

        # consider all values that are max
        max_indeces = [i for i in range(len(objective)) if objective[i]==max(objective)]
        max_index = np.random.choice(max_indeces)

        return list(self.expanded_tree[node])[max_index]

    #@profile
    def _select_with_greedy_policy(self, node):

        if node not in self.expanded_tree:
            return np.random.choice(node.find_children())

        else:
            # objective = list(self._uct_values_of_children(node, no_exploration=True))  # TODO at generating the plan, should we be greedy wrt rewards or rewards plus habits; probs latter but earlier I had only rewards
            objective = list(
                            self._uct_values_of_children(node, no_exploration=True)
                            )
            max_index = objective.index(max(objective))
            return list(self.expanded_tree[node])[max_index]

    #@profile
    def _uct_values_of_children(self, node, no_exploration=False):

        N_all_node = 0
        for child in self.expanded_tree[node]:
            N_all_node += self.N[child]
        x = np.full(len(self.expanded_tree[node]), np.nan)
        for i, child in enumerate(self.expanded_tree[node]):

            if self.N[child] == 0:
                # allow for division by zero that would equal to inf; this results in all children being tried out at least once in case the default N value is 0
                x[i] = math.inf
            else:
                if no_exploration:
                    x[i] = self.Q[child] / self.N[child]
                else:
                    x[i] = self.Q[child] / self.N[child] + self.c * math.sqrt(np.log(N_all_node) / self.N[child])

        return x

    #@profile
    def _habit_values_of_children(self, node, no_exploration=False):

        x = np.full(len(self.expanded_tree[node]), np.nan)
        for i, child in enumerate(self.expanded_tree[node]):
            x[i] = child.habit_value

        return self.agent.H * x

    def _simulation_costs_of_children(self, node, no_exploration=False):

        x = np.full(len(self.expanded_tree[node]), np.nan)
        for i, child in enumerate(self.expanded_tree[node]):
            x[i] = child.simulation_cost

        return self.agent.S * x

    #@profile
    def _rollout(self, node, terminal=False):
        "Returns the reward for a random simulation (to completion) of `node`"

        if terminal:
            return node.reward()

        else:
            
            self.total_rollouts += 1

            children = node.find_children(allow_openloop=False)
            while children != []:

                node = np.random.choice(children)
                children = node.find_children(allow_openloop=False)  # TODO we could also allow openloop here

            return node.reward()

    #@profile
    def _backpropagate(self, path, reward):
        "Send the reward back up to the ancestors of the leaf"
        for node in reversed(path):
            self.N[node] += 1
            self.Q[node] += reward * (self.gamma ** (node.n_primitive_steps - 1))

    def n_distractor_nodes_in_tree(self):
        solution_terminals  = [terminal for terminal in self.leaves['terminal'] if terminal.reward()==1]
        solution_paths      = [solution_terminal.get_path() for solution_terminal in solution_terminals]
        solution_nodes      = set(flatten(solution_paths))
        all_nodes           = list(self.expanded_tree.keys()) + self.leaves['terminal']
        distractor_nodes    = [node for node in all_nodes if node not in solution_nodes]
        return len(distractor_nodes)

    def average_branching_factor(self):
        return np.mean([len(self.expanded_tree[node]) for node in self.expanded_tree.keys()])

class Node:

    def __init__(self, tree, world, state, n_primitive_steps=0, simulation_cost=0, habit_value=0, upstream_entropy=0, parent=None):
        self.tree               = tree
        self.world              = world  # TODO if tree has world then nodes don't need to have it
        self.state              = state
        # {'action_history':action_history, 'context':context, 'covered':covered, 'available_inventory':available_inventory}

        if parent is not None:
            self.n_primitive_steps  = parent.n_primitive_steps + len(state['action_history'][-1])
            self.simulation_cost    = simulation_cost  # TODO work this out
            # self.habit_value        = np.sum(1 - np.array(state['openloop_action_probabilities'])) ## Sum of 1-p for all actions
            # self.habit_value        = state['openloop_action_probabilities'][0] + (len(state['openloop_action_probabilities'])-1)  # cost of first action is proportional to it's probability and the cost of every following action in the chunk is at minimum, as they are predetermined; however, rather than quantifying the cost relative to uniform porbabilities, we quantify the benefit from 0 to 1 as the probability
            self.habit_value        = state['openloop_action_probabilities'][0]  # not biasing the model to prefer longer chunks directly; this allows for analyzing the sheer benefit of deeper search with chunks
            self.upstream_entropy   = upstream_entropy
            self.parent             = parent

        else:
            self.n_primitive_steps  = n_primitive_steps
            self.simulation_cost    = simulation_cost  # TODO work this out
            self.habit_value        = habit_value
            self.upstream_entropy   = upstream_entropy
            self.parent             = parent

    def __repr__(self):
        return str(self.state['action_history'])  # there is a many-to-one mapping from absolute to relative (e.g. first absolute action, regardless where the position is, will be x=0, y=0), so we should show absolute if we want to see the distinction between all nodes

    def is_terminal(self):
        # no children
        return self.find_children(allow_openloop=False) == []

    def get_path(self):
        path = [self]
        while path[0]!=self.tree.start_node:
            path.insert(0, path[0].parent)

        return path

    def reward(self):
        if self.tree.agent.generative:
            # TODO if we want to prevent holes, implement this
            # if hole_condition_to_be_implemented:
            if False:
                return 0
            else:
                # convexity objective: proportion of pixels of the polygon that are in the convex hull
                coords = []
                for x in range(self.state.covered.shape[0]):
                    for y in range(self.state.covered.shape[1]):
                        if self.state.covered[x][y] > 0:
                            coords.append([x,y])
                coords = np.array(coords)
                # convexity objective: proportion of pixels of the polygon that are in the convex hull
                return np.sum(a) / ConvexHull(coords).area
        else:
            # unit reward when silhouette matched, zero otherwise
            return 1 if np.array_equal(self.world.silhouette.pixels, self.state['covered']) else 0

    #@profile
    def build_perceptual_chunk(self, openloop_action_chunk):

        action                      = openloop_action_chunk[0]
        shape                       = self.world.shape_inventory[action['shape']]
        perceptual_chunk_pixels     = shape.getDisplacedPixels(action['x'], action['y'], action['angle'])
        i=1
        # while i<len(openloop_action_chunk) and self.valid(action):
        while i<len(openloop_action_chunk):
            action                  = openloop_action_chunk[i]
            shape                   = self.world.shape_inventory[action['shape']]
            perceptual_chunk_pixels += shape.getDisplacedPixels(action['x'], action['y'], action['angle'])
            i+=1

        return perceptual_chunk_pixels

    #@profile
    def valid(self, action):

        shape                   = self.world.shape_inventory[action['shape']]
        new_shape_pixels        = shape.getDisplacedPixels(action['x'], action['y'], action['angle'])
        new_shape_pixels        *= 2  # we tag the identity of the new shape versus the covered area that will be labelled as 1
        new_covered             = (self.state['covered'] + new_shape_pixels).astype(int)
        new_covered_nolayers    = np.where(new_covered>0, 1, 0)

        # Conditions:

        #1 if out of bounds of silhouette
        if not self.tree.agent.generative:
            # is it within bounds?
            # if generative, there is no silhouette constraint (in fact, we are generating the silhouette)
            if self.world.silhouette.outOfBounds(new_covered_nolayers):
                return False

        #2 out of bounds of board
        not_out_of_board_bounds = True
        xs_sorted = sorted(np.where(new_covered_nolayers==1)[0])
        ys_sorted = sorted(np.where(new_covered_nolayers==1)[1])
        if max([xs_sorted[i+1]-xs_sorted[i] for i in range(len(xs_sorted)-1)])>1:
            return False
        if max([ys_sorted[i+1]-ys_sorted[i] for i in range(len(ys_sorted)-1)])>1:
            return False

        #3 if intersects with previously placed shapes
        if int(np.max(new_covered)) != 2:
            return False

        #4 if new shape touches previos ones

        touches = False

        # first shape placed
        if np.sum(self.state['covered']) == 0:
            for x,y in np.argwhere(new_covered==2):
                # touches floor
                if x==new_covered.shape[0]-1:
                    touches = True

        else:
            for x,y in np.argwhere(new_covered==2):
                # touches covered area above pixel
                if x>0:
                    if new_covered[x-1,y] == 1:
                        touches = True
                # touches covered area below pixel
                if x<new_covered.shape[0]-1:
                    if new_covered[x+1,y] == 1:
                        touches = True
                # touches covered area left of pixel
                if y>0:
                    if new_covered[x,y-1] == 1:
                        touches = True
                # touches covered area right of pixel
                if y<new_covered.shape[1]-1:
                    if new_covered[x,y+1] == 1:
                        touches = True

        if not touches:
            return False

        return True

    def absolute_to_relative_actions(self, openloop_action_chunk):

        openloop_relative_action_chunk = []

        # in case first action in history, this will be the reference and it should also not be converted (i.e. shouldn't be centered to zero)
        if len(self.state['action_history'])==0:
            previous_action = openloop_action_chunk[0]
            openloop_relative_action_chunk.append(previous_action)
            openloop_action_chunk = openloop_action_chunk[1:]
        else:
            previous_action = copy.deepcopy(self.state['action_history'][-1][-1])

        for action in openloop_action_chunk:
            relative_action = copy.deepcopy(action)
            relative_action['x'] -= previous_action['x']
            relative_action['y'] -= previous_action['y']
            openloop_relative_action_chunk.append(relative_action)
            previous_action = copy.deepcopy(action)

        return openloop_relative_action_chunk

    def relative_to_absolute_actions(self, openloop_relative_action_chunk):

        openloop_action_chunk = []

        # in case first action in history, this will be the reference and it should also not be converted
        if len(self.state['action_history'])==0:
            previous_action = openloop_relative_action_chunk[0]
            openloop_action_chunk.append(previous_action)
            openloop_relative_action_chunk = openloop_relative_action_chunk[1:]
        else:
            previous_action = copy.deepcopy(self.state['action_history'][-1][-1])

        for relative_action in openloop_relative_action_chunk:
            action = copy.deepcopy(relative_action)
            action['x'] += previous_action['x']
            action['y'] += previous_action['y']
            openloop_action_chunk.append(action)
            previous_action = copy.deepcopy(action)

        return openloop_action_chunk

    def get_openloop_action_probabilities(self, openloop_relative_action_chunk):

        context = copy.deepcopy(self.state['context'])
        openloop_action_probabilities = []
        for relative_action in openloop_relative_action_chunk:
            p = self.tree.agent.sequence_model.word_probability_all_samples(self.tree.agent.t, context, str(relative_action))
            openloop_action_probabilities.append(p)
            # print(context)
            # print(str(relative_action))
            # print(p)
            context.append({x: relative_action[x] for x in ['shape', 'angle']})

        return openloop_action_probabilities

    def update_action_space(self, openloop_action_chunk):
        # Validity check: did we use shapes that weren't even available
        # TODO this has to be simplified

        shapes_used_names = [action['shape'] for action in openloop_action_chunk]
        available_inventory_names = list(self.state['available_inventory'].keys())

        # If shape is used multiple times in the action chunk, violating the rule of the game
        if len(set(shapes_used_names))<len(shapes_used_names):
            return False

        # If the shapes in the action chunk are not available
        elif not (set(shapes_used_names) & set(available_inventory_names)) == set(shapes_used_names):
            return False

        # If the shapes are available, we compute the remaining available inventory after using the shapes
        else:
            to_remove = []
            for available in available_inventory_names:
                for shape in shapes_used_names:
                    if shape==available:
                        to_remove.append(available)

            new_available_inventory = {k: self.state['available_inventory'][k] for k in available_inventory_names if k not in to_remove}

            return new_available_inventory



    #@profile
    def generate_relative_action_chunk(self, action):

        sequence_model = self.tree.agent.sequence_model
        t              = self.tree.agent.t

        context = copy.deepcopy(self.state['context'])
        context.append({x: action[x] for x in ['shape', 'angle']})
        pred_distr = sequence_model.get_predictive_distribution(t=t, u=context)

        openloop_relative_action_chunk = self.absolute_to_relative_actions([action])

        # after first observation, the conditional entropy will be 0 (unless the uniform prior contains all possible actions -- this could be done if the action space was onlt shapes*rotations; but it's unfeasible because it's shapes*rotations*relative position)
        if len(pred_distr)>1:
            # print(np.round(pred_distr),2)
            # print(entropy(pred_distr))
            while (entropy(pred_distr) < self.tree.agent.openloop_threshold) and (len(openloop_relative_action_chunk)<self.tree.agent.max_openloop_length):

                # predict a relative action
                new_action_relative  = ast.literal_eval(sequence_model.predict_next_word(t=t, u=context))
                openloop_relative_action_chunk.append(new_action_relative)

                # predictive context only contains shape and angle, not position (because we need to disregard the position of the starting element, either absolute or relative to a previous noncunked element)
                context.append({x: new_action_relative[x] for x in ['shape', 'angle']})
                pred_distr      = sequence_model.get_predictive_distribution(t=t, u=context)

        if len(openloop_relative_action_chunk)==1:
            return False
        else:
            return openloop_relative_action_chunk

    #@profile
    def simulate_next_state(self, openloop_action_chunk=None, openloop_relative_action_chunk=None):
        # action: dict of [shape, angle, x, y]

        if openloop_action_chunk is None:
            openloop_action_chunk = self.relative_to_absolute_actions(openloop_relative_action_chunk)
        elif openloop_relative_action_chunk is None:
            openloop_relative_action_chunk = self.absolute_to_relative_actions(openloop_action_chunk)
        # else openloop_action_chunk is None and openloop_relative_action_chunk is None:
        #     return RuntimeError

        c_a = str([self.state['covered'], openloop_action_chunk])

        # retrieve
        if c_a in self.tree.known_dyn:
        # if False:
            # print('retrieve')
            perceptual_chunk_pixels = self.tree.known_dyn[c_a]
            # if not numpy array
            if not type(perceptual_chunk_pixels).__module__ == NP_NAME:
                return False

        # compute anew
        else:

            # TODO not an elegant way of checking... update_action_space() could return False or something instead
            if type(self.update_action_space(openloop_action_chunk))!=dict:
                return False

            if not self.valid(openloop_action_chunk[0]):
                self.tree.known_dyn[c_a] = False
                return False

            perceptual_chunk_pixels = self.build_perceptual_chunk(openloop_action_chunk)
            self.tree.known_dyn[c_a] = copy.deepcopy(perceptual_chunk_pixels)

        new_state = {'action_history'                  : copy.deepcopy(self.state['action_history'])                           + [openloop_action_chunk],
                        'relative_action_history'      : copy.deepcopy(self.state['relative_action_history'])                  + [openloop_relative_action_chunk],
                        'openloop_action_probabilities': self.get_openloop_action_probabilities(openloop_relative_action_chunk),
                        'context'                      : copy.deepcopy(self.state['context']) + [{x: action[x] for x in ['shape', 'angle']} for action in openloop_action_chunk],  # predictive context only contains shape and angle, not position (because we need to disregard the position of the starting element, either absolute or relative to a previous noncunked element)
                        'covered'                      : self.state['covered']                                                 + perceptual_chunk_pixels,
                        'available_inventory'          : self.update_action_space(openloop_action_chunk)}

        return new_state

    #@profile
    def find_children(self, allow_openloop=True):
        children = []

        for shape in self.state['available_inventory'].keys():
            for angle in self.world.angles:
                for x in range(self.world.silhouette_x_range):
                    for y in range(self.world.silhouette_y_range):
                        action = {'shape':shape, 'angle':angle, 'x':x, 'y':y}
                        next_state_with_primitive_action = self.simulate_next_state(openloop_action_chunk=[action])

                        if next_state_with_primitive_action:
                            next_states = [next_state_with_primitive_action]

                            if allow_openloop:
                                openloop_relative_action_chunk = self.generate_relative_action_chunk(action=action)
                                if openloop_relative_action_chunk:
                                    next_state_with_action_chunk = self.simulate_next_state(openloop_relative_action_chunk=openloop_relative_action_chunk)
                                    if next_state_with_action_chunk:
                                        next_states.append(next_state_with_action_chunk)

                            for next_state in next_states:

                                child = Node(tree               = self.tree,
                                                world           = self.world,
                                                state           = next_state,
                                                simulation_cost = ANGLES_COSTS[angle],
                                                parent          = self
                                            )

                                children.append(child)

        return children

    # def get_prob_distr_of_children(self):
    # # TODO: needed for upstream entropy calculations
    #     probs = []
    #     for child in self.expanded_tree:
    #         # probability is the start action of latest action chunk
    #         child.state['action_history'][-1][0]
    #         print(str(child.state['relative_action_history'][-1][0]))
    #         probs.append(self.tree.agent.sequence_model.word_probability_all_samples(self.tree.agent.t, child.state['context'], str(child.state['relative_action_history'][-1][0])))
    #
    #     return [p/sum(probs) for p in probs]
    #
    # def get_entropy(self):
    # # TODO: needed for upstream entropy calculations
    #     probs = self.get_prob_distr_of_children()
    #     return sum([(-np.log2(p)*p) for p in probs])


class HCRP_MCTS_Agent:

    def __init__(self, HCRP_level               = 3,
                        action_space            = None,
                        heuristic_coefficient   = 1,
                        simulation_coefficient  = 1,
                        habit_coefficient       = 1,
                        entropy_coefficient     = 0,
                        openloop_threshold      = 1,
                        max_openloop_length     = 4,
                        node_budget             = 10,
                        n_rollouts              = 5,
                        max_total_rollouts      = 500,
                        c                       = 1,
                        gamma                   = 1,
                        default_N               = 1,
                        generative              = False,
                        halt_at_solution        = True):

        # self.sequence_model     = HCRP_LM(strength=[1]*HCRP_level, decay_constant=[10]*HCRP_level)  # TODO play with forgetfulness
        self.sequence_model     = HCRP_LM(strength=[10]*HCRP_level, dishes=[str(a) for a in action_space])
        self.D                  = heuristic_coefficient
        self.S                  = simulation_coefficient
        self.H                  = habit_coefficient
        self.E                  = entropy_coefficient
        self.openloop_threshold = openloop_threshold  # if conditional distribution of next action has lower entropy than this, we stay openloop
        self.max_openloop_length= max_openloop_length
        self.t                  = 0   # track the timelife of our agent across tasks
        self.node_budget        = node_budget
        self.n_rollouts         = n_rollouts
        self.max_total_rollouts = max_total_rollouts
        self.c                  = c     # exploration bonus weight for MCTS
        self.gamma              = gamma
        self.default_N          = default_N
        self.generative         = generative

    def find_path(self, world, show_tree=False, tree_history_gif_name=False, BFS=False, halt_at_solution=True):

        tree = MCTS(agent=self, world=world, n_rollouts=self.n_rollouts, max_total_rollouts=self.max_total_rollouts, c=self.c, gamma=self.gamma, default_N=self.default_N)

        if self.generative:
            while (tree.n_nodes_evaluated < self.node_budget):
                tree.run_one_MCTS_iter(node=tree.start_node, show_tree=show_tree, tree_history_gif_name=tree_history_gif_name, BFS=BFS, halt_at_solution=halt_at_solution)

        else:
            if tree_history_gif_name:
                trees=[]
            while (not tree.halt) and (tree.n_nodes_evaluated < self.node_budget) and tree.leaves['TBD'] and (tree.total_rollouts < tree.max_total_rollouts):
                tree.run_one_MCTS_iter(node=tree.start_node, show_tree=show_tree, tree_history_gif_name=tree_history_gif_name, BFS=BFS, halt_at_solution=halt_at_solution)

                if tree_history_gif_name:
                    trees.append(copy.deepcopy(tree))

        path = [tree.start_node]
        node = tree.start_node
        while not node.is_terminal():
            node = tree._select_with_greedy_policy(node)
            path.append(node)
        tree.path = path

        # print('------------------------')
        # print(tree.expanded)

        if tree_history_gif_name:
            tree.trees = trees

        return tree

    def solve(self,
                world,
                show_tree               = False,
                tree_history_gif_name   = False,
                BFS                     = False,
                optimal_steps_plot_name = False,
                optimal_path_plot_name  = False,
                update_sequenece_model  = True):

        start = time.time()
        tree = self.find_path(world=world, show_tree=show_tree, tree_history_gif_name=tree_history_gif_name)

        if tree_history_gif_name:
            create_tree_gif(tree, tree_history_gif_name)

        t = time.time() - start
        # print(f'took {t} to solve')

        if update_sequenece_model:
            # TODO the dishes are strings and we do literal eval when sampling from HCRP in order to convert back the action into the original dict
            self.sequence_model.fit(t_start          = self.t,
                                    corpus_segments  = tree.path[-1].state['context'],
                                    choices_segments = [str(x) for x in flatten_list(tree.path[-1].state['relative_action_history'])],
                                    frozen           = False,
                                    observation      = 'choices'
                                    )
            self.t += len(tree.path[-1].state['context'])

        if optimal_path_plot_name:
            plot_state(tree.path[-1], save_name=optimal_path_plot_name+ '.png')

        if optimal_steps_plot_name:
            for i,node in enumerate(tree.path):
                plot_state(node, save_name=optimal_steps_plot_name + '_' + str(i) + '.png')

        return tree
