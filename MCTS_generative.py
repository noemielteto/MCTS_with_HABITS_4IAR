from .MCTS_Tangram import *
from .shape_inventory import get_shape_inventory
from .plot import plot_state, plot_tree

import _pickle as cPickle
import random

def get_empirical_complexity(world, n_trials=3, min_n_nodes=1, max_n_nodes=200):
    # it is crucial that default_n is 1, meaning no obligatory first-time exploration -- allows for solution with less nodes, will be used in experiments

    outlier_complexity = False

    recognition_agent = HCRP_MCTS_Agent(HCRP_level=2, openloop_threshold=0, habit_coefficient=0, node_budget=200, n_rollouts=1, c=1, default_N=1, generative=False, halt_at_solution=True, action_space=world.action_space)
    start = time.time()
    n_nodes_evaluated_trials = []
    for trial in range(n_trials):
        print(trial)
        tree = recognition_agent.find_path(world)
        n_nodes_evaluated = tree.n_nodes_evaluated

        if n_nodes_evaluated<min_n_nodes or n_nodes_evaluated>max_n_nodes:
            outlier_complexity = True
            break
        else:
            n_nodes_evaluated_trials.append(n_nodes_evaluated)

    median_complexity = np.median(n_nodes_evaluated_trials)
    print(f'took {time.time()-start} to assess minimum number of nodes.')

    if outlier_complexity:
        return None
    else:
        return median_complexity


def generate_silhouettes(exp_name, openloop_action_chunk, silhouette_size, silhouette_types, empirical_complexity=True, plot_tree=False, online_board_height=10):

    silhouettes = []
    silhouette_counts = dict([(k,0) for k in silhouette_types])

    angles = [0]
    chunk_size = len(openloop_action_chunk)
    reduced_openloop_action_chunk = [action['shape'] for action in openloop_action_chunk]
    shape_inventory, chunked_shape_inventory, unchunked_shape_inventory = get_shape_inventory(reduced_openloop_action_chunk)
    board_size = shape_inventory['shape_1'].pixels.shape[0]

    mock_silhouette  = TangramSilhouette(np.ones((board_size,board_size)))
    world            = World(mock_silhouette, shape_inventory, angles)
    generative_agent = HCRP_MCTS_Agent(HCRP_level=2, openloop_threshold=0, habit_coefficient=0, node_budget=0, c=2, generative=True, action_space=world.action_space)  # exploration has to be high in generative case because any result is rewarded, and without a high exploration bonus, the model will be lead by the Q values and persevering on the same trajectory all the time
    tree             = MCTS(agent=generative_agent, world=world)
    chunk_pixels     = tree.start_node.build_perceptual_chunk(openloop_action_chunk)
    chunk            = TangramShape(pixels=chunk_pixels, name='chunk')

    chunk_fragments = {}
    chunk_fragments_reduced_action_sequences = {}

    for initial_fragment_size in range(1,chunk_size):
        openloop_action_chunk_fragment                                  = openloop_action_chunk[:initial_fragment_size]
        chunk_fragment_pixels                                           = tree.start_node.build_perceptual_chunk(openloop_action_chunk_fragment)
        chunk_fragment                                                  = TangramShape(pixels=chunk_fragment_pixels, name='chunk')
        chunk_fragments[initial_fragment_size]                          = chunk_fragment
        chunk_fragments_reduced_action_sequences[initial_fragment_size] = [a['shape'] for a in openloop_action_chunk_fragment]

    while min(silhouette_counts.values()) < 100:
    # While we have fewer than 100 examples of each silhouette type

        trial_type = np.random.choice(a=['NoSeq', 'Seq', 'Adverserial'], p=[0.2, 0.6, 0.2])

        if trial_type=='Seq':
            # For generating SeqOblig, SeqAmbig, and SeqReverse
            shape_inventory_subset              = dict(random.sample(unchunked_shape_inventory.items(), silhouette_size-chunk_size))
            shape_inventory_subset['chunk']     = chunk

        elif trial_type=='Adverserial':
            fragment_size = np.random.choice(list(chunk_fragments.keys()))

            shape_inventory_subset              = dict(random.sample(unchunked_shape_inventory.items(), silhouette_size-fragment_size))
            shape_inventory_subset['fragment']  = chunk_fragments[fragment_size]

        else:
            # For generating NoSeq
            shape_inventory_subset              = dict(random.sample(unchunked_shape_inventory.items(), silhouette_size))

        # train silhouette here doesn't matter because we will disregard its constraints
        world = World(mock_silhouette, shape_inventory_subset, angles)
        start = time.time()
        tree = generative_agent.find_path(world)
        print(f'took {time.time()-start} to generate')
        silhouette_pixels               = tree.path[-1].state['covered']
        silhouette_pixels_leftshifted   = np.roll(silhouette_pixels, -np.where(silhouette_pixels==1)[1].min(), axis=1)
        silhouette = TangramSilhouette(silhouette_pixels_leftshifted)

        ###############################################################################
        # ASSESS SILHOUETTE TYPE

        world = World(silhouette, shape_inventory, angles)
        # Here we re-initiliase the agent because it's node budget gets decreased in the complexity assessment part of the code below
        recognition_agent = HCRP_MCTS_Agent(HCRP_level=2, openloop_threshold=0, habit_coefficient=0, node_budget=5000, n_rollouts=1, c=1, default_N=0, generative=False, action_space=world.action_space)
        start = time.time()
        full_tree = recognition_agent.find_path(world, BFS=True, halt_at_solution=False)
        print(f'took {time.time()-start} to build full tree')

        solution_terminals = [terminal for terminal in full_tree.leaves['terminal'] if terminal.reward()==1]

        chunk_forward, chunk_backward, chunk_nonadjacent = False, False, False
        if trial_type=='Seq':
            # Decide whether SeqOblig, SeqAmbig, or SeqReverse

            for terminal in solution_terminals:
                relative_action_history = flatten(terminal.state['relative_action_history'])
                reduced_action_history = [action['shape'] for action in relative_action_history]  ## TODO state dimensionality reduction should be a function

                if is_ordered_sublist_of(long_list=reduced_action_history, short_list=reduced_openloop_action_chunk):
                    chunk_forward = True

                elif is_reversed_sublist_of(long_list=reduced_action_history, short_list=reduced_openloop_action_chunk):
                    chunk_backward = True

                elif set(reduced_openloop_action_chunk).issubset(reduced_action_history):
                    chunk_nonadjacent = True

            # specify trial type further
            if chunk_forward and not chunk_backward and not chunk_nonadjacent:
                trial_type = 'SeqOblig'
            elif chunk_backward and not chunk_forward and not chunk_nonadjacent:
                trial_type = 'SeqReverse'
            else:
                trial_type = 'SeqAmbig'

        elif trial_type=='Adverserial':
            # decide whether SoftAdverserial or HardAdverserial

            if fragment_size>1:
                # Optional: We could assert that the fragment is SeqOblig?
                trial_type = 'HardAdverserial'

            else:
                trial_type = 'SoftAdverserial'

        # If we don't want to include special trial types
        if trial_type not in silhouette_types:
            continue

        # Silhouette complexity proxy
        if empirical_complexity:
            complexity = get_empirical_complexity(world)
        else:
            # cheaper, weaker proxy
            complexity = full_tree.n_nodes_evaluated/len(solution_terminals)

        if complexity:

            solutions = []

            for solution in solution_terminals:

                solution_y_inverted = []

                for action in solution.state['action_history']:
                    shape = action[0]['shape']
                    block_height = shape_inventory[shape].pixels.shape[0] - min(np.where(shape_inventory[shape].pixels)[0])

                    solution_y_inverted.append(shape)
                    solution_y_inverted.append(action[0]['x'])
                    solution_y_inverted.append(online_board_height - block_height - action[0]['y'])

                solutions.append(solution_y_inverted)

            silhouettes.append({'silhouette' : silhouette,
                                'trial_type' : trial_type,
                                'complexity' : complexity,
                                'solutions'  : solutions})

            silhouette_counts[trial_type] +=1
            print(f'Silhouette counts for stimset {exp_name}: {silhouette_counts}')

        else:
            print(f'outlier complexity')
            continue

        with open(f'silhouettes_for_{exp_name}.pickle', 'wb') as handle:
            cPickle.dump(silhouettes, handle)

        if plot_tree:
            plot_tree(full_tree, save_name=f'{exp_name}_{trial_type}_{str(len(silhouettes[trial_type]))}.png')
