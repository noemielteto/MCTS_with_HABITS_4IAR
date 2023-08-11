from .tangram import *
import os
import _pickle as cPickle
import numpy as np
import platform
import copy
#
# mock_shape = TangramShape('mock', np.zeros((2,2)))
# print(mock_shape)

def get_trial_sequence(silhouette_filename,
                        n_trials_per_type = {'NoSeq'            : 9,
                                            'SeqOblig'          : 12,
                                            'SeqReverse'        : 0,
                                            'SeqAmbig'          : 0,
                                            'SoftAdverserial'   : 0,
                                            'HardAdverserial'   : 0},
                        min_complexity  = 1,
                        max_complexity  = 50,
                        max_distance    = 0,
                        randomize_order = True,
                        unambiguous     = True):

    # select requested trial types
    n_trials_per_type = dict([(k,v) for (k,v) in n_trials_per_type.items() if v != 0])

    cwd = os.getcwd()
    if 'Windows' in platform.platform().split('-'):
        slash = '\\'
    else:
        slash = '/'
    silhouette_path = cwd + f"{slash}silhouettes{slash}"

    p = silhouette_path + silhouette_filename
    with open(p, 'rb') as handle:
        silhouettes = cPickle.load(handle)

    ############################################################################
    # Remove duplicates

    silhouettes_new = []
    silhouettes_solutions = []
    for s in silhouettes:
        if s['solutions'] not in silhouettes_solutions:
            silhouettes_new.append(s)
            silhouettes_solutions.append(s['solutions'])
    silhouettes = copy.deepcopy(silhouettes_new)

    ############################################################################
    # If we want unique solutions

    # if unambiguous:
    #     silhouettes_new = []
    #     for s in silhouettes:
    #         if len(s['solutions'])==1:
    #             silhouettes_new.append(s)
    # silhouettes = copy.deepcopy(silhouettes_new)

    ############################################################################

    silhouettes_by_complexity = {}
    for complexity_value in range(min_complexity, max_complexity+1):
        silhouettes_with_complexity = [s for s in silhouettes if complexity_value-max_distance<=int(s['complexity'])<=complexity_value+max_distance]

        # if there are silhouettes of this complexity at all
        if len(silhouettes_with_complexity):
            n_silhouettes_per_type = [len([s for s in silhouettes if (s['trial_type']==trial_type and (complexity_value-max_distance<=int(s['complexity'])<=complexity_value+max_distance))]) for trial_type in n_trials_per_type.keys()]
            # print(f'for complexity value {complexity_value} {n_silhouettes_per_type}')
            min_n_silhouettes_per_type = min(n_silhouettes_per_type)

            # if there is at least one silhouette of each type with this complexity value
            if min_n_silhouettes_per_type:
                silhouettes_by_complexity[complexity_value] = []
                for trial_type in n_trials_per_type.keys():
                    silhouettes_to_add = np.random.choice([s for s in silhouettes if (s['trial_type']==trial_type and (complexity_value-max_distance<=int(s['complexity'])<=complexity_value+max_distance))], min_n_silhouettes_per_type)
                    silhouettes_by_complexity[complexity_value] += list(silhouettes_to_add)
                if len(silhouettes_by_complexity[complexity_value]) == 0:
                    del silhouettes_by_complexity[complexity_value]

    trial_sequence = []
    n_trials_per_type_chosen = dict([(k,0) for k in n_trials_per_type.keys()])

    while n_trials_per_type_chosen != n_trials_per_type:
        complexity_value = np.random.choice(list(silhouettes_by_complexity.keys()))
        for trial_type in n_trials_per_type.keys():
            if n_trials_per_type_chosen[trial_type] < n_trials_per_type[trial_type]:

                choose_from = [s for s in silhouettes_by_complexity[complexity_value] if s['trial_type']==trial_type]
                if len(choose_from)==0:
                    raise ValueError("Not enough silhouettes in file to create this sequence.")

                chosen = np.random.choice(choose_from)
                silhouettes_by_complexity[complexity_value].remove(chosen)
                # if we emptied it out the list, we won't want to pick this compplexity value again
                if len(silhouettes_by_complexity[complexity_value]) == 0:
                    del silhouettes_by_complexity[complexity_value]
                trial_sequence.append(chosen)
                n_trials_per_type_chosen[trial_type] += 1

    if randomize_order:
        # shuffled
        trial_sequence = list(np.random.choice(trial_sequence, size=len(trial_sequence), replace=False))
    else:
        # ordered by trial type
        trial_sequence = [s for trial_type in n_trials_per_type.keys() for s in trial_sequence if s['trial_type']==trial_type]

    return trial_sequence
