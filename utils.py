import numpy as np

def flatten_dict(dictionary, parent_key='', sep='_'):
    flattened_dict = {}
    for key, value in dictionary.items():
        new_key = parent_key + sep + key if parent_key else key
        if isinstance(value, dict):
            flattened_dict.update(flatten_dict(value, new_key, sep))
        else:
            flattened_dict[new_key] = value
    return flattened_dict

def flatten_list(xss):
    return [x for xs in xss for x in xs]

def is_ordered_sublist_of(long_list, short_list):

    item = short_list[0]
    if item in long_list:
        i = long_list.index(item)
        if long_list[i:i+len(short_list)] == short_list:
            return True
        else:
            return False
    else:
        return False

def is_reversed_sublist_of(long_list, short_list):

    item = short_list[0]
    if item in long_list:
        i = long_list.index(item)
        if list(reversed(long_list[i-len(short_list)+1:i+1])) == short_list:
            return True
        else:
            return False
    else:
        return False

def entropy(probs):
    return sum([(-np.log2(p)*p) for p in probs])
