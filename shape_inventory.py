from .tangram import *

def get_shape_inventory(chunked_shapes=None, inventory='default'):

    if inventory=='default':

        shape_1 = TangramShape(np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]
                                         ),
                                         'shape_1')

        shape_2 = TangramShape(np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0]]
                                         ),
                                         'shape_2')

        shape_3 = TangramShape(np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0]]
                                         ),
                                         'shape_3')

        shape_4 = TangramShape(np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [1,1,1,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]
                                         ),
                                         'shape_4')

        shape_5 = TangramShape(np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [1,1,1,0,0,0,0,0,0,0,0,0,0,0,0]]
                                         ),
                                         'shape_5')

        shape_6 = TangramShape(np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0]]
                                         ),
                                         'shape_6')
        shape_7 = TangramShape(np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [1,1,1,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [1,0,1,0,0,0,0,0,0,0,0,0,0,0,0]]
                                         ),
                                         'shape_7')

        shape_inventory = {
                            'shape_1': shape_1,
                            'shape_2': shape_2,
                            'shape_3': shape_3,
                            'shape_4': shape_4,
                            'shape_5': shape_5,
                            'shape_6': shape_6,
                            'shape_7': shape_7
                            }

    elif inventory=='WM':

        shape_1 = TangramShape(np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0]]
                                         ),
                                         'shape_1')

        shape_2 = TangramShape(np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]
                                         ),
                                         'shape_2')

        shape_3 = TangramShape(np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [1,1,1,0,0,0,0,0,0,0,0,0,0,0,0]]
                                         ),
                                         'shape_3')

        shape_4 = TangramShape(np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0]]
                                         ),
                                         'shape_4')

        shape_inventory = {
                            'shape_1': shape_1,
                            'shape_2': shape_2,
                            'shape_3': shape_3,
                            'shape_4': shape_4
                            }

    if chunked_shapes is None:
        chunked_shapes = []

    chunked_shape_inventory = dict([(k,v) for (k,v) in shape_inventory.items() if k in chunked_shapes])
    unchunked_shape_inventory = dict([(k,v) for (k,v) in shape_inventory.items() if k not in chunked_shapes])

    return shape_inventory, chunked_shape_inventory, unchunked_shape_inventory