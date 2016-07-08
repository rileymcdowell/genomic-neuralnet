# Constants for use in model optimization.

from sklearn.utils.extmath import cartesian

def _get_hidden_sizes():
    hidden_sizes = (2,4,8,16,32,64,128)

    one_layer = cartesian((hidden_sizes,)*1) 
    two_layers = cartesian((hidden_sizes,)*2) 
    three_layers = cartesian((hidden_sizes,)*3) 

    tup = lambda x: map(tuple, x)
    return tup(one_layer) + tup(two_layers) + tup(three_layers)

EPOCHS = 4000
BATCH_SIZE = 100
HIDDEN = _get_hidden_sizes()
DROPOUT = (0.1, 0.2, 0.3, 0.4, 0.5)
WEIGHT_DECAY = (1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
