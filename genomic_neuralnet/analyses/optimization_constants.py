# Constants for use in model optimization.

import numpy as np
from sklearn.utils.extmath import cartesian

_base_numbers = np.arange(1, 8)
_one_layer_nets = [ (3**x,) for x in _base_numbers] 
_two_layer_nets = [ (2**(x+1), 2**x) for x in _base_numbers if x > 2]
_three_layer_nets = [ (2**(x+2), 2**(x+1), 2**x) for x in _base_numbers if x > 2]

def _get_hidden_sizes():
    return _one_layer_nets + _two_layer_nets + _three_layer_nets

RUNS = 3 
EPOCHS = 2000 # Should be divisible by 250.
BATCH_SIZE = 100
HIDDEN = _get_hidden_sizes()
DROPOUT = (0.1, 0.2, 0.3, 0.4, 0.5)
WEIGHT_DECAY = (1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1)

