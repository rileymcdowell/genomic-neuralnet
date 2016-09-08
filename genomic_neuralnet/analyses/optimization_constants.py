# Constants for use in model optimization.

import numpy as np
from sklearn.utils.extmath import cartesian
from genomic_neuralnet.config import NUM_FOLDS
from genomic_neuralnet.util import get_is_time_stats, get_timing_size

LARGE_TIMING_HIDDEN_SHAPE = [(64, 32)]
SMALL_TIMING_HIDDEN_SHAPE = [(27,)]

_base_numbers = np.arange(1, 9, 2)
_one_layer_nets = [ (3**x,) for x in _base_numbers] 
_two_layer_nets = [ (2**(x+1), 2**x) for x in _base_numbers]
_three_layer_nets = [ (2**(x+2), 2**(x+1), 2**x) for x in _base_numbers]

def _get_hidden_sizes():
    if get_timing_size() == 'small': 
        return SMALL_TIMING_HIDDEN_SHAPE
    elif get_timing_size() == 'large':
        return LARGE_TIMING_HIDDEN_SHAPE 
    else:
        return _one_layer_nets + _two_layer_nets + _three_layer_nets

RUNS = 2
EPOCHS = 12000 # Should be divisible by 1000.
HIDDEN = _get_hidden_sizes()
DROPOUT = (0.2, 0.3, 0.4, 0.5)
WEIGHT_DECAY = (1e-7, 1e-6, 1e-5, 1e-4)

assert len(DROPOUT) == len(WEIGHT_DECAY), 'Params must have same length'

# The amount to multiply by to correct for sample size when missing one parameter.
SINGLE_MULTIPLIER = len(DROPOUT)
# The amount to multiply by to correct for sample size when missing two parameters.
DOUBLE_MULTIPLIER = len(DROPOUT) * len(WEIGHT_DECAY)

def main():
    """ Print out some info about network parameters """
    print(_one_layer_nets)
    print(_two_layer_nets)
    print(_three_layer_nets)
    print('Total: ', len(_get_hidden_sizes()))

    print('N: hidden * dropout * weight-decay')
    hidden = len(_get_hidden_sizes())
    format_tuple = hidden, len(DROPOUT), len(WEIGHT_DECAY)
    print('N: {} * {} * {}'.format(*format_tuple))
    print('N: {}'.format(np.prod(format_tuple)))

    print('Containing {} runs of {} folds each'.format(RUNS, NUM_FOLDS))


    print('Total Fits: {}'.format(np.prod(np.array(format_tuple + (RUNS,) + (NUM_FOLDS,)))))


    assert len(_one_layer_nets) == len(_two_layer_nets)
    assert len(_two_layer_nets) == len(_three_layer_nets)

if __name__ == '__main__':
    main()
