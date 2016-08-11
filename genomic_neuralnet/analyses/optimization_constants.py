# Constants for use in model optimization.

import numpy as np
from sklearn.utils.extmath import cartesian
from genomic_neuralnet.config import NUM_FOLDS

_base_numbers = np.arange(1, 8)
_one_layer_nets = [ (3**x,) for x in _base_numbers] 
_two_layer_nets = [ (2**(x+1), 2**x) for x in _base_numbers]
_three_layer_nets = [ (2**(x+2), 2**(x+1), 2**x) for x in _base_numbers]

def _get_hidden_sizes():
    return _one_layer_nets + _two_layer_nets + _three_layer_nets

RUNS = 3 
EPOCHS = 2000 # Should be divisible by 250.
BATCH_SIZE = 100
HIDDEN = _get_hidden_sizes()
DROPOUT = (0.2, 0.3, 0.4, 0.5)
WEIGHT_DECAY = (1e-7, 1e-6, 1e-5, 1e-4)

def main():
    """ Print out some info about network parameters """
    print(_one_layer_nets)
    print(_two_layer_nets)
    print(_three_layer_nets)
    print('Total: ', len(_get_hidden_sizes()))

    print('N: hidden * dropout * weight-decay')
    hidden = len(_get_hidden_sizes()) / 3
    format_tuple = hidden, len(DROPOUT), len(WEIGHT_DECAY)
    print('N: {} * {} * {}'.format(*format_tuple))
    print('N: {{{}, {}, {}}}'.format(*tuple(np.cumprod(format_tuple))))

    print('Containing {} runs of {} folds each'.format(RUNS, NUM_FOLDS))


    assert len(_one_layer_nets) == len(_two_layer_nets)
    assert len(_two_layer_nets) == len(_three_layer_nets)

if __name__ == '__main__':
    main()
