import numpy as np

from genomic_neuralnet.methods.generic_keras_net import get_net_prediction

def get_knet_pred( train_data, train_truth, test_data, test_truth
                 , hidden=(5,), weight_decay=0.0, dropout_prob=0.0
                 , learning_rate=None, epochs=25, batch_size=100): 

    return get_net_prediction( train_data, train_truth, test_data, test_truth
                             , net_config=net_config, weight_decay=weight_decay)

