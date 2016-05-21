from __future__ import print_function

class NeuralnetConfig(object):
    def __init__(self): 
        """ 
        A container for neural network training options.

        Attributes:
            max_epochs       The maximum number of epochs to train under any circumstance.
                             If try_convergence == False, this is also the minimum number of epochs
                             to train. default = 10000

            continue_epochs  Number of epochs to continue training when trying for a lower error score. 
                             This option is ignored if try_convergence is False. default = 25

            try_convergence  Try to reach a point where the error of the network is no longer decreasing,
                             suggesting that training has converged to the minimum error point. If False,
                             train max_epochs times regardless of error of the network. default = True

            learning_rate    The learning rate for the network training function. This is specific to
                             the network archetecture, dataset, and training algorithm. default = 0.1

            batch_splits     The number of roughly equal sized chunks to split the training data
                             into for each training batch inside of every epoch. default = 5

            hidden_layers    The shape of the hidden layer neurons for the network. Passing a
                             tuple consisting of multiple hidden layers is allowed. default = (10,) 
                             which is one hidden layer w/ 10 neurons.

        """
        self.max_epochs = 10000 
        self.continue_epochs = 25 
        self.try_convergence = True
        self.learning_rate = 0.1
        self.batch_splits = 5 
        self.hidden_layers = (10,)

