from __future__ import print_function

class NeuralnetConfig(object):
    def __init__(self): 
        """ 
        A container for neural network training options.

        Attributes:
            max_epochs               The maximum number of epochs to train under any circumstance.
                                     If try_convergence == False, this is also the minimum number of epochs
                                     to train. default = 10000

            continue_epochs          Number of epochs to continue training when trying for a lower error score. 
                                     This option is ignored if try_convergence is False. default = 25

            try_convergence          Try to reach a point where the error of the network is no longer decreasing,
                                     suggesting that training has converged to the minimum error point. If False,
                                     train max_epochs times regardless of error of the network. default = True

            initial_learning_rate    The learning rate for the network training function. This is specific to
                                     the network archetecture, dataset, and training algorithm. default = 0.01

            active_learning_rate     Reward reduced errors with faster learning rates and increased error with lower learning
                                     rates. Default = True.

            learning_rate_multiplier The multiplier for the learning rate if the last epoch reduced total network error.
                                     default = 1.05

            learning_rate_divisor    The divisor for the learning rate if the last epoch increased total network error.
                                     default = 2.0

            batch_splits             The number of roughly equal sized chunks to split the training data
                                     into for each training batch inside of every epoch. default = 5

            hidden_layers            The shape of the hidden layer neurons for the network. Passing a
                                     tuple consisting of multiple hidden layers is allowed. default = (10,) 
                                     which is one hidden layer w/ 10 neurons.

            report_every             Number of epochs between printing status reports.

        """
        self.max_epochs = 10000 
        self.continue_epochs = 25 
        self.try_convergence = True
        self.active_learning_rate = True
        self.initial_learning_rate = 0.01
        self.learning_rate_multiplier = 1.05
        self.learning_rate_divisor = 2.0
        self.batch_splits = 5 
        self.hidden_layers = (10,)
        self.report_every = 100

