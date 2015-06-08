import joblib

TRAIT_NAME = 'FLOSD'
CYCLES = 50 
TRAIN_SIZE = 0.9
# Limit the number of markers required to participate in analysis.
REQUIRED_MARKERS_PROPORTION = 0.0 
# Set the number of cores to use for processing. 
# Defaults to all _logical_ cores.
CPU_CORES = joblib.cpu_count() 

# Neuralnet Settings
MAX_EPOCHS = 500
CONTINUE_EPOCHS = 50 
