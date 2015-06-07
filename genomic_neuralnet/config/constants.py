import joblib

TRAIT_NAME = 'FLOLD'
CYCLES = 4
TRAIN_SIZE = 0.9
REQUIRED_MARKERS_PROPORTION = 0.8 # 80% markers required to participate.
CPU_CORES = joblib.cpu_count() - 1 # Leave 1 free hyperthread.

# Neuralnet Settings
MAX_EPOCHS = 40 
CONTINUE_EPOCHS = 3
