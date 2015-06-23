import joblib

CYCLES = 32 
TRAIN_SIZE = 0.9
# Limit the number of markers required to participate in analysis.
REQUIRED_MARKERS_PROPORTION = 0.0 
# Set the number of cores to use for processing. 
CPU_CORES = joblib.cpu_count()

# Neuralnet Settings
MAX_EPOCHS = 200 # If TRY_CONVERGENCE is False, this is also the minimum # epochs.
CONTINUE_EPOCHS = 10 # Ignored if TRY_CONVERGENCE is False
TRY_CONVERGENCE = False
USE_ARAC = False # If arac library is installed, we can train networks faster.
