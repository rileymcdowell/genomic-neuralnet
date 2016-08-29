import joblib

# Set the number of calls a marker must have to participate in analysis.
REQUIRED_MARKER_CALL_PROPORTION = 0.8
# Set the number of markers a sample must have to participate in analysis.
REQUIRED_MARKERS_PER_SAMPLE_PROP = 0.5

# Set the number of cores to use for processing. 
CPU_CORES = joblib.cpu_count()
# The number of folds to use for cross-validation.
NUM_FOLDS = 5

# Pick a processing backend for the modeling.
CELERY_BACKEND = 'celery'
JOBLIB_BACKEND = 'joblib'
SINGLE_CORE_BACKEND = 'single-core'

MAX_EPOCHS = 1000
CONTINUE_EPOCHS = 1000
USE_ARAC = False
TRY_CONVERGENCE = False

INIT_CELERY = False # Must be True to use celery backend.
