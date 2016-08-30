
# Wait up to 15 minutes for each iteration.
BROKER_TRANSPORT_OPTIONS = {'visibility_timeout': 900}
# Do not pre-fetch work.
CELERYD_PREFETCH_MULTIPLIER = 1 
# Do not ack messages until work is completed.
CELERY_ACKS_LATE = True

