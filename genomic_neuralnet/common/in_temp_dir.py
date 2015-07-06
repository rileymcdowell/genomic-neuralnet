from __future__ import print_function
from functools import wraps
import os
import shutil
from tempfile import mkdtemp

def in_temp_dir(func):
    """ Run a function inside of a temp directory w/ automatic cleanup """
    @wraps(func)
    def wrapped_func(*args, **kwargs):
        try:
            temp_dir = mkdtemp('.neuralnet')
            os.chdir(temp_dir)
            return func(*args, **kwargs)
        except Exception as e:
            import traceback
            traceback.print_exc()
        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
    return wrapped_func

