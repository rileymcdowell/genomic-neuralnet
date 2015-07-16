from functools import wraps

class Namespace:
    pass

class RequiredValueNotTrueException(Exception):
    """ 
    We need this hacky namespace class for python2.7. 
    pep-3104 (nonlocal keyword) solved this in python 3.x
    """
    pass

def require_true(boolean_expression, false_message=None, *args, **kwargs):
    ns = Namespace()
    ns.false_message = false_message
    def check_name(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            #nonlocal false_message <- python3.x only.
            self = args[0]
            if boolean_expression(self):
                return func(*args, **kwargs)
            else:
                if ns.false_message == None:
                    ns.false_message = 'Required attribute was not defined'
                raise RequiredValueNotTrueException(ns.false_message)
        return wrapper
    return check_name
