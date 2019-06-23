# Define a phony jit decorator that does nothing.  Beware of very slow execution...
def jit(*args1, **kwargs1):
    def real_decorator(function):
        def wrapper():
            function(*args, **kwargs)
        return wrapper
    return real_decorator