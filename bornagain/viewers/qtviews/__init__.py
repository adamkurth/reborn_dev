try:
    from .qtviews import *
except ImportError:
    print('ImportError on qtviews -- probably because you do not have the pyopengl package.')
try:
    from .padviews import *
except ImportError:
    print('ImportError on padviews -- probably because you do not have the pyopengl package.')
