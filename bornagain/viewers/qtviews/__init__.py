from .qtviews import *
try:
    from .padviews import *
except ImportError:
    print('Import error on padviews -- probably because you do not have the pyopengl package.')
