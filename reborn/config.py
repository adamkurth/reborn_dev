""" Standard configs for reborn.  Has global effect. """
import os
configs = dict()
if os.path.exists(__file__.replace('config.py', 'custom_config.py')):
    try:
        from custom_config import configs
    except ImportError:
        print('You do not have a proper custom_config module')
# Default settings:
recognized_keys = ['debug', 'padview_debug', 'autocompile_fortran']
# Global debug setting
configs['debug'] = configs.get('debug', 0)
# PADView debug setting
configs['padview_debug'] = configs.get('padview_debug', 0)
# Automatically recompile fortran if source is changed
configs['autocompile_fortran'] = configs.get('autocompile_fortran', True)
if len(configs.keys()) > len(recognized_keys):
    raise ValueError('There are unrecognized keys in custom_config.py')
