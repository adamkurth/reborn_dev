""" Standard configs for reborn.  Has global effect. """
import os
if os.path.exists(__file__.replace('config.py', 'custom_config.py')):
    try:
        from custom_config import configs
    except ImportError:
        print('You do not have a proper custom_config module')
else:
    configs = dict()
# Default settings:
# Global debug setting
configs['debug'] = configs.get('debug', 0)
# PADView debug setting
configs['padview_debug'] = configs.get('padview_debug', 0)
# Automatically recompile fortran if source is changed
configs['autocompile_fortran'] = configs.get('autorecompile_fortran', True)
