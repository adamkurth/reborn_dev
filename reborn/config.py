# This file is part of reborn <https://kirianlab.gitlab.io/reborn/>.
#
# reborn is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# reborn is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with reborn.  If not, see <https://www.gnu.org/licenses/>.

""" Standard configs for reborn.  Has global effect. """
import os
configs = dict()
# Default settings:
recognized_keys = ['debug', 'padview_debug', 'autocompile_fortran']
# Global debug setting
configs['debug'] = configs.get('debug', 0)
# PADView debug setting
configs['padview_debug'] = configs.get('padview_debug', 0)
# Automatically recompile fortran if source is changed
configs['autocompile_fortran'] = configs.get('autocompile_fortran', True)
# Now we override with custom configs if available
if os.path.exists(__file__.replace('config.py', 'custom_config.py')):
    try:
        from .custom_config import configs as custom
        print('**** you have a custom reborn configuration! ****')
        print(__file__.replace('config.py', 'custom_config.py'))
        for k in custom:
            configs[k] = custom[k]
    except ImportError:
        pass
if len(configs.keys()) > len(recognized_keys):
    raise ValueError('There are unrecognized keys in custom_config.py', configs.keys(), recognized_keys)
