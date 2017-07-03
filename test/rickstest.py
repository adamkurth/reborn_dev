import sys
sys.path.append('..')
from bornagain.simulate import clcore

clcore.helpme()
core = clcore.ClCore(context=None, queue=None, group_size=1,double_precision=False)
print('single ok')
core = clcore.ClCore(context=None, queue=None, group_size=1,double_precision=True)
print('double ok')
