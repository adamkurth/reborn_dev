# coding=utf-8
import spglib

file_name = 'spgrp.py'
f = open(file_name,'w+')
write = f.write

def write_translations(s):
    write('{')
    write("'translations': [\n")
    write(',\n'.join(['array([%5g,%5g,%5g])' % tuple(t) for t in s['translations']]))
    write('],\n')

def write_rotation(r):
    s = ('array([[%5g,%5g,%5g],\n'  % tuple(r[0,:]))
    s += ('       [%5g,%5g,%5g],\n' % tuple(r[1,:]))
    s += ('       [%5g,%5g,%5g]])'  % tuple(r[2,:]))
    return s

def write_rotations(s):
    write("'rotations':[\n")
    write(',\n'.join([write_rotation(r) for r in s['rotations']]))
    write(']}')
    
def write_entry(s):
    write_translations(s)
    write_rotations(s)

write('# coding=utf-8\n\n')

write('from numpy import array\n')
write(
"""
# This module is auto-generated from tools/spgrp-autogen/spgrp-autogen.py.
# Look here for more information:
#   http://pmsl.planet.sci.kobe-u.ac.jp/~seto/?page_id=37&lang=en
#
# This module defines the following lists:
#   _spgrp_ops: The list of dictionaries containing the symmetry operations
#               (translations and rotations).
#       _hmsym: The list of Hermann–Mauguin symbols (python strings)
#       _sgnum: The list of space group numbers (ranging from 1-230) corresponding to
#               each of the Hall numbers (ranging from 1-530).  I don't yet understand
#               what the Hall numbers are... but there is a useful table here:
#               http://pmsl.planet.sci.kobe-u.ac.jp/~seto/?page_id=37&lang=en
#
"""
)

write("""
# These are the Hermann-Mauguin symbols in string format
_hmsym = ['P 1','P -1','P 1 2 1','P 1 1 2','P 2 1 1','P 1 21 1','P 1 1 21','P 21 1 1','C 1 2 1','A 1 2 1','I 1 2 1','A 1 1 2','B 1 1 2','I 1 1 2','B 2 1 1','C 2 1 1','I 2 1 1','P 1 m 1','P 1 1 m','P m 1 1','P 1 c 1','P 1 n 1','P 1 a 1','P 1 1 a','P 1 1 n','P 1 1 b','P b 1 1','P n 1 1','P c 1 1','C 1 m 1','A 1 m 1','I 1 m 1','A 1 1 m','B 1 1 m','I 1 1 m','B m 1 1','C m 1 1','I m 1 1','C 1 c 1','A 1 n 1','I 1 a 1','A 1 a 1','C 1 n 1','I 1 c 1','A 1 1 a','B 1 1 n','I 1 1 b','B 1 1 b','A 1 1 n','I 1 1 a','B b 1 1','C n 1 1','I c 1 1','C c 1 1','B n 1 1','I b 1 1','P 1 2/m 1','P 1 1 2/m','P 2/m 1 1','P 1 21/m 1','P 1 1 21/m','P 21/m 1 1','C 1 2/m 1','A 1 2/m 1','I 1 2/m 1','A 1 1 2/m','B 1 1 2/m','I 1 1 2/m','B 2/m 1 1','C 2/m 1 1','I 2/m 1 1','P 1 2/c 1','P 1 2/n 1','P 1 2/a 1','P 1 1 2/a','P 1 1 2/n','P 1 1 2/b','P 2/b 1 1','P 2/n 1 1','P 2/c 1 1','P 1 21/c 1','P 1 21/n 1','P 1 21/a 1','P 1 1 21/a','P 1 1 21/n','P 1 1 21/b','P 21/b 1 1','P 21/n 1 1','P 21/c 1 1','C 1 2/c 1','A 1 2/n 1','I 1 2/a 1','A 1 2/a 1','C 1 2/n 1','I 1 2/c 1','A 1 1 2/a','B 1 1 2/n','I 1 1 2/b','B 1 1 2/b','A 1 1 2/n','I 1 1 2/a','B 2/b 1 1','C 2/n 1 1','I 2/c 1 1','C 2/c 1 1','B 2/n 1 1','I 2/b 1 1','P 2 2 2','P 2 2 21','P 21 2 2','P 2 21 2','P 21 21 2','P 2 21 21','P 21 2 21','P 21 21 21','C 2 2 21','A 21 2 2','B 2 21 2','C 2 2 2','A 2 2 2','B 2 2 2','F 2 2 2','I 2 2 2','I 21 21 21','P m m 2','P 2 m m','P m 2 m','P m c 21','P c m 21','P 21 m a','P 21 a m','P b 21 m','P m 21 b','P c c 2','P 2 a a','P b 2 b','P m a 2','P b m 2','P 2 m b','P 2 c m','P c 2 m','P m 2 a','P c a 21','P b c 21','P 21 a b','P 21 c a','P c 21 b','P b 21 a','P n c 2','P c n 2','P 2 n a','P 2 a n','P b 2 n','P n 2 b','P m n 21','P n m 21','P 21 m n','P 21 n m','P n 21 m','P m 21 n','P b a 2','P 2 c b','P c 2 a','P n a 21','P b n 21','P 21 n b','P 21 c n','P c 21 n','P n 21 a','P n n 2','P 2 n n','P n 2 n','C m m 2','A 2 m m','B m 2 m','C m c 21','C c m 21','A 21 m a','A 21 a m','B b 21 m','B m 21 b','C c c 2','A 2 a a','B b 2 b','A m m 2','B m m 2','B 2 m m','C 2 m m','C m 2 m','A m 2 m','A b m 2','B m a 2','B 2 c m','C 2 m b','C m 2 a','A c 2 m','A m a 2','B b m 2','B 2 m b','C 2 c m','C c 2 m','A m 2 a','A b a 2','B b a 2','B 2 c b','C 2 c b','C c 2 a','A c 2 a','F m m 2','F 2 m m','F m 2 m','F d d 2','F 2 d d','F d 2 d','I m m 2','I 2 m m','I m 2 m','I b a 2','I 2 c b','I c 2 a','I m a 2','I b m 2','I 2 m b','I 2 c m','I c 2 m','I m 2 a','P 2/m 2/m 2/m','P 2/n 2/n 2/n','P 2/n 2/n 2/n','P 2/c 2/c 2/m','P 2/m 2/a 2/a','P 2/b 2/m 2/b','P 2/b 2/a 2/n','P 2/b 2/a 2/n','P 2/n 2/c 2/b','P 2/n 2/c 2/b','P 2/c 2/n 2/a','P 2/c 2/n 2/a','P 21/m 2/m 2/a','P 2/m 21/m 2/b','P 2/b 21/m 2/m','P 2/c 2/m 21/m','P 2/m 2/c 21/m','P 21/m 2/a 2/m','P 2/n 21/n 2/a','P 21/n 2/n 2/b','P 2/b 2/n 21/n','P 2/c 21/n 2/n','P 21/n 2/c 2/n','P 2/n 2/a 21/n','P 2/m 2/n 21/a','P 2/n 2/m 21/b','P 21/b 2/m 2/n','P 21/c 2/n 2/m','P 2/n 21/c 2/m','P 2/m 21/a 2/n','P 21/c 2/c 2/a','P 2/c 21/c 2/b','P 2/b 21/a 2/a','P 2/c 2/a 21/a','P 2/b 2/c 21/b','P 21/b 2/a 2/b','P 21/b 21/a 2/m','P 2/m 21/c 21/b','P 21/c 2/m 21/a','P 21/c 21/c 2/n','P 2/n 21/a 21/a','P 21/b 2/n 21/b','P 2/b 21/c 21/m','P 21/c 2/a 21/m','P 21/m 2/c 21/a','P 21/m 21/a 2/b','P 21/b 21/m 2/a','P 2/c 21/m 21/b','P 21/n 21/n 2/m','P 2/m 21/n 21/n','P 21/n 2/m 21/n','P 21/m 21/m 2/n','P 21/m 21/m 2/n','P 2/n 21/m 21/m','P 2/n 21/m 21/m','P 21/m 2/n 21/m','P 21/m 2/n 21/m','P 21/b 2/c 21/n','P 2/c 21/a 21/n','P 21/n 21/c 2/a','P 21/n 2/a 21/b','P 2/b 21/n 21/a','P 21/c 21/n 2/b','P 21/b 21/c 21/a','P 21/c 21/a 21/b','P 21/n 21/m 21/a','P 21/m 21/n 21/b','P 21/b 21/n 21/m','P 21/c 21/m 21/n','P 21/m 21/c 21/n','P 21/n 21/a 21/m','C 2/m 2/c 21/m','C 2/c 2/m 21/m','A 21/m 2/m 2/a','A 21/m 2/a 2/m','B 2/b 21/m 2/m','B 2/m 21/m 2/b','C 2/m 2/c 21/a','C 2/c 2/m 21/b','A 21/b 2/m a','A 21/c 2/a 2/m','B 2/b 21/c 2/m','B 2/m 21/a 2/b','C 2/m 2/m 2/m','A 2/m 2/m 2/m','B 2/m 2/m 2/m','C 2/c 2/c 2/m','A 2/m 2/a 2/a','B 2/b 2/m 2/b','C 2/m 2/m 2/a','C 2/m 2/m 2/b','A 2/b 2/m 2/m','A 2/c 2/m 2/m','B 2/m 2/c 2/m','B 2/m 2/a 2/m','C 2/c 2/c 2/a','C 2/c 2/c 2/a','C 2/c 2/c 2/b','C 2/c 2/c 2/b','A 2/b 2/a 2/a','A 2/b 2/a 2/a','A 2/c 2/a 2/a','A 2/c 2/a 2/a','B 2/b 2/c 2/b','B 2/b 2/c 2/b','B 2/b 2/a 2/b','B 2/b 2/a 2/b','F 2/m 2/m 2/m','F 2/d 2/d 2/d','F 2/d 2/d 2/d','I 2/m 2/m 2/m','I 2/b 2/a 2/m','I 2/m 2/c 2/b','I 2/c 2/m 2/a','I 2/b 2/c 2/a','I 2/c 2/a 2/b','I 2/m 2/m 2/a','I 2/m 2/m 2/b','I 2/b 2/m 2/m','I 2/c 2/m 2/m','I 2/m 2/c 2/m','I 2/m 2/a 2/m','P 4','P 41','P 42','P 43','I 4','I 41','P -4','I -4','P 4/m','P 42/m','P 4/n','P 4/n','P 42/n','P 42/n','I 4/m','I 41/a','I 41/a','P 4 2 2','P 4 21 2','P 41 2 2','P 41 21 2','P 42 2 2','P 42 21 2','P 43 2 2','P 43 21 2','I 4 2 2','I 41 2 2','P 4 m m','P 4 b m','P 42 c m','P 42 n m','P 4 c c','P 4 n c','P 42 m c','P 42 b c','I 4 m m','I 4 c m','I 41 m d','I 41 c d','P -4 2 m','P -4 2 c','P -4 21 m','P -4 21 c','P -4 m 2','P -4 c 2','P -4 b 2','P -4 n 2','I -4 m 2','I -4 c 2','I -4 2 m','I -4 2 d','P 4/m 2/m 2/m','P 4/m 2/c 2/c','P 4/n 2/b 2/m','P 4/n 2/b 2/m','P 4/n 2/n 2/c','P 4/n 2/n 2/c','P 4/m 21/b m','P 4/m 21/n c','P 4/n 21/m m','P 4/n 21/m m','P 4/n 21/c c','P 4/n 21/c c','P 42/m 2/m 2/c','P 42/m 2/c 2/m','P 42/n 2/b 2/c','P 42/n 2/b 2/c','P 42/n 2/n 2/m','P 42/n 2/n 2/m','P 42/m 21/b 2/c','P 42/m 21/n 2/m','P 42/n 21/m 2/c','P 42/n 21/m 2/c','P 42/n 21/c 2/m','P 42/n 21/c 2/m','I 4/m 2/m 2/m','I 4/m 2/c 2/m','I 41/a 2/m 2/d','I 41/a 2/m 2/d','I 41/a 2/c 2/d','I 41/a 2/c 2/d','P 3','P 31','P 32','R 3','R 3','P -3','R -3','R -3','P 3 1 2','P 3 2 1','P 31 1 2','P 31 2 1','P 32 1 2','P 32 2 1','R 3 2','R 3 2','P 3 m 1','P 3 1 m','P 3 c 1','P 3 1 c','R 3 m','R 3 m','R 3 c','R 3 c','P -3 1 2/m','P -3 1 2/c','P -3 2/m 1','P -3 2/c 1','R -3 2/m','R -3 2/m','R -3 2/c','R -3 2/c','P 6','P 61','P 65','P 62','P 64','P 63','P -6','P 6/m','P 63/m','P 6 2 2','P 61 2 2','P 65 2 2','P 62 2 2','P 64 2 2','P 63 2 2','P 6 m m','P 6 c c','P 63 c m','P 63 m c','P -6 m 2','P -6 c 2','P -6 2 m','P -6 2 c','P 6/m 2/m 2/m','P 6/m 2/c 2/c','P 63/m 2/c 2/m','P 63/m 2/m 2/c','P 2 3','F 2 3','I 2 3','P 21 3','I 21 3','P 2/m -3','P 2/n -3','P 2/n -3','F 2/m -3','F 2/d -3','F 2/d -3','I 2/m -3','P 21/a -3','I 21/a -3','P 4 3 2','P 42 3 2','F 4 3 2','F 41 3 2','I 4 3 2','P 43 3 2','P 41 3 2','I 41 3 2','P -4 3 m','F -4 3 m','I -4 3 m','P -4 3 n','F -4 3 c','I -4 3 d','P 4/m -3 2/m','P 4/n -3 2/n','P 4/n -3 2/n','P 42/m -3 2/n','P 42/n -3 2/m','P 42/n -3 2/m','F 4/m -3 2/m','F 4/m -3 2/c','F 41/d -3 2/m','F 41/d -3 2/m','F 41/d -3 2/c','F 41/d -3 2/c','I 4/m -3 2/m','I 41/a -3 2/d']
""")

write("""
# These are the spacegroup numbers (1-230) corresponding to each Hall number (1-530)
_sgnum = [1,2,3,3,3,4,4,4,5,5,5,5,5,5,5,5,5,6,6,6,7,7,7,7,7,7,7,7,7,8,8,8,8,8,8,8,8,8,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,10,10,10,11,11,11,12,12,12,12,12,12,12,12,12,13,13,13,13,13,13,13,13,13,14,14,14,14,14,14,14,14,14,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,16,17,17,17,18,18,18,19,20,20,20,21,21,21,22,23,24,25,25,25,26,26,26,26,26,26,27,27,27,28,28,28,28,28,28,29,29,29,29,29,29,30,30,30,30,30,30,31,31,31,31,31,31,32,32,32,33,33,33,33,33,33,34,34,34,35,35,35,36,36,36,36,36,36,37,37,37,38,38,38,38,38,38,39,39,39,39,39,39,40,40,40,40,40,40,41,41,41,41,41,41,42,42,42,43,43,43,44,44,44,45,45,45,46,46,46,46,46,46,47,48,48,49,49,49,50,50,50,50,50,50,51,51,51,51,51,51,52,52,52,52,52,52,53,53,53,53,53,53,54,54,54,54,54,54,55,55,55,56,56,56,57,57,57,57,57,57,58,58,58,59,59,59,59,59,59,60,60,60,60,60,60,61,61,62,62,62,62,62,62,63,63,63,63,63,63,64,64,64,64,64,64,65,65,65,66,66,66,67,67,67,67,67,67,68,68,68,68,68,68,68,68,68,68,68,68,69,70,70,71,72,72,72,73,73,74,74,74,74,74,74,75,76,77,78,79,80,81,82,83,84,85,85,86,86,87,88,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,125,126,126,127,128,129,129,130,130,131,132,133,133,134,134,135,136,137,137,138,138,139,140,141,141,142,142,143,144,145,146,146,147,148,148,149,150,151,152,153,154,155,155,156,157,158,159,160,160,161,161,162,163,164,165,166,166,167,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,191,192,193,194,195,196,197,198,199,200,201,201,202,203,203,204,205,206,207,208,209,210,211,212,213,214,215,216,217,218,219,220,221,222,222,223,224,224,225,226,227,227,228,228,229,230]
""")

write("""
# These are the 530 dictionaries containing rotations and translations for each of the
# Hall numbers above.
""")
write('_spgrp_ops = [\n')
n = 530
for i in range(1,n+1):
    s = spglib.get_symmetry_from_database(i) 
    write('# Hall Number %3d\n' % i)
    write_entry(s)
    if i == n: break
    write(',\n')
write(']\n')

f.close()

print('Wrote file: ' + file_name)