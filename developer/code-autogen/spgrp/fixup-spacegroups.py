import pandas as pd
import string
import sys

write = sys.stdout.write

seto = pd.read_csv('seto.csv')

hms = seto['Full notation']

for hm_orig in hms:

    hm = [c for c in hm_orig]
    hm_fix = []
    printme = False
    for c in hm:
        if c not in string.printable:
            printme = True
            hm_fix.append(' ')
        else:
            hm_fix.append(c)
    hm_fix = ''.join(hm_fix)
    write("'"+hm_fix+"',")
#     if printme:
#         print(hm_orig)
#         print(''.join(hm_fix)) 
#         print('')   
        
