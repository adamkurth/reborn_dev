#Not tested at all!!!!!!

import numpy as np

class rotationCorrelation(object):

   def __init__(self,L):
      self.L = L
      self.dlist = []
      lnint = np.log(np.arange(1,2*L+1))
      facln = np.zeros(2*L+1,dtype=np.float64)
      for i in range(2*L):
         facln[i+1] = facln[i]+lnint[i]
      for l in range(L+1):
         d = np.zeros([2*l+1,2*l+1],dtype=np.float64)
         for m in range(-l,l+1):
            for n in range(-l,l+1):
               expf = 0.5*(facln[l+m]+facln[l-m]+facln[l+n]+facln[l-n])\
                  -facln[2]*l
               kmin = np.max([0,n-m])
               kmax = np.min([l+n,l-m])
               if ((kmin+m-n) % 2 == 1):
                  fac = -1.0
               else:
                  fac = 1.0
               for k in range(kmin,kmax+1):
                  d[m,n] += fac*np.exp(expf-facln[l+n-k]-facln[k]
                     -facln[m-n+k]-facln[l-m-k])
                  fac = -fac
         self.dlist.append(d)
   
   def correlation(Ilmm):
      t = np.zeros([2*self.L+1,2*self.L+1,2*self.L+1],dtype=np.complex128)
      for l in range(L+1):
         d = self.dlist[l]
         for m in range(-l,l+1):
            for h in range(-l,l+1):
               for mp in range(-l,l+1):
                  t[m,h,:] += d[m,h]*d[h,:]*Ilmm[l,m,:]
      return np.fft.fftn(t)




