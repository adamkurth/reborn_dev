import numpy as np
from scipy.special import lpmn

class ylmIntegration(object):
   """
   Uniform spherical coordinate angular grid and integration weights
   Input:
      L = maximum L for Y^*_lm Y_lm to be correctly integrated
   Calculates:
      dtheta the angular spacing of grid points
      theta a 2*L+2 array of equally spaced midpoint theta values
      phi a 2*L+2 array of equally spaced, with first point 0, phi values
      w a 2*L+2 array of weights for the theta points, includes the dphi
   """

   def __init__(self,L):
      self.dtheta = np.pi/(2*L+2)
      self.theta = (np.arange(2*L+2)+0.5)*self.dtheta
      self.phi = np.arange(2*L+2)*2.0*self.dtheta
      self.w = np.zeros(2*L+2,dtype=np.float64)
      self.L = L
      for j in range(2*L+2):
         for i in range(L+1):
            self.w[j] += 4*np.pi/(2*(L+1)**2)*np.sin(((2*j+1)*np.pi)/(4*(L+1)))\
                  *np.sin(((2*j+1)*(2*i+1)*np.pi)/(4*(L+1)))/(2*i+1)
      # calculate Y_lm(theta_j,phi=0) m nonnegative using numpy P_lm
      pml = np.zeros([2*L+2,L+1,L+1],dtype=np.float64)
      self.ylm = np.zeros([2*L+2,L+1,L+1],dtype=np.float64)
      lfac = np.zeros(2*L+1,dtype=np.float64)
      for i in range(1,2*L+1):
         lfac[i] = lfac[i-1]+np.log(i)
      for j in range(2*L+2):
         pml[j,:,:], _ = lpmn(L,L,np.cos(self.theta[j]))
      for l in range(L+1):
         for m in range(L+1):
            fac = np.sqrt((2*l+1)/(4.0*np.pi)*np.exp(lfac[l-m]-lfac[l+m]))
            self.ylm[:,l,m] = fac*pml[:,m,l]

   def ylmcoef(self,f):
      """
         Calculates expansion coefficients for ylm of function f
         f[theta,phi] must be an numpy array of real or complex values at 
         the self.theta, self.phi values. The coefficients are returned
         in an numpy array ylmc[l,m] of shape L+1,2*L+1, with unused m
         values = 0.  Python wrap around is used so negative m values give
         correct coefficient.
      """
      ylmc = np.zeros([self.L+1,2*self.L+1],dtype=np.complex128)
      ft = np.fft.fft(f)
      for l in range(self.L+1):
         ylmc[l,0] = np.sum(self.ylm[:,l,0]*self.w*ft[:,0])
         fac = -1.0
         for m in range(1,l+1):
            ylmc[l,m] = np.sum(self.ylm[:,l,m]*self.w*ft[:,m])
            ylmc[l,-m] = fac*np.sum(self.ylm[:,l,m]*self.w*ft[:,-m])
            fac = -fac
      return ylmc
