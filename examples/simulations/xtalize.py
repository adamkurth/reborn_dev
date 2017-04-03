import bornagain.simulate.thorn as thorn
import numpy as np

class PhotoSystem1:

    def __init__(self, nPixels, wavelen, detdist, pixsize, n_unit, 
        spherical=False, inner_perc=.2, make_sims=True, make_latt=True):
        #   simple detector object, reads in scattering amplitudes and outputs intensity
        self.det = thorn.SimpleDetector(nPixels=nPixels, 
            wavelength=wavelen, 
            detectorDistance=detdist,
            pixelSize=pixsize)

        self.mol = thorn.Molecule(pdbFilePath='1jb0.pdb')

#       1d vectors of fractional coors, for use below
        x,y,z = self.mol.get_1d_frac_coords()

        mono1 = self.mol.transform( x, y, z ) # transforms are in fractional coors 
        mono2 = self.mol.transform( -y+1, x-y, z )
        mono3 = self.mol.transform( -x+y+1,-x+1, z )
        self.trimer1 = thorn.Atoms.aggregate([ mono1, mono2, mono3])

        mono4 = self.mol.transform( -x+2,-y+1,z+.5 )
        mono5 = self.mol.transform( x-y+1,x,z+.5 )
        mono6 = self.mol.transform( y+1,-x+y+1,z+.5 )
        self.trimer2 = thorn.Atoms.aggregate([ mono4, mono5, mono6])

        self.t1_com = self.trimer1.xyz.mean(0)
        self.t2_com = self.trimer2.xyz.mean(0)
        
        if make_sims:
            self.make_simulators()
        if make_latt:
            self.make_lattice(n_unit, spherical, inner_perc)



    def make_simulators(self):
        print("Making simulators for each trimer")
#       simulators
        self.Tmol1 = thorn.ThornAgain(self.det.q_vecs, self.trimer1.xyz-self.t1_com, self.trimer1.Z )
        self.Tmol2 = thorn.ThornAgain(self.det.q_vecs, self.trimer2.xyz-self.t2_com, self.trimer2.Z )


    def make_lattice(self, n_unit, spherical, inner_perc):
#       build the lattice, extend it n_units along each lattice vector 
        self.mol.lat.assemble(n_unit=n_unit, spherical=spherical) 

#       first lets make a big lattice
        self.lat1 = self.mol.lat.vecs + self.t1_com
        self.lat2 = self.mol.lat.vecs + self.t2_com
        self.dbl_lat = np.vstack( [self.lat1, self.lat2] )

#       these are the intices correspinding to 
#       each lattice
        n_lat_pts = self.mol.lat.vecs.shape[0]
        self.inds_lat1 = np.arange( n_lat_pts  )
        self.inds_lat2 = np.arange( n_lat_pts, 2*n_lat_pts  )

#       find the lattice center point
        lat_center = self.dbl_lat.mean(0)
#       and distance from center of each lattice point
        lat_rads = np.sqrt( np.sum( (self.dbl_lat - lat_center)**2,1))

#       maximum direction from center such that 
#       we can fit a sphere inside the lattice
        max_rad = min( self.dbl_lat.max(0))/2.

#       choose an inner sphere within which to jitter the
#       sphere center
        self.inner_rad = inner_perc*max_rad

#       boundary of sphere,  such that it does not exceed max_rad
        self.outer_rad = max_rad - self.inner_rad

    def make_hit(self, qmin, qmax, flux):

        self.Amol1 = self.Tmol1.run(rand_rot=True)
        self.Amol2 = self.Tmol2.run(force_rot_mat=self.Tmol1.rot_mat)

#       choose a random point within the inner sphere
        phi = np.random.uniform(0,2*np.pi)
        costheta = np.random.uniform(-1,1)
        u = random.random()

        theta = arccos( costheta )
        r = self.inner_rad * np.power( u, 1/3. )

        x = r * np.sin( theta) * np.cos( phi )
        y = r * np.sin( theta) * np.sin( phi )
        z = r * np.cos( theta )

        rand_sphere_cent = np.array( [x,y,z])

#       select all points within the newly defined sphere
        lat_rads_rand = np.sqrt( np.sum( (self.dbl_lat - rand_sphere_cent)**2,1))

        rand_lat1 = self.lat1[ (lat_rads_rand < self.outer_rad) [self.inds_lat1]]
        rand_lat2 = self.lat2[ (lat_rads_rand < self.outer_rad)[self.inds_lat2]]

#       simulate each lattice now
        Tlat1 = thorn.ThornAgain(self.det.q_vecs, rand_lat1 )
        Tlat2 = thorn.ThornAgain(self.det.q_vecs, rand_lat2 )
        self.Alat1 = Tlat1.run(force_rot_mat=self.Tmol1.rot_mat)
        self.Alat2 = Tlat2.run(force_rot_mat=self.Tmol1.rot_mat)

        img = self.det.readout_finite(self.Alat1*self.Amol1 + self.Alat2*self.Amol2, qmin=qmin, qmax=qmax, flux=flux)
        return img 

PS1 = PhotoSystem1( nPixels=2000, wavelen=6.53, detdist=0.2, pixsize=0.00005, 
    n_unit=150, make_sims=True, make_latt=True)

#img = PS1.make_hit( qmin=.02, qmax=2, flux=1e27)
