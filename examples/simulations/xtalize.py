import h5py
import numpy as np

from bornagain.simulate import ThornAgain
from bornagain import SimpleDetector
from bornagain.target import Atoms, Molecule, Lattice


class PhotoSystem1:

    def __init__(self, n_pixels, wavelen, detdist, pixsize, n_unit,
                 spherical=False, inner_perc=.2, make_sims=True, make_latt=True):

        """
        Parameters
        ==========
        n_pixels, float
            the number of pixels along one dimension of the Simple detector

        wavelen, float
            photon wavelength in ANGSTROM

        detdist, float
            distrance from interaction region to the detector, assuming perpen-
            dicular detector plane and incident beam direction, in METERS

        pixsize, float
            edge length of square pixel in METERS

        n_unit, int
            number of unit cells in the xtal along each xtal vector a,b,c

        spherical, bool
            whether or not a spherical boundary is applied to the xtal

        inner_perc, float
            parameter for randomizing the center of the spherical boundary in
            order to simulate randomized lattice boundary effects

        make_sims, bool
            whether to make the ThornAgain simulators

        make_latt, bool
            whether to make the lattices
        """
        # simple detector object, reads in scattering amplitudes and outputs
        # intensity
        self.det = SimpleDetector(n_pixels=n_pixels,
                                        wavelen=wavelen,
                                        detdist=detdist,
                                        pixsize=pixsize)

        self.mol = Molecule(pdbFilePath='../data/pdb/1jb0.pdb')

#       1d vectors of fractional coors, for use below
        x, y, z = self.mol.get_1d_frac_coords()

        # transforms are in fractional coors
        mono1 = self.mol.transform(x, y, z)
        mono2 = self.mol.transform(-y + 1, x - y, z)
        mono3 = self.mol.transform(-x + y + 1, -x + 1, z)
        self.trimer1 = Atoms.aggregate([mono1, mono2, mono3])

        mono4 = self.mol.transform(-x + 2, -y + 1, z + .5)
        mono5 = self.mol.transform(x - y + 1, x, z + .5)
        mono6 = self.mol.transform(y + 1, -x + y + 1, z + .5)
        self.trimer2 = Atoms.aggregate([mono4, mono5, mono6])

        self.t1_com = self.trimer1.xyz.mean(0)
        self.t2_com = self.trimer2.xyz.mean(0)

        if make_sims:
            self.make_simulators()
        if make_latt:
            self.make_lattice(n_unit, spherical, inner_perc)

    def make_simulators(self):
        print("Making simulators for each trimer")
#       simulators
        self.Tmol1 = ThornAgain(
            self.det.Q,
            (self.trimer1.xyz -
            self.t1_com)[:],
            self.trimer1.Z[:])
        self.Tmol2 = ThornAgain(
            self.det.Q,
            (self.trimer2.xyz -
            self.t2_com)[:],
            self.trimer2.Z[:])

    def make_lattice(self, n_unit, spherical, inner_perc):
        #       build the lattice, extend it n_units along each lattice vector
        self.mol.lat.assemble(n_unit=n_unit, spherical=spherical)

#       first lets make a big lattice
        self.lat1 = self.mol.lat.vecs + self.t1_com
        self.lat2 = self.mol.lat.vecs + self.t2_com
        self.dbl_lat = np.vstack([self.lat1, self.lat2])

#       these are the indices corresponding to
#       each lattice
        n_lat_pts = self.mol.lat.vecs.shape[0]
        self.inds_lat1 = np.arange(n_lat_pts)
        self.inds_lat2 = np.arange(n_lat_pts, 2 * n_lat_pts)

#       find the lattice center point
        lat_center = self.dbl_lat.mean(0)
#       and distance from center of each lattice point
        lat_rads = np.sqrt(np.sum((self.dbl_lat - lat_center)**2, 1))

#       maximum direction from center such that
#       we can fit a sphere inside the lattice
        max_rad = min(self.dbl_lat.max(0)) / 2.

#       choose an inner sphere within which to jitter the
#       sphere center
        self.inner_rad = inner_perc * max_rad

#       boundary of sphere,  such that it does not exceed max_rad
        self.outer_rad = max_rad - self.inner_rad

    def make_hit(self, finite=True, qmin=None, qmax=None, flux=None, rand_rot=True):

        if finite:
            assert( qmin is not None)
            assert( qmax is not None)
            assert( flux is not None)

        self.Amol1 = self.Tmol1.run(rand_rot=rand_rot)
        self.Amol2 = self.Tmol2.run(force_rot_mat=self.Tmol1.rot_mat)

#       choose a random point within the inner sphere
        phi = np.random.uniform(0, 2 * np.pi)
        costheta = np.random.uniform(-1, 1)
        u = np.random.random()

        theta = np.arccos(costheta)
        r = self.inner_rad * np.power(u, 1 / 3.)

        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)

        rand_sphere_cent = np.array([x, y, z])

#       select all points within the newly defined sphere
        lat_rads_rand = np.sqrt(
            np.sum((self.dbl_lat - rand_sphere_cent)**2, 1))

        self.rand_lat1 = self.lat1[(lat_rads_rand < self.outer_rad)[
            self.inds_lat1]]
        self.rand_lat2 = self.lat2[(lat_rads_rand < self.outer_rad)[
            self.inds_lat2]]

#       simulate each lattice now
        Tlat1 = ThornAgain(self.det.q_vecs, self.rand_lat1)
        Tlat2 = ThornAgain(self.det.q_vecs, self.rand_lat2)
        self.Alat1 = Tlat1.run(force_rot_mat=self.Tmol1.rot_mat)
        self.Alat2 = Tlat2.run(force_rot_mat=self.Tmol1.rot_mat)

        if finite:
            img = self.det.readout_finite(
                self.Alat1 *
                self.Amol1 +
                self.Alat2 *
                self.Amol2,
                qmin=qmin,
                qmax=qmax,
                flux=flux)
        else: 
            img = self.det.readout(
                self.Alat1 *
                self.Amol1 +
                self.Alat2 *
                self.Amol2)
        return img

if __name__ == "__main__":
    PS1 = PhotoSystem1(n_pixels=2000, wavelen=6.53, detdist=0.2, pixsize=0.00007,
                       n_unit=150, make_sims=True, make_latt=True)
    
    many = False
    if many:


        f = h5py.File('/home/dermen/some_PS1_hits.hdf5', 'w')
        for i in xrange(1000):
            img = PS1.make_hit( qmin=.02, qmax=4.2, flux=1e27)
            f.create_dataset('hit%d/img'%i, data=img.astype(np.float32))
            f.create_dataset('hit%d/lat1'%i, data=PS1.rand_lat1.astype(np.float32))
            f.create_dataset('hit%d/lat2'%i, data=PS1.rand_lat2.astype(np.float32))
            f.create_dataset('hit%d/rot'%i, data=PS1.Tmol1.rot_mat)
        f.close()

    verify = False
# verify

    if verify:
        from thor.scatter import simulate_atomic
        from thor.structure import load_coor 

        p = PS1
        xyz1 = np.vstack( [p.Tmol1.atom_vecs[:,:3] + L for L in p.rand_lat1])
        xyz2 = np.vstack( [p.Tmol2.atom_vecs[:,:3] + L for L in p.rand_lat2])

        Z1 = np.hstack( [p.Tmol1.atomic_nums for _ in  p.rand_lat1])
        Z2 = np.hstack( [p.Tmol2.atomic_nums for _ in  p.rand_lat2])

        xyzZ1 = concatenate((xyz1,Z1[:,None]), axis=1)
        xyzZ2 = concatenate((xyz2,Z2[:,None]), axis=1)

        all_coors= np.vstack( (xyzZ1, xyzZ2) )

        xyz = all_coors[:,:3]
        rot_mat = p.Tmol1.rot_mat.reshape( (3,3))
        xyzR = vstack( [dot ( rot_mat, c ) for c in xyz] )

        all_coors[:,:3] = xyzR

        np.savetxt('test.coor', all_coors, fmt='%.4f')


        t1 = load_coor( 'test.coor')
        qs = p.Tmol1.q_vecs[:,:3]

        amps = simulate_atomic(t1,1,qs)
        img2 = (abs(amps)**2).reshape( p.det.img_sh)

        from mpl_toolkits.axes_grid1 import make_axes_locatable

        fig = figure(1)
        subplot(131)
        ax1 = gca()
        ax1_img = imshow( img, norm=mpl.colors.LogNorm())
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(ax1_img, cax)
        cbar.ax.set_ylabel('photon counts', rotation=270, labelpad=15)
        ax1.set_title('colvolved simulation')
        ax1.axis('off')

        subplot(132)
        ax2 = gca()
        ax2_img = imshow( img2, norm=mpl.colors.LogNorm())
        divider = make_axes_locatable(ax2)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(ax2_img, cax)
        cbar.ax.set_ylabel('photon counts', rotation=270, labelpad=15)
        ax2.set_title('all-atom simulation')
        ax2.axis('off')


        subplot(133)
        ax3 = gca()
        ax3_img = imshow( (img-img2)/(.5*img+.5*img2) , norm=mpl.colors.SymLogNorm(0.00001))
        divider = make_axes_locatable(ax3)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(ax3_img, cax)
        cbar.ax.set_ylabel(r'$\Delta\,$ (counts) / Mean(counts)', rotation=270, labelpad=15)
        ax3.set_title('difference of left two images')
        ax3.axis('off')

