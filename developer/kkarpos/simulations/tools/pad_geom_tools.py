import reborn
from reborn.detector import RadialProfiler, tiled_pad_geometry_list
from reborn.detector import PADGeometry, PADGeometryList
from scipy.spatial.transform import Rotation
import numpy as np

from reborn.viewers.qtviews.padviews import PADView2
from reborn.viewers.mplviews import view_pad_data

class PADTools:



    def __init__(self,
                    pad_geometry,
                    beam,
                    detector_config: dict=None,
                    mask=None):

        # ensure PADGeometryList type
        self.pad_geometry = PADGeometryList(pad_geometry)

        if mask:
            self.mask = mask
        self.beam = beam


    def update_geom(self, 
                        detector_distance: float=None,
                        detector_offset: float=None,
                        detector_thickness: float=None,
                        detector_rotation: float=None,
                        binned_pixels: int=None,
                        beamstop_diameter: float=None):

        r"""Updates the detector PADGeometry config.
        Arguments 
                detector_distance (float, meters): distance from sample interaction to detector
                detector_offset (float, meters): detector offset from center, measured from the center
                detector_thickness (float, meters): thickness of the detector in
                detector_rotation (float, radians): detector rotation about the center pixel, measured in radians
                binned_pixels (int): Number of pixels binned
        """

        self.detector_config = {'detector_distance': 0,
                                    'detector_offset': 0,
                                    'detector_thickness': 0,
                                    'detector_rotation': 0,
                                    'binned_pixels': 1,
                                    'beamstop_diameter': 0}
        if detector_distance:
            self.detector_config['detector_distance'] = detector_distance
        if detector_offset:
            self.detector_config['detector_offset'] = detector_offset
        if detector_thickness:
            self.detector_config['detector_thickness'] = detector_thickness
        if detector_rotation:
            self.detector_config['detector_rotation'] = detector_rotation
        if binned_pixels:
            self.detector_config['binned_pixels'] = binned_pixels
            self._binned()

        self._update_geom()
        self.update_mask()

        if beamstop_diameter:
            self.detector_config['beamstop_diameter'] = beamstop_diameter
            self._mask_forward_scatter(self.detector_config['beamstop_diameter'])


    def _update_geom(self):
        r"""Does the PADGeometry updating.
        """
        theta = self.detector_config['detector_rotation']
        R = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
        for i in self.pad_geometry:
            i.fs_vec = np.dot(i.fs_vec, R.T)
            i.ss_vec = np.dot(i.ss_vec, R.T)
            i.t_vec = np.dot(i.t_vec, R.T)
            i.t_vec[0] += self.detector_config['detector_offset']
            i.t_vec[2] += self.detector_config['detector_distance']

    def _binned(self):
        self.pad_geometry = self.pad_geometry.binned(self.detector_config['binned_pixels'])


    def update_mask(self, mask=None):
        if mask is None:
            self.mask = self.pad_geometry.ones()
        else:
            self.mask = mask

    def display_mask(self, mask=None, viewer='pyqt'):
        r"""Display the mask

        Arguments:
            mask (ndarray): The flattened mask array. If None, will use the mask from this class. 
                            If a mask is passed, will display given mask.
            viewer (str): Which viewer you would like to use. Default is PADView2 (pyqt). 
                            Only two options allowed: 
                                1) PADView2 :'pyqt'
                                2) view_pad_data : 'mpl'. 

        """
        if mask is None:
            mask = self.mask

        if viewer == 'pyqt':
            PADView2(pad_geometry=self.pad_geometry, raw_data=mask)
        elif viewer == 'mpl':
            view_pad_data(pad_geometry=self.pad_geometry, pad_data=mask)
        else:
            raise ValueError("Incorrect choice for mask viewer. Current options are 'mpl' or 'pyqt'")

    def display_pads(self, mask=None, viewer='pyqt', raw_data=None):
        r"""Display the pads, meant to be a quick viewer

        Arguments:
            mask (ndarray): The flattened mask array. If None, will use the mask from this class. 
                            If a mask is passed, will display given mask.
            viewer (str): Which viewer you would like to use. Default is PADView2 (pyqt). 
                            Only two options allowed: 
                                1) PADView2 :'pyqt'
                                2) view_pad_data : 'mpl'. 

        """
        if mask is None:
            mask = self.mask
        pads = self.pad_geometry.copy()
        if raw_data is None:
            raw_data = [np.random.random(p.n_pixels)*p.polarization_factors(beam=self.beam) for p in pads]

        if viewer == 'pyqt':
            PADView2(pad_geometry=pads, raw_data=raw_data, mask_data=mask)
        elif viewer == 'mpl':
            print("mpl viewer does not accommodate masks.")
            view_pad_data(pad_geometry=pads, pad_data=raw_data)
        else:
            raise ValueError("Incorrect choice for mask viewer. Current options are 'mpl' or 'pyqt'")


    def _mask_forward_scatter(self, beamstop_diameter):
        theta = np.arctan(beamstop_diameter/self.detector_config['detector_distance'])
        self.mask[self.pad_geometry.scattering_angles(self.beam) < theta] = 0













