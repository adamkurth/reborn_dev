import numpy as np
from simulate_droplets import DropletGetter
from reborn.analysis.optimize import sphere_form_factor
from reborn import detector
from reborn.viewers.qtviews.padviews import PADView
import simulate_droplets
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar,minimize
droplet_radius =1.6e-8
radius_range = np.arange(1,20)*1e-8


def create_droplet_getter(droplet_radius=5e-8):
    simulate_droplets.drop_radius = droplet_radius
    dg = DropletGetter()
    return dg

dg = create_droplet_getter()

#####################################
## Droplet radius estimation logic ##
class Peak:
    '''
    Peak class usid in persistent homology peak finding.
    DOI: 10.1007/978-3-658-32182-6_13
    https://www.sthu.org/research/publications/files/Hub20.pdf
    https://www.sthu.org/blog/13-perstopology-peakdetection/index.html
    '''
    def __init__(self, startidx):
        self.born = self.left = self.right = startidx
        self.died = None

    def get_persistence(self, seq):
        return float("inf") if self.died is None else seq[self.born] - seq[self.died]

def get_persistent_homology(seq):
    '''
    Persistent Homology Peak finding:
    DOI: 10.1007/978-3-658-32182-6_13
    https://www.sthu.org/research/publications/files/Hub20.pdf
    https://www.sthu.org/blog/13-perstopology-peakdetection/index.html
    '''
    peaks = []
    # Maps indices to peaks
    idxtopeak = [None for s in seq]
    # Sequence indices sorted by values
    indices = range(len(seq))
    indices = sorted(indices, key = lambda i: seq[i], reverse=True)

    # Process each sample in descending order
    for idx in indices:
        lftdone = (idx > 0 and idxtopeak[idx-1] is not None)
        rgtdone = (idx < len(seq)-1 and idxtopeak[idx+1] is not None)
        il = idxtopeak[idx-1] if lftdone else None
        ir = idxtopeak[idx+1] if rgtdone else None

        # New peak born
        if not lftdone and not rgtdone:
            peaks.append(Peak(idx))
            idxtopeak[idx] = len(peaks)-1

        # Directly merge to next peak left
        if lftdone and not rgtdone:
            peaks[il].right += 1
            idxtopeak[idx] = il

        # Directly merge to next peak right
        if not lftdone and rgtdone:
            peaks[ir].left -= 1
            idxtopeak[idx] = ir

        # Merge left and right peaks
        if lftdone and rgtdone:
            # Left was born earlier: merge right to left
            if seq[peaks[il].born] > seq[peaks[ir].born]:
                peaks[ir].died = idx
                peaks[il].right = peaks[ir].right
                idxtopeak[peaks[il].right] = idxtopeak[idx] = il
            else:
                peaks[il].died = idx
                peaks[ir].left = peaks[il].left
                idxtopeak[peaks[ir].left] = idxtopeak[idx] = ir

    # This is optional convenience
    return sorted(peaks, key=lambda p: p.get_persistence(seq), reverse=True)

def find_minima(ID,persistence_limit=1,boundary_size=0):
    '''
    Finds minima in a SAXS pattern ID using persistent homology,
    :param persistence_limit: Defines the minimum persistence ('height') a peak needs to have in order to be considered. 
    :param boundary_size: Value between 0 and 0.5 (0%-50%) that specifies how much of the data is omitted at the begining and end of the data sequence.
    '''
    n_values=len(ID)
    peaks = get_persistent_homology(-1*np.log(ID))
    peak_ids = np.array([p.born for p in peaks])
    arg_ids = np.argsort(peak_ids)
    peak_ids = peak_ids[arg_ids]
    peak_persistence = np.array([p.get_persistence(-1*np.log(ID)) for p in peaks])
    peak_persistence=peak_persistence[arg_ids]
    persistence_mask = (peak_persistence>= persistence_limit) & (peak_persistence<np.inf)
    print('boundaries =',[boundary_size*n_values,(1-boundary_size)*n_values],'boundary_size = ', boundary_size,'n_values = ',n_values) 
    boundary_mask = (peak_ids > boundary_size*n_values) & (peak_ids < (1-boundary_size)*n_values)
    mask = persistence_mask & boundary_mask
    print('minima persistences = {}'.format(peak_persistence[mask]))
    return peak_ids[mask]


class SphericalDroplets:
    def __init__(self, q=None):
        if q is None:
            q = np.linspace(0,1e10,517)
        self.q = q.copy()

        #The following are zeros of f(x)=sin(x)-x*cos(x), they corresponds to minima in the spherical form factor.
        self.standard_I_R_minima = [4.49340945790906,
                                    7.725251836937707,
                                    10.90412165942889,
                                    14.06619391283147,
                                    17.22075527193077,
                                    20.37130295928756,
                                    23.51945249868901,
                                    26.66605425881267,
                                    29.81159879089296,
                                    32.95638903982248]

        # minima distances are monotonically decreasing and converging to pi for large x,since (sin(x)-x*cos(x)-> -x*cos(x) for x>>1). The monotonical decreasing bit is just a numerical guess right now but seems to be true.
        self.standard_max_minima_distance = np.diff(self.standard_I_R_minima)[0]
        
    def estimate_droplet_radius_by_minima(self,ID,min_args,minima_start=0):
        '''
        estimates droplet size by SAXS minima positions
        :param ID: SAXS data
        :param min_args: minima indices of ID
        :param minima_start: Specifies to which formfactor minima the first minima in ID corresponds to. This excludes the formfactor minima at q=0 so minima_start=0 actually denotes the first minima in the formfactor.
        '''
        estimated_r,std_r=np.nan,np.nan
        if len(min_args)<1:
            print('Found_minima at = {}'.format(min_args))
            raise AssertionError('SAXS profile does not resolve the first minima of the formfactor, no size determination possible.')            
        n_used_minima = min(len(self.standard_I_R_minima[minima_start:]),len(min_args))
        standard_minima=self.standard_I_R_minima[minima_start:][:n_used_minima]
        estimated_rs = standard_minima/self.q[min_args[:n_used_minima]]
        estimated_r = np.mean(estimated_rs)
        if len(estimated_rs)>1:
            std_r = np.std(estimated_rs)
        confidence_interval = ((standard_minima[0]-self.standard_max_minima_distance/2)/self.q[min_args[0]],(standard_minima[0]+self.standard_max_minima_distance/2)/self.q[min_args[0]])
        return estimated_r,std_r,confidence_interval

    def fit_scaling_parameter(self,ID,IR,A_range):
        '''Simple wrapper around scipy.optimize.minimize_scalar
        :param ID: SAXS data
        :param IR: spherical_formfactor
        :param A_range: Is a tuple of two floats (start,stop), that describe the scaling factor range in which to optimize.'''
        def diff(a):
            diff = (ID-a*IR)**2
            return np.mean(diff)
            
        A = minimize_scalar(diff,method='bounded',bounds = A_range).x
        return A
        
    def fit_profile(self, I_D, minima_start=0,minima_persistence_limit=1,q_boundary_percent=0.05,A_fit_order_range=5):
        '''
        Fits a spherical formfactor to a given SAXS profile based on its minima positions.
        If no minima can be detected in the SAXS profile it throws an AssertionError.
        :param I_D: SAXS data
        :param minima_start: Specifies to which formfactor minima the first minima in ID corresponds to. This excludes the formfactor minima at q=0 so minima_start=0 actually denotes the first minima in the formfactor.
        :param minima_persistence_limit: Defines the minimum persistence ('height') a peak needs to have in order to be considered.        
        :param boundary_size: Value between 0 and 0.5 (0%-50%) that specifies how much of the data is omitted at the begining and end of the data sequence during minima determination.
        :param A_fit_order_range: Defines the fit are of the Scaling factor. E.g. a value of 2 means that the search range is (initial_guess*1e2 , initial_guess*1e-2)
        '''
        min_args=find_minima(ID,persistence_limit = minima_persistence_limit,boundary_size=q_boundary_percent)
        print('min args = {}'.format(min_args))
        radius,radius_std,radius_confidence_interval = self.estimate_droplet_radius_by_minima(I_D,min_args,minima_start=minima_start)
        unscaled_intensity =  (sphere_form_factor(radius=radius, q_mags=self.q, check_divide_by_zero=True))**2
        if len(min_args)<2:
            start=min_args[0]-min_args[0]//2
            stop=min_args[0]+min_args[0]//2
        else:
            start = min_args[0]-(min_args[1]-min_args[0])//2
            stop = min_args[1]
        q_slice = slice(start,stop)
        A_order_guess =  np.floor(np.log10(np.sum(I_D[q_slice])/np.sum(unscaled_intensity[q_slice])))
        A_range = (10**(A_order_guess-A_fit_order_range),10**(A_order_guess+A_fit_order_range))
        A = self.fit_scaling_parameter(I_D[q_slice],unscaled_intensity[q_slice],A_range)
        
        return radius,{'A':A,'radius_std':radius_std,'radius_confidence_interval':radius_confidence_interval,'q_slice':q_slice,'IR_min':unscaled_intensity}


###########
## Tests ##
if __name__=='__main__':

    # using first frame to initialize RadialProfiler and SphericalDroplets class
    first_frame = dg.get_frame()
    epix_pad_shape = first_frame.get_pad_geometry()[0].shape()
    epix_slice = slice(0,4*np.prod(epix_pad_shape))
    epix_geom = first_frame.get_pad_geometry()[:4]
    max_q = np.max(first_frame.q_mags[epix_slice])
    profiler = detector.RadialProfiler(beam=first_frame.get_beam(), pad_geometry=first_frame.get_pad_geometry()[:4], n_bins=100, q_range=np.array([0, max_q]))
    qs = profiler.q_bin_centers
    sp_droplet = SphericalDroplets(q = qs)







    if False:
        view = PADView(data = np.split(average,4),pad_geometry = epix_geom)
    
    # calculate SAXS pattern (takes about 5 mins)
    if True:
        start = 1.5e-8
        stop = 1.5e-7
        n_steps=113
        step = (stop-start)/n_steps
        rs = np.arange(start,stop,step)
        estimated_r = np.zeros(n_steps,dtype=float)
        r_dicts=[]
        for ir,r in enumerate(rs):
            dg = create_droplet_getter(droplet_radius=r)
            n_frames = 10
            average = np.zeros(epix_slice.stop,dtype=float)
            processed_frames = 0
            for i in range(n_frames):
                data = dg.get_next_frame().processed_data[epix_slice]
                average[:] = (data.astype(float)+processed_frames*average.astype(float))/(processed_frames+1) 
                processed_frames+=1
                #ID = profiler.quickstats(average)['sum']
            #size_list = sp_droplet.fit_profile(ID)
            #estimated_sizes.append(size_list[0])
            ID = profiler.quickstats(average)['sum']    
            try:
                r,r_dict = sp_droplet.fit_profile(ID,minima_persistence_limit=1,minima_start=0,q_boundary_percent=0.05)
                r_dicts.append(r_dict)
                estimated_r[ir]=r
            except AssertionError as e:
                print(e)    
    if True:
        rs_p = rs*1e9
        e_rs_p = estimated_r*1e9
        rel_diff=np.abs(e_rs_p-rs_p)/rs_p*100
        errors = np.array([d['radius_std'] for d in r_dicts ])*1e9
        errors_diff = np.sqrt(errors**2*((e_rs_p-rs_p)/(rs_p*np.abs(e_rs_p-rs_p)))**2)*100
        nan_mask = np.isnan(errors)
        errors2 = np.array([np.diff(d['radius_confidence_interval'])/2 for d in r_dicts]).flatten()*1e9
        # Create simulated vs calculated droplet size plot
        if True:
            plt.plot(rs_p,e_rs_p,color='tab:blue', linewidth=1)
            plt.errorbar(rs_p[~nan_mask],e_rs_p[~nan_mask],yerr=errors[~nan_mask],color='tab:blue', ecolor='tab:blue',capsize=3, linewidth=1, label='estimated radius +- 2*std')
            y = e_rs_p[~nan_mask]
            yerr = errors[~nan_mask]
            plt.fill_between(x=rs_p[~nan_mask], y1=y - yerr, y2=y + yerr, alpha=0.3)
            plt.title('Droplet radius estimation')
            plt.xlabel('Simulated droplet radius R_sim [nm]')
            plt.ylabel('Calculated droplet radius R_calc [nm]')
            plt.legend()
            plt.show()

        # Create relative difference plot
        if True:
            plt.plot(rs_p,rel_diff,color='tab:blue', linewidth=1)
            plt.errorbar(rs_p[~nan_mask],rel_diff[~nan_mask],yerr=errors_diff[~nan_mask],color='tab:blue', ecolor='tab:blue',capsize=3, linewidth=1, label='relative difference +- std')
            y = rel_diff[~nan_mask]
            yerr = errors_diff[~nan_mask]
            plt.fill_between(x=rs_p[~nan_mask], y1=y - yerr, y2=y + yerr, alpha=0.3)
            plt.title('Droplet radius estimation')
            plt.xlabel('Simulated droplet radius R_sim [nm]')
            plt.ylabel('relative difference D = |R_calc - R_sim| / R_sim [%]')
            plt.legend()
            plt.show()


    # Create fit vs data plot#
    if True:
        dg = create_droplet_getter(droplet_radius=5e-8)
        n_frames = 10
        average = np.zeros(epix_slice.stop,dtype=float)
        processed_frames = 0
        for i in range(n_frames):
            data = dg.get_next_frame().processed_data[epix_slice]
            average[:] = (data.astype(float)+processed_frames*average.astype(float))/(processed_frames+1) 
            processed_frames+=1
                #ID = profiler.quickstats(average)['sum']
            #size_list = sp_droplet.fit_profile(ID)
            #estimated_sizes.append(size_list[0])
        ID = profiler.quickstats(average)['sum']    
        try:
            r,r_dict = sp_droplet.fit_profile(ID,minima_persistence_limit=1,minima_start=0,q_boundary_percent=0.05)
            fited_profile = r_dict['A']*r_dict['IR_min']
            plt.semilogy(qs*1e-9,fited_profile,label='Fitted spherical formfactor')
            #print('r = {} std {} confidence {}, r2={}'.format(r,r_dict['radius_std'],r_dict['radius_confidence_interval'],r2))
        except AssertionError as e:
            print(e)
        plt.semilogy(qs*1e-9,ID,label='Simulated SAXS data (10 patterns)')
        plt.legend()
        plt.xlabel('q [nm^-1]')
        plt.ylabel('Intensity [a.U.]')
        plt.title('Fit vs Data for 50nm Droplet')
        plt.show()








