import math

import numpy as np
import scipy

class InterferenceHelper():
    C = 299792458.0
    PI = 3.14159265358979323846
    HIDDEN_LOSS = 200.
    NOISE_FLOOR_DBM = -94.
    BOLTZMANN = 1.3803e-23
    NOISEFIGURE = 13

    @staticmethod
    def bandwidth_txpr_to_noise_dBm(B):
        # we assume constant noise floor for simplicity.
        return InterferenceHelper.NOISE_FLOOR_DBM

    @staticmethod
    def fre_dis_to_loss_dB(fre_Hz, dis):
        """
        We use log-distance path loss model, assuming a factory or office scenario.
        @article{series2017propagation,
        title={Propagation data and prediction methods for the planning of short-range outdoor radiocommunication systems and radio local area networks in the frequency range 300 MHz to 100 GHz},
        author={Series, P},
        journal={ITU recommmendations},
        pages={1411--9},
        year={2017}}
        """
        L = 20. * math.log10(fre_Hz/1e6) + 16 - 28
        loss = L + 28 * math.log10(dis+1) # at least one-meter distance
        return loss
    
    @staticmethod
    def db_to_dec(snr_db):
        return 10.**(snr_db/10.)

    @staticmethod
    def dec_to_db(snr_dec):
        return 10.* math.log10(snr_dec)

    @staticmethod
    def polyanskiy_model(snr_dec, L, B, T):
        # we use polyanskiy model to compute the error rate
        nu = - L * math.log(2.) + B * T * math.log(1+snr_dec)
        do = math.sqrt(B * T * (1. - 1./((1.+snr_dec)**2)))
        return scipy.stats.norm.sf(nu/do)

    @staticmethod
    def err(x, L, B, T, max_err):
        snr = InterferenceHelper.db_to_dec(x)
        return InterferenceHelper.polyanskiy_model(snr, L, B, T)/max_err - 1.

    @staticmethod
    def bisection_method(L, B, T, max_err=1e-5, a=-5., b=30., tol=0.1):
        # find minimum snr for given channel setup and error rate.
        if InterferenceHelper.err(a, L, B, T, max_err) * InterferenceHelper.err(b, L, B, T, max_err) >= 0:
            print("Bisection method fails.")
            return None

        while (InterferenceHelper.err(a, L, B, T, max_err) - InterferenceHelper.err(b, L, B, T, max_err)) > tol:
            midpoint = (a + b) / 2
            if InterferenceHelper.err(midpoint, L, B, T, max_err) == 0:
                return midpoint
            elif InterferenceHelper.err(a, L, B, T, max_err) * InterferenceHelper.err(midpoint, L, B, T, max_err) < 0:
                b = midpoint
            else:
                a = midpoint

        return (a + b) / 2

class ApStaNetwork(InterferenceHelper):
    """
    this class implements an ap-sta network to setup wi-fi network simulation.
    
    """
    def __init__(self, cell_edge = 20., cell_size = 20, sta_density_per_1m2 = 5e-3, fre_Hz = 5e9, txp_dbm_hi = 5., min_s_n_ratio = 0.1, packet_bit = 800, bandwidth = 5e6, slot_time=1.25e-4, max_err = 1e-5, seed=1):
        """
        Initializes the simulation environment with the given parameters.

        Parameters:
        -----------
        cell_edge : float
            The edge length of a single cell in meters.
        cell_size : int
            The number of cells along one dimension of the grid.
        sta_density_per_1m2 : float
            The density of stations (STAs) per square meter.
        fre_Hz : float
            The frequency of the signal in Hertz.
        txp_dbm_hi : float
            The transmission power of the access points (APs) in dBm.
        txp_offset : float
            The offset to be applied to the transmission power.
        min_s_n_ratio : float
            The minimum signal-to-noise ratio required.
        packet_bit : int
            The size of each packet in bits.
        bandwidth : float
            The bandwidth of the signal in Hertz.
        slot_time : float
            The time slot duration in seconds.
        max_err : float
            The maximum allowable error rate.
        seed : int
            Seed for random number generation.
        """
        
        
        self.rand_gen_loc = np.random.default_rng(seed)
        self.rand_gen_fad = np.random.default_rng(seed)
        self.rand_gen_mob = np.random.default_rng(seed)

        self.cell_edge = cell_edge
        self.cell_size = cell_size

        self.grid_edge = self.cell_edge * self.cell_size

        self.n_ap = int(self.cell_size ** 2)
        self.ap_offset = self.cell_edge / 2.

        self.sta_density_per_1m2 = sta_density_per_1m2
        self.sta_density_per_grid = self.sta_density_per_1m2 * self.cell_edge ** 2
        self.n_sta = int(self.cell_size**2 * self.sta_density_per_grid)

        self.fre_Hz = fre_Hz
        self.txp_dbm_hi = txp_dbm_hi
        self.min_s_n_ratio = min_s_n_ratio
        self.packet_bit = packet_bit
        self.bandwidth = bandwidth
        self.slot_time = slot_time
        self.max_err = max_err

        self.ap_locs = None
        self.sta_locs = None
        self.sta_dirs = None

        self.min_sinr = None
        self.loss = None

        self._config_ap_locs()
        self._config_sta_locs()
        self._config_sta_dirs()

    def get_loss_ap_ap(self):
        ret = np.zeros((self.n_ap,self.n_ap))
        for i in range(self.n_ap):
            for j in range(self.n_ap):
                if i == j:
                    continue
                ret[i,j] = self._get_loss_between_locs(self.ap_locs[i],self.ap_locs[j])
        return ret

    def get_loss_sta_ap(self):
        ret = np.ones((self.n_sta,self.n_ap))*np.inf
        for i in range(self.n_sta):
            while np.min(ret[i,:]) > 90:
                for j in range(self.n_ap):
                    ret[i,j] = self._get_loss_between_locs(self.sta_locs[i],self.ap_locs[j])
        return ret

    def get_loss_sta_sta(self):
        ret = np.zeros((self.n_sta,self.n_sta))
        for i in range(self.n_sta):
            for j in range(i,self.n_sta):
                if i == j:
                    continue
                ret[i,j] = self._get_loss_between_locs(self.sta_locs[i],self.sta_locs[j])
                ret[j,i] = ret[i,j]
        return ret

    def _get_loss_between_locs(self, a, b):
        dis = np.linalg.norm(np.array(a)-np.array(b),ord=2)
        return self._get_loss_distance(dis)

    def _get_loss_distance(self, dis):
        #shadowing is disabled
        return self.fre_dis_to_loss_dB(self.fre_Hz,dis)

    def convert_loss_sta_ap_threshold(self, loss):
        ret = np.copy(loss)
        ret[ret>(self.txp_dbm-self.min_rssi_dbm)] = self.HIDDEN_LOSS
        return ret

    def convert_loss_sta_sta_binary(self, loss):
        ret = np.copy(loss)
        ret[np.logical_or(ret>(self.txp_dbm-self.min_rssi_dbm), self.txp_dbm-ret-self.NOISE_FLOOR_1MHZ_DBM<0.)] = 0.
        ret[ret>0.] = 1.
        return ret

    def _config_ap_locs(self):
        x=np.linspace(0 + self.ap_offset, self.grid_edge - self.ap_offset, self.cell_size)
        y=np.linspace(0 + self.ap_offset, self.grid_edge - self.ap_offset, self.cell_size)
        xx,yy=np.meshgrid(x,y)
        self.ap_locs = np.array((xx.ravel(), yy.ravel())).T

    def _config_sta_locs(self):
        self.sta_locs = self.rand_gen_loc.uniform(low=0.,high=self.grid_edge,size=(self.n_sta,2))

    def _config_sta_dirs(self):
        dd = self.rand_gen_mob.standard_normal(size=(self.n_sta,2))
        self.sta_dirs = dd/np.linalg.norm(dd,axis=1,keepdims=True)

    def _get_random_dir(self):
        dd = self.rand_gen_mob.standard_normal(2)
        return dd/np.linalg.norm(dd)

    def _compute_min_sinr(self):
        min_sinr_db = InterferenceHelper.bisection_method(self.packet_bit,self.bandwidth,  self.slot_time, self.max_err)
        self.min_sinr = InterferenceHelper.db_to_dec(min_sinr_db)
        return self.min_sinr

    def rand_user_mobility(self, mobility_in_meter_s = 0., t_us = 0, resolution_us = 1.):

        if mobility_in_meter_s == 0. or t_us == 0.:
            return
        n_step = math.ceil(t_us/resolution_us)
        for n in range(n_step):
            for i in range(self.n_sta):
                dd = self.sta_dirs[i] * mobility_in_meter_s * resolution_us/1e6
                x = self.sta_locs[i][0] + dd[0]
                y = self.sta_locs[i][1] + dd[1]
                if 0 <= x <= self.grid_edge and 0 <= y <= self.grid_edge:
                    self.sta_locs[i] = np.array([x,y])
                else:
                    self.sta_dirs[i] = self._get_random_dir()

    def check_cell_edge_snr_err(self):
        l = InterferenceHelper.fre_dis_to_loss_dB(self.fre_Hz,self.cell_edge/2*math.sqrt(2))
        s_db = self.txp_dbm_hi - l - self.bandwidth_txpr_to_noise_dBm(self.bandwidth)
        s_dec = InterferenceHelper.db_to_dec(s_db)
        err = InterferenceHelper.polyanskiy_model(s_dec,self.packet_bit,self.bandwidth,self.slot_time)
        print("snr_db", s_db, "snr_dec", s_dec, "err", err)
        return



if __name__ == "__main__":
    print("test ApStaNetwork")
    t = ApStaNetwork()
    t.check_cell_edge_snr_err()
    
    
