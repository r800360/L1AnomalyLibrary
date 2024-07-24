import h5py
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc

import tensorflow as tf
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.initializers import HeUniform
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Input, Dense, BatchNormalization, LeakyReLU
# from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TerminateOnNaN, TensorBoard

plt.style.use(hep.style.CMS)

class L1AnomalyBase:
    def __init__(self, background_file, signal_files, signal_labels, blackbox_file, classVar = False):
	    #Config flags
        self.gpu_install_flag = False
		#File Attributes
        self.background_file = background_file
        self.signal_files = signal_files
        self.signal_labels = signal_labels
        self.blackbox_file = blackbox_file
		#Size Attributes
        # self.test_size = test_size
        # self.val_size = val_size
        if classVar == True:
            self.nfeat = 4
        else:
            self.nfeat = 3
		#Constants
        self.nmet = 1
        self.nele = 4
        self.nmu = 4
        self.njet = 10
        self.ele_off = 1
        self.mu_off = self.nmet + self.nele
        self.jet_off = self.nmet + self.nele + self.nmu
        self.phi_max = np.pi
        self.ele_eta_max = 3.0
        self.mu_eta_max = 2.1
        self.jet_eta_max = 4.0
		
    @staticmethod    
    def find_nearest(self, array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx

    @staticmethod
    def scale_pt(self, X, pt_scaler=None):
        pt = X[:, 0::self.nfeat]
        if pt_scaler is None:
            pt_scaler = StandardScaler()
            pt_scaled = pt_scaler.fit_transform(pt)
        else:
            pt_scaled = pt_scaler.transform(pt)
        X_scaled = np.copy(X)
        X_scaled[:, 0::self.nfeat] = np.multiply(pt_scaled, pt != 0)
        return X_scaled, pt_scaler

	#0 for background, 1 for signal, 2 for blackbox
    def load_data(self, type = 0):
        if self.classVar == False:
            if type == 0:
			    #load background data
                with h5py.File(self.background_file, "r") as f:
                    self.background_data = f["Particles"][:, :, :-1]
            elif type == 1:
				#load all four signals
                with h5py.File(self.signal_files[0], "r") as f:
                    self.signal_data_0 = f["Particles"][:, :, :-1]
                with h5py.File(self.signal_files[1], "r") as f:
                    self.signal_data_1 = f["Particles"][:, :, :-1]
                with h5py.File(self.signal_files[2], "r") as f:
                    self.signal_data_2 = f["Particles"][:, :, :-1]
                with h5py.File(self.signal_files[3], "r") as f:
                    self.signal_data_3 = f["Particles"][:, :, :-1]
            elif type == 2:
                #load blackbox
                with h5py.File(self.background_file, "r") as f:
                    self.blackbox_data = f["Particles"][:, :, :-1]
        else:
            if type == 0:
                #load background data
                with h5py.File(self.background_file, "r") as f:
                    self.background_data = f["Particles"][:, :, :]
            elif type == 1:
				#load all four signals
                with h5py.File(self.signal_files[0], "r") as f:
                    self.signal_data_0 = f["Particles"][:, :, :]
                with h5py.File(self.signal_files[1], "r") as f:
                    self.signal_data_1 = f["Particles"][:, :, :]
                with h5py.File(self.signal_files[2], "r") as f:
                    self.signal_data_2 = f["Particles"][:, :, :]
                with h5py.File(self.signal_files[3], "r") as f:
                    self.signal_data_3 = f["Particles"][:, :, :]
            elif type == 2:
                #load blackbox
                with h5py.File(self.background_file, "r") as f:
                    self.blackbox_data = f["Particles"][:, :, :]
			
    def preprocess_data(self, RS = 42, test_size=0.2, val_size=0.2):
        self.test_size = test_size
        self.val_size = val_size
        self.X_train_val, self.X_test = train_test_split(self.background_data.reshape(self.background_data.shape[0], -1),
                                                         test_size=self.test_size,
                                                         shuffle=True,
                                                         random_state=RS)
        self.X_train, self.X_val = train_test_split(self.X_train_val, test_size=self.val_size, shuffle=True, random_state=RS)
        self.X_train_scaled, self.pt_scaler = self.scale_pt(self.X_train)
        self.X_val_scaled, _ = self.scale_pt(self.X_val, self.pt_scaler)
        self.X_test_scaled, _ = self.scale_pt(self.X_test, self.pt_scaler)
    
    def preprocess_data_v2(self, RS = 42, filter_size=0.2, test_size=0.2):
        self.test_size = test_size
        self.filter_size = filter_size
        self.X_train_test, self.X_traneous = train_test_split(self.background_data.reshape(self.background_data.shape[0], -1),
                                                         test_size=self.filter_size,
                                                         shuffle=True,
                                                         random_state=RS)
        self.X_train, self.X_test = train_test_split(self.X_train_test, test_size=self.test_size, shuffle=True, random_state=RS)
        self.X_train_scaled, self.pt_scaler = self.scale_pt(self.X_train)
        self.X_test_scaled, _ = self.scale_pt(self.X_test, self.pt_scaler)
    
    