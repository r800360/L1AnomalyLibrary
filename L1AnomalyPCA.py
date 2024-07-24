import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from cuml.decomposition import PCA

import L1AnomalyBase
import L1AnomalyGPU
import L1AnomalyPlot

class L1AnomalyGPUPCA(L1AnomalyBase):

    def __init__(self, background_file, signal_files, signal_labels, blackbox_file, classVar = False, job=False):
        super().__init__(self, background_file, signal_files, signal_labels, blackbox_file, classVar)
        if job == False:
            L1AnomalyGPU.gpu_notebook_install()
        else:
            L1AnomalyGPU.gpu_job_install()
        
        self.background_data = super.load_data(self, type = 0)
        self.signal_data = super.load_data(self, type = 1)
        self.blackbox_data = super.load_data(self, type = 2)
        
    def setup_model(self, n_components, random_state = None):
        self.model = PCA(n_components = n_components, random_state=random_state)
        
    def plot_bokeh(self, signal = 3):
        L1AnomalyPlot.StandardBokehSplit(self, RS=42, background_data = self.background_data,
                                        signal_data = self.signal_data[signal], blackbox_data = self.blackbox_data)
        L1AnomalyPlot.Bokeh(self, reducer = self.model, reducer_string = "PCA")
            
    def plot_explained_background_variance(self, RS = 42, train_size = 0.8):
        X_train, X_test = train_test_split(
            self.background_data.reshape(self.background_data.shape[0], -1),
            train_size=train_size,
            shuffle=True,
            random_state=RS,
        )

        print("Train Test Split Complete")

        X_train_scaled, pt_scaler = L1AnomalyBase.scale_pt(X_train)

        X_train_standard = StandardScaler().fit_transform(X_train_scaled)

        print("Train scale_pt Complete")

        X_test_scaled, _ = L1AnomalyBase.scale_pt(X_test, pt_scaler)

        X_test_standard = StandardScaler().fit_transform(X_test_scaled)

        print("Train and Test scale_pt Complete")

        nums = np.arange(1,58)
        var_ratio = []
        for num in nums:
            pca = PCA(n_components=num)
            pca.fit(X_train_standard)
            var_ratio.append(np.sum(pca.explained_variance_ratio_))
        plt.clf()
        plt.figure()#figsize=(4,2),dpi=150)
        plt.grid()
        plt.plot(nums,var_ratio,marker='o')
        plt.xlabel('n_components')
        plt.ylabel('Explained variance ratio')
        plt.title('n_components vs. Explained Variance Ratio')
        plt.savefig(f"GPU_TSVD_num_components.png")
        
    def plot_ROC(self):
        raise NotImplementedError