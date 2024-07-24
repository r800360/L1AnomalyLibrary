from sklearn.preprocessing import StandardScaler

import L1AnomalyBase
import L1AnomalyPlot

class L1AnomalyGaussian(L1AnomalyBase):
    def __init__(self, background_file, signal_files, signal_labels, blackbox_file, classVar = False, job=False):
        super().__init__(self, background_file, signal_files, signal_labels, blackbox_file, classVar)
        self.background_data = super.load_data(self, type = 0)
        self.signal_data = super.load_data(self, type = 1)
        self.blackbox_data = super.load_data(self, type = 2)
        
        
    def setup_model(self):
        self.model = StandardScaler()
        super().preprocess_data()
        self.model.fit(self.X_train)
        
    def plot_ROC(self):
        L1AnomalyPlot.ROC(self, self.model, "Gaussian")
        