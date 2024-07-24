from sklearn.manifold import Isomap

import L1AnomalyBase
import L1AnomalyPlot

class L1AnomalyIsomap(L1AnomalyBase):
    
    def __init__(self, background_file, signal_files, signal_labels, blackbox_file, classVar = False):
        super().__init__(self, background_file, signal_files, signal_labels, blackbox_file, classVar)
        self.background_data = super.load_data(self, type = 0)
        #returns an array with four signals chronologically
        self.signal_data = super.load_data(self, type = 1)
        self.blackbox_data = super.load_data(self, type = 2)
        
        
    def setup_model(self, n_components, n_neighbors=5):
        self.model = Isomap(n_components = n_components, n_neighbors=n_neighbors)
        
    def plot_bokeh(self, signal = 3):
        L1AnomalyPlot.StandardBokehSplit(self, RS=42, background_data = self.background_data,
                                        signal_data = self.signal_data[signal], blackbox_data = self.blackbox_data)
        L1AnomalyPlot.Bokeh(self, reducer = self.model, reducer_string = "Isomap")