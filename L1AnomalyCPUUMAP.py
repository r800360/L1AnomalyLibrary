import datetime
import h5py
import matplotlib.pyplot as plt
import numpy as np
import umap

from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split

import L1AnomalyBase
import L1AnomalyEncoderUtil
import L1AnomalyPlot

class L1AnomalyCPUUMAP(L1AnomalyBase):
    def __init__(self, background_file, signal_files, signal_labels, blackbox_file, classVar = False, job=False):
        super().__init__(self, background_file, signal_files, signal_labels, blackbox_file, classVar)
        
        self.background_data = super.load_data(self, type = 0)
        self.scaled_background_data, pt_scaler = L1AnomalyBase.scale_pt(self.background_data)
        
        #returns a (0-indexed) array with four signals chronologically:
        #"$LQ \\to b\\tau$", "$A \\to 4\\ell$", "$h^{{\pm}} \\to \\tau\\nu$", "$h^{{0}} \\to \\tau\\tau$"
        self.signal_data = super.load_data(self, type = 1)
        
        self.blackbox_data = super.load_data(self, type = 2)
        
        super().preprocess_data_v2(filter_size = 0.96, test_size = 0.2)
        
    def plot_bokeh(self, signal = 3):
        L1AnomalyPlot.StandardBokehSplit(self, RS=42, background_data = self.background_data,
                                        signal_data = self.signal_data[signal], blackbox_data = self.blackbox_data)
        L1AnomalyPlot.Bokeh(self, reducer = self.model, reducer_string = "UMAP")
    
    def get_loss(X_scaled,inv_transform_data):
    #Extract loss from model predictions using make_mse_per_sample function analyzed above
        return np.array(L1AnomalyEncoderUtil.make_mse_per_sample(X_scaled, inv_transform_data))
    
    def get_background_loss(self):
        plt.clf()
        print("Starting UMAP training embedding and plot generation")
        self.model = umap.UMAP(low_memory=True)
        #X_train_scaled_distance_matrix = distance_matrix(X_train_scaled)
        trainEmbedding = self.model.fit_transform(self.X_train_scaled)
        plt.scatter(trainEmbedding[:, 0], trainEmbedding[:, 1], c=np.arange(self.X_train_scaled.shape[0]))
        plt.title('UMAP Projection of Scaled Training Data', fontsize=24)
        #plt.savefig("One-Tenth Percent UMAP Scaled Projection Training Background Data" + x.strftime("%m-%d-%Y-%H:%M:%S"))
        #plt.clf()
        inv_transform_training_data = self.model.inverse_transform(trainEmbedding)
        print("Successful Training Embedding, Projection Figure, and Inverse Transform")

        self.background_loss = self.get_loss(self.X_train_scaled, inv_transform_training_data)

        
    def plot_ROC(self, signal_filter_size = 0.80):
        plt.figure()
        for signal_file, signal_label in zip(self.signal_files, self.signal_labels):
            with h5py.File(signal_file, "r") as f:
                large_signal_data = f["Particles"][:, :, :]

            signal_data, signal_excess = train_test_split(
                large_signal_data.reshape(large_signal_data.shape[0], -1),
                test_size=signal_filter_size,
                shuffle=True,
                random_state=42,
            )

            signal_data = signal_data.reshape(signal_data.shape[0], -1)
            print("Signal Data Ready")
            output_file_signal_data = signal_label + str(signal_filter_size) + ".h5"
            with h5py.File(output_file_signal_data, "w") as hf_signal_data:
                hf_signal_data.create_dataset(signal_label, data=signal_data)
            print("Signal Data Saved")
            signal_data_scaled, _ = L1AnomalyBase.scale_pt(signal_data, self.pt_scaler)
            merged_labels = np.concatenate(
                [np.zeros(self.X_train.shape[0]), np.ones(signal_data.shape[0])], axis=0
            )

            print("Starting UMAP signal embedding: " + signal_label)
            inv_transform_signal_data = self.model.inverse_transform(self.model.transform(signal_data_scaled))
            #plt.scatter(signalEmbedding[:, 0], signalEmbedding[:, 1], c=np.arange(signal_data_scaled.shape[0]))
            #plt.title('UMAP Projection of Scaled Signal Data', fontsize=24)
            #plt.savefig("One-Tenth Percent UMAP Scaled Projection Signal Data" + x.strftime("%m-%d-%Y-%H:%M:%S"))
            #print("Successful Signal Embedding and Projection Figure")
            #plt.clf()
            #inv_transform_signal_data = model.inverse_transform(signalEmbedding)
            
            signal_loss = self.get_loss(signal_data_scaled, inv_transform_signal_data)
            merged_loss = np.concatenate([self.background_loss, signal_loss], axis=0)
            
            print("Successful UMAP signal embedding, inverse transform, and loss computations: " + signal_label)
            
            fpr, tpr, thresholds = roc_curve(merged_labels, merged_loss)
            tpr_1em5 = L1AnomalyBase.find_nearest(fpr, 1e-5)
            plt.plot(
                fpr,
                tpr,
                label=f"{signal_label}, AUC={auc(fpr, tpr)*100:.2f}%, TPR@FPR $10^{{-5}}$={tpr[tpr_1em5]*100:.3f}%",
            )

        plt.legend(title="UMAP baseline")
        plt.plot([1e-6, 1], [1e-6, 1], "k--")
        plt.plot([1e-5, 1e-5], [1e-6, 1], "r-.")
        plt.xlim([1e-6, 1])
        plt.ylim([1e-6, 1])
        plt.loglog()
        x = datetime.datetime.now()
        plt.savefig(f"New UMAP ROC" + x.strftime("%m-%d-%Y-%H:%M:%S"))
        #plt.plot([0, 1], [0, 1], "k--")
        #plt.plot([1e-5, 1e-5], [1e-6, 1], "r-.")
        #plt.xlim([0, 1])
        #plt.ylim([0, 1])
        #plt.loglog()