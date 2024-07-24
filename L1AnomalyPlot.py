import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import h5py
import mplhep as hep

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc

from bokeh.plotting import figure, show, output_notebook
from bokeh.models import HoverTool, ColumnDataSource, CategoricalColorMapper
from bokeh.palettes import Spectral10

import L1AnomalyBase

plt.style.use(hep.style.CMS)

#utility class for plotting
class L1AnomalyPlot():
	#Make BokehJS Plots with any manifold learning technique
	
    @staticmethod
    def StandardBokehSplit(self, RS, background_data, signal_data, blackbox_data,
                           X_test_size=0.99995, T_test_size=0.999995, S_test_size=0.9999, B_test_size=0.99995):
        self.X_sample, X_traneous = train_test_split(
			background_data.reshape(background_data.shape[0], -1),
			test_size=X_test_size,
			shuffle=True,
			random_state=RS,
		)
		
        self.T_sample, T_traneous = train_test_split(
			background_data.reshape(background_data.shape[0], -1),
			test_size=T_test_size,
			shuffle=True,
			random_state=RS*7,
		)

        self.S_sample, S_traneous = train_test_split(
			signal_data.reshape(signal_data.shape[0], -1),
			test_size=S_test_size,
			shuffle=True,
			random_state=RS,
        )

        self.B_sample, B_traneous = train_test_split(
			blackbox_data.reshape(blackbox_data.shape[0], -1),
			test_size=B_test_size,
			shuffle=True,
			random_state=RS,
        )

    def BokehClassless(self, reducer, reducer_string):
        raise NotImplementedError
		
    def BokehClass(self, reducer, reducer_string):
        print("Train Test Split Complete!")
		
        X_sample, x_pt_scaler = L1AnomalyBase.scale_pt(X_sample)
        S_sample, s_pt_scaler = L1AnomalyBase.scale_pt(S_sample)
        B_sample, b_pt_scaler = L1AnomalyBase.scale_pt(B_sample)
        T_sample, t_pt_scaler = L1AnomalyBase.scale_pt(T_sample)
		#Every event has different color

        X_sample_pt = X_sample[:, 0::4]
        X_sample_eta = X_sample[:, 1::4]
        X_sample_phi = X_sample[:, 2::4]
        X_sample_class = X_sample[:, 3::4]
        
        S_sample_pt = S_sample[:, 0::4]
        S_sample_eta = S_sample[:, 1::4]
        S_sample_phi = S_sample[:, 2::4]
        S_sample_class = S_sample[:, 3::4]
        
        B_sample_pt = B_sample[:, 0::4]
        B_sample_eta = B_sample[:, 1::4]
        B_sample_phi = B_sample[:, 2::4]
        B_sample_class = B_sample[:, 3::4]
        
        T_sample_pt = T_sample[:, 0::4]
        T_sample_eta = T_sample[:, 1::4]
        T_sample_phi = T_sample[:, 2::4]
        T_sample_class = T_sample[:, 3::4]
        
        #print(reducer_string + " Object")
        #reducer = LocallyLinearEmbedding(n_components=2)
        
        print(reducer_string + " Fit")
        reducer.fit(X_sample)
        print(reducer_string + " Transform")
        embedding = reducer.transform(X_sample)
        
        print("Signal " + reducer_string + " Transform")
        sembedding = reducer.transform(S_sample)
        
        print("Blackbox " + reducer_string + " Transform")
        bembedding = reducer.transform(B_sample)
        
        print("Test Background " + reducer_string + " Transform")
        tembedding = reducer.transform(T_sample)
		#assert(np.all(embedding == reducer.embedding_))
		#assert(np.all(sembedding == sreducer.embedding_))
		#assert(np.all(bembedding == breducer.embedding_))

        output_notebook()
        
        X_sample_df = pd.DataFrame(embedding, columns=('x', 'y'))
        S_sample_df = pd.DataFrame(sembedding, columns=('x', 'y'))
        B_sample_df = pd.DataFrame(bembedding, columns=('x', 'y'))
        T_sample_df = pd.DataFrame(tembedding, columns=('x', 'y'))
		
		# Concatenate the DataFrame and NumPy arrays
        new_data = np.concatenate([X_sample_df.values, X_sample_pt, X_sample_eta, X_sample_phi, X_sample_class], axis=1)
        S_new_data = np.concatenate([S_sample_df.values, S_sample_pt, S_sample_eta, S_sample_phi, S_sample_class], axis=1)
        B_new_data = np.concatenate([B_sample_df.values, B_sample_pt, B_sample_eta, B_sample_phi, B_sample_class], axis=1)
        T_new_data = np.concatenate([T_sample_df.values, T_sample_pt, T_sample_eta, T_sample_phi, T_sample_class], axis=1)
		
		# Create a new DataFrame with the concatenated data
        columns = list(X_sample_df.columns) + [f'pt_{i}' for i in range(19)] + [f'eta_{i}' for i in range(19)] + [f'phi_{i}' for i in range(19)] + [f'class_{i}' for i in range(19)]
        scolumns = list(S_sample_df.columns) + [f'pt_{i}' for i in range(19)] + [f'eta_{i}' for i in range(19)] + [f'phi_{i}' for i in range(19)] + [f'class_{i}' for i in range(19)]
        bcolumns = list(B_sample_df.columns) + [f'pt_{i}' for i in range(19)] + [f'eta_{i}' for i in range(19)] + [f'phi_{i}' for i in range(19)] + [f'class_{i}' for i in range(19)]
        tcolumns = list(T_sample_df.columns) + [f'pt_{i}' for i in range(19)] + [f'eta_{i}' for i in range(19)] + [f'phi_{i}' for i in range(19)] + [f'class_{i}' for i in range(19)]
		
        new_df = pd.DataFrame(new_data, columns=columns)
        s_new_df = pd.DataFrame(S_new_data, columns=scolumns)
        b_new_df = pd.DataFrame(B_new_data, columns=bcolumns)
        t_new_df = pd.DataFrame(T_new_data, columns=tcolumns)
		
        datasource = ColumnDataSource(new_df)
        sdatasource = ColumnDataSource(s_new_df)
        bdatasource = ColumnDataSource(b_new_df)
        tdatasource = ColumnDataSource(t_new_df)

        plot_figure = figure(
			title= reducer_string + ' projection of the Background dataset',
			width=600,
			height=600,
			tools=('pan, wheel_zoom, reset')
		)

        tooltips = "<div>"
        labels = ["MET", "E1", "E2", "E3", "E4", "M1", "M2", "M3", "M4", "J1", "J2", "J3", "J4", "J5", "J6", "J7", "J8", "J9", "J10"]
        for i in range(19):
			#Plot everything from all the indices from first 3 particles - color points from each particle different color
            tooltips += f"""
			<div>
				<div>
					<span style='font-size: 18px'>""" + labels[i] + """: </span>
					<span style='font-size: 18px'>(@pt_""" + str(i) + """,</span>
					<span style='font-size: 18px'>@eta_""" + str(i) + """,</span>
					<span style='font-size: 18px'>@phi_""" + str(i) + """,</span>
					<span style='font-size: 18px'>@class_""" + str(i) + """)</span>
				</div>
			</div>
			"""

        tooltips += "</div>"
        plot_figure.add_tools(HoverTool(tooltips=tooltips))
        plot_figure.circle(
			'x',
			'y',
			source=datasource,
			#color=dict(field='digit', transform=color_mapping),
			line_alpha=0.6,
			fill_alpha=0.6,
			size=4,
			legend_label="Background"
		)
        
        plot_figure.circle(
			'x',
			'y',
			source=sdatasource,
			#color=dict(field='digit', transform=color_mapping),
			line_alpha=0.6,
			fill_alpha=0.6,
			size=4,
			color="red",  # Customize color for signal data
			legend_label="Signal"
		)
        
        plot_figure.circle(
			'x',
			'y',
			source=bdatasource,
			#color=dict(field='digit', transform=color_mapping),
			line_alpha=0.6,
			fill_alpha=0.6,
			size=4,
			color="purple",  # Customize color for signal data
			legend_label="Blackbox"
		)
        
        plot_figure.circle(
			'x',
			'y',
			source=tdatasource,
			#color=dict(field='digit', transform=color_mapping),
			line_alpha=0.6,
			fill_alpha=0.6,
			size=4,
			color="green",  # Customize color for signal data
			legend_label="Test Background"
		)
        
        show(plot_figure)

    @staticmethod
    def Bokeh(self, reducer, reducer_string):
        if (self.nfeat == 3):
            self.BokehClassless(self, reducer, reducer_string)
        else:
            self.BokehClass(self, reducer, reducer_string)
            
            
    @staticmethod
    def ROC(self, reducer, reducer_string):
        plt.figure()
        for signal_file, signal_label in zip(self.signal_files, self.signal_labels):
            with h5py.File(signal_file, 'r') as f:
                if self.classVar == False:
                    signal_data = f['Particles'][:,:,:-1]
                else:
                    signal_data = f['Particles'][:,:,:]

            signal_data = signal_data.reshape(signal_data.shape[0], -1)
            merged_data = np.concatenate([self.X_test, signal_data], axis=0)
            merged_labels = np.concatenate([np.zeros(self.X_test.shape[0]), np.ones(signal_data.shape[0])], axis=0)
            merged_data_trans = reducer.transform(merged_data)
            merged_loss = np.sum(merged_data_trans ** 2, axis=-1)
            fpr, tpr, thresholds = roc_curve(merged_labels, merged_loss)

            tpr_1em5 = L1AnomalyBase.find_nearest(fpr, 1e-5)
            plt.plot(fpr, tpr, label=f"{signal_label}, AUC={auc(fpr, tpr)*100:.2f}%, TPR@FPR $10^{{-5}}$={tpr[tpr_1em5]*100:.3f}%")
        plt.legend(title = reducer_string + " baseline")
        plt.plot([1e-6, 1], [1e-6, 1], 'k--')
        plt.plot([1e-5, 1e-5], [1e-6, 1], 'r-.')
        plt.xlim([1e-6, 1])
        plt.ylim([1e-6, 1])
        plt.loglog()
        plt.savefig(f"gaussian_roc_curve.png")