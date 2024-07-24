import datetime
import matplotlib.pyplot as plt
import h5py
import numpy as np
from sklearn.metrics import roc_curve, auc
#Class that handles AE and VAE and bagging models
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import HeUniform
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, LeakyReLU, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TerminateOnNaN, TensorBoard

from sklearn.model_selection import train_test_split

import L1AnomalyBase
import L1AnomalyPlot

class L1AnomalyEncoder(L1AnomalyBase):
    
    def __init__(self, background_file, signal_files, signal_labels, blackbox_file, classVar = False, latent_dim=3, variational = False):
        super().__init__(self, background_file, signal_files, signal_labels, blackbox_file, classVar)
        self.background_data = super.load_data(self, type = 0)
        #returns an array with four signals chronologically
        self.signal_data = super.load_data(self, type = 1)
        self.blackbox_data = super.load_data(self, type = 2)
        self.latent_dim = latent_dim
        self.variational = variational

    def dnnae_architecture(self, inputs):
        x = BatchNormalization()(inputs)
        x = Dense(32, kernel_initializer=HeUniform())(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.3)(x)
        x = Dense(16, kernel_initializer=HeUniform())(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.3)(x)
        x = Dense(self.latent_dim, kernel_initializer=HeUniform())(x)
        
        self.intermediate = x
        
        x = Dense(16, kernel_initializer=HeUniform())(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.3)(x)
        x = Dense(32, kernel_initializer=HeUniform())(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.3)(x)
        outputs = Dense(self.X_train.shape[1], kernel_initializer=HeUniform())(x)
        return outputs
    
    def dnnvae_architecture(self, inputs):
        #Batch Normalization
        x = BatchNormalization()(inputs)
        #Block 1
        x = Dense(32, kernel_initializer=HeUniform())(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.3)(x)
        
        #Block 2
        x = Dense(16, kernel_initializer=HeUniform())(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.3)(x)
        
        #Block 3
        x = Dense(self.latent_dim, kernel_initializer=HeUniform())(x)
        meanLatentSpaceVector = Dense(3, activation='linear')(x)
        logVarVector = Dense(3, activation='linear')(x)
        epsilon = tf.random.normal(tf.shape(meanLatentSpaceVector), mean=0.0, stddev=1.0)
        z = meanLatentSpaceVector + tf.exp(0.5 * logVarVector) * epsilon
        
        self.intermediate = z
        
        # Block 4
        z = Dense(16, kernel_initializer=HeUniform())(z)
        z = BatchNormalization()(z)
        z = LeakyReLU(alpha=0.3)(z)

        # Block 5
        z = Dense(32, kernel_initializer=HeUniform())(z)
        z = BatchNormalization()(z)
        z = LeakyReLU(alpha=0.3)(z)
        
        decoderEpsilon = tf.random.normal(tf.shape(meanLatentSpaceVector), mean=0.0, stddev=1.0)
        decoderZ = meanLatentSpaceVector + tf.exp(0.5 * logVarVector) * decoderEpsilon
        
        # Output Layer
        outputs = Dense(self.X_train.shape[1], kernel_initializer=HeUniform())(decoderZ)
        outputs = Concatenate(axis=1)([meanLatentSpaceVector, outputs, logVarVector])
        print("Model architecture setup")
        
        return outputs
		
    def build_model(self):
        
        super.preprocess_data()
        
        callbacks = [
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.1,
                patience=2,
                verbose=1,
                mode="auto",
                min_delta=0.0001,
                cooldown=2,
                min_lr=1e-6,
            ),
            TerminateOnNaN(),
            EarlyStopping(
                monitor="val_loss", verbose=1, patience=10, restore_best_weights=True
            ),
            TensorBoard(
                log_dir=("./VAELOGS" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
            ),
        ]


        self.inputs = Input(shape=(self.X_train.shape[1],))
        
        #self.intermediate will be set here
        if self.variational == False:
            outputs = self.dnnae_architecture(self, self.inputs)
        else:
            outputs = self.dnnvae_architecture(self, self.inputs)
        
        self.model = Model(inputs=self.inputs, outputs=outputs)
        #Use inheritance to grab the loss function from L1AnomalyBase
        self.model.compile(optimizer=Adam(lr=0.00001), loss=self.make_mse)
        self.model.fit(
            self.X_train,
            self.X_train_scaled,
            epochs=10,
            batch_size=1024,
            validation_data=(self.X_val, self.X_val_scaled),
            callbacks=callbacks,
        )

    def plot_bokeh(self, signal=3, RS = 42, test_size = 0.20):
        L1AnomalyPlot.StandardBokehSplit(self, RS=42, background_data = self.background_data,
                                        signal_data = self.signal_data[signal], blackbox_data = self.blackbox_data)
        self.X_train_val = self.X_sample
        self.X_sample, self.X_val = train_test_split(
            self.X_train_val,
            test_size=test_size,
            shuffle=True,
            random_state=RS,
        )
        
        self.X_sample_unscaled = self.X_sample
        self.X_val_unscaled = self.X_val

        self.X_sample, self.x_pt_scaler = L1AnomalyBase.scale_pt(self.X_sample)
        self.X_val, self.x_val_pt_scaler = L1AnomalyBase.scale_pt(self.X_val)
        self.S_sample, self.s_pt_scaler = L1AnomalyBase.scale_pt(self.S_sample)
        self.B_sample, self.b_pt_scaler = L1AnomalyBase.scale_pt(self.B_sample)
        self.T_sample, self.t_pt_scaler = L1AnomalyBase.scale_pt(self.T_sample)
        #Every event has different color
        
        self.X_sample_pt = self.X_sample[:, 0::4]
        self.X_sample_eta = self.X_sample[:, 1::4]
        self.X_sample_phi = self.X_sample[:, 2::4]
        self.X_sample_class = self.X_sample[:, 3::4]

        self.S_sample_pt = self.S_sample[:, 0::4]
        self.S_sample_eta = self.S_sample[:, 1::4]
        self.S_sample_phi = self.S_sample[:, 2::4]
        self.S_sample_class = self.S_sample[:, 3::4]

        self.B_sample_pt = self.B_sample[:, 0::4]
        self.B_sample_eta = self.B_sample[:, 1::4]
        self.B_sample_phi = self.B_sample[:, 2::4]
        self.B_sample_class = self.B_sample[:, 3::4]
        
        self.T_sample_pt = self.T_sample[:, 0::4]
        self.T_sample_eta = self.T_sample[:, 1::4]
        self.T_sample_phi = self.T_sample[:, 2::4]
        self.T_sample_class = self.T_sample[:, 3::4]
        
        self.reducer = Model(inputs=self.inputs, outputs=self.intermediate)
        
        if self.variational == False:
            L1AnomalyPlot.Bokeh(self, reducer = self.reducer, reducer_string = "DNNAE")
        else:
            L1AnomalyPlot.Bokeh(self, reducer = self.reducer, reducer_string = "DNNVAE")        
        
    def build_bagging_model(self):
        input_a = Input(shape=(self.X_train.shape[1],))
        input_b = Input(shape=(self.X_train.shape[1],))

        output_a = self.dnnae_architecture(input_a)
        output_b = self.dnnae_architecture(input_b)

        self.model = Model(inputs=[input_a, input_b], outputs=[output_a, output_b])
        self.model.compile(optimizer=Adam(lr=0.00001), loss=self.make_mse)

    def train_model(self, epochs=10, batch_size=1024):
        callbacks = [
            ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=2, verbose=1, mode="auto", min_delta=0.0001, cooldown=2, min_lr=1e-6),
            TerminateOnNaN(),
            EarlyStopping(monitor="val_loss", verbose=1, patience=10, restore_best_weights=True)
        ]

        self.model.fit([self.X_train, self.X_train], [self.X_train_scaled, self.X_train_scaled],
                       epochs=epochs, batch_size=batch_size,
                       validation_data=([self.X_val, self.X_val], [self.X_val_scaled, self.X_val_scaled]),
                       callbacks=callbacks)

    def generate_roc_curve(self):
        background_loss = self.get_loss([self.X_test, self.X_test], [self.X_test_scaled, self.X_test_scaled])
        plt.figure()
        for signal_file, signal_label in zip(self.signal_files, self.signal_labels):
            with h5py.File(signal_file, "r") as f:
                signal_data = f["Particles"][:, :, :-1]
            signal_data = signal_data.reshape(signal_data.shape[0], -1)
            signal_data_scaled, _ = self.scale_pt(signal_data, self.pt_scaler)
            merged_labels = np.concatenate([np.zeros(self.X_test.shape[0]), np.ones(signal_data.shape[0])], axis=0)
            signal_loss = self.get_loss([signal_data, signal_data], [signal_data_scaled, signal_data_scaled])
            merged_loss = np.concatenate([background_loss, signal_loss], axis=0)
            fpr, tpr, thresholds = roc_curve(merged_labels, merged_loss)
            tpr_1em5 = L1AnomalyBase.find_nearest(fpr, 1e-5)
            plt.plot(fpr, tpr, label=f"{signal_label}, AUC={auc(fpr, tpr)*100:.2f}%, TPR@FPR $10^{{-5}}$={tpr[tpr_1em5]*100:.3f}%")
        plt.legend(title="DNNAE baseline")
        plt.plot([1e-6, 1], [1e-6, 1], "k--")
        plt.plot([1e-5, 1e-5], [1e-6, 1], "r-.")
        plt.xlim([1e-6, 1])
        plt.ylim([1e-6, 1])
        plt.xlabel("Background Efficiency (FPR)")
        plt.ylabel("Signal Efficiency (TPR)")
        plt.xscale("log")
        plt.yscale("log")
        plt.grid(True)
        plt.show()
