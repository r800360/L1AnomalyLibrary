import datetime
import h5py
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import HeUniform
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, Dense, Dropout, MaxPooling2D, concatenate, Flatten, GlobalAveragePooling2D, BatchNormalization, LeakyReLU, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TerminateOnNaN, TensorBoard

# GPU UMAP
import cudf
from cuml.manifold.umap import UMAP as cumlUMAP


import L1AnomalyBase
import L1AnomalyGPU

class L1AnomalyGPUUMAP(L1AnomalyBase):

    def __init__(self, background_file, signal_files, signal_labels, blackbox_file, classVar = False, job=False):
        super().__init__(self, background_file, signal_files, signal_labels, blackbox_file, classVar)
        if job == False:
            L1AnomalyGPU.gpu_notebook_install()
        else:
            L1AnomalyGPU.gpu_job_install()
        
        self.background_data = super.load_data(self, type = 0)
        self.scaled_background_data, pt_scaler = L1AnomalyBase.scale_pt(self.background_data)
        
        #returns a (0-indexed) array with four signals chronologically:
        #"$LQ \\to b\\tau$", "$A \\to 4\\ell$", "$h^{{\pm}} \\to \\tau\\nu$", "$h^{{0}} \\to \\tau\\tau$"
        self.signal_data = super.load_data(self, type = 1)
        
        self.blackbox_data = super.load_data(self, type = 2)
        
		# super.load_data(self)
		# self.background_data = self.background_data.reshape(self.background_data.shape[0], -1)

    def plot_background_signal(self, RS = 42, train_size=0.10, signal_train_size=0.99):
        #Plotting Data split - test is unused
        X_train, X_test = train_test_split(
			self.background_data.reshape(self.background_data.shape[0], -1),
			train_size=train_size,
			shuffle=True,
			random_state = RS
		)
        
        signal_train, signal_test = train_test_split(
			self.signal_data.reshape(self.signal_data.shape[0], -1),
			signal_train_size=signal_train_size,
			shuffle=True,
            random_state = RS
		)

        labels_background = np.zeros(X_train.shape[0])
        labels_signal = np.ones(signal_train.shape[0])
        labels_combined = np.concatenate((labels_background, labels_signal))
        
        X_train_combined = np.concatenate((X_train, signal_train))
        print("Train Test Split Complete")
        
        
        X_train_combined_scaled, pt_scaler = L1AnomalyBase.scale_pt(X_train_combined)
        print("scale_pt Complete")
        
        newEmbedding = cumlUMAP(low_memory=True, random_state = 42).fit_transform(X_train_combined_scaled)
        plt.figure(figsize=(10, 6))  # Adjust width and height as needed
        plt.scatter(newEmbedding[:, 0], newEmbedding[:, 1], c=labels_combined, cmap='coolwarm', alpha=0.05, marker='.', s=1)
        plt.title('UMAP Scaled Projection of Training Data', fontsize=24)
        plt.colorbar(label='Labels (0: Background, 1: Signal)')
        plt.savefig("10PB99PS UMAP Scaled Projection Training Data" + datetime.datetime.now().strftime("%m-%d-%Y-%H:%M:%S"))
        print("Success")
        plt.clf()

    def make_embedding(self, n_components = 2, n_neighbors=100, batch_size = 134520):
		#Roughly tenth in each batch
        embeddings = []
        for i in range(0, len(self.scaled_background_data), batch_size):
            batch_data = self.scaled_background_data[i:i+batch_size]
            print("Batch embedding started")
            embedding_batch = cumlUMAP(n_neighbors=n_neighbors,n_components=n_components).fit_transform(batch_data)
            embeddings.append(embedding_batch)
            print("Batch embedding complete")
        self.embedding = np.concatenate(embeddings)
        self.n_components = n_components
        self.n_neighbors = n_neighbors
		#embedding = cumlUMAP(low_memory=True).fit_transform(scaled_background_data)
        print("Embedding Complete")

		# Save the embedding to an HDF5 file
        self.output_file = "NN" + str(n_neighbors) + "_" + str(n_components) + "D_UMAP_background_embedding.h5"
        with h5py.File(self.output_file, "w") as hf:
            hf.create_dataset("UMAP_background_embedding", data=self.embedding)
        print(f"Embedding saved to {self.output_file}")

    
    def split_embedding(self, RS = 42, train_size = 0.7, val_size=0.5, forward = False):
        if forward == False:
            self.X_train, self.X_temp, self.y_train, self.y_temp = train_test_split(self.embedding, self.scaled_background_data, train_size=train_size, random_state=RS)
        else:
            self.X_train, self.X_temp, self.y_train, self.y_temp = train_test_split(self.scaled_background_data, self.embedding, train_size=train_size, random_state=RS)
        self.X_val, self.X_test, self.y_val, self.y_test = train_test_split(self.X_temp, self.y_temp, train_size=val_size, random_state=RS)

    
    # Define the Neural Network Architecture - call with build_train_eval_model
    def build_model(self, input_dim, output_dim):
        model = Sequential([
			Dense(512, activation='relu', input_dim=input_dim),
			Dense(256, activation='relu'),
			Dense(128, activation='relu'),
            # Dense(80, activation='relu'),
            # Dense(57, activation='relu'),
			Dense(output_dim, activation='linear')
		])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model


	#Use embedding associated with current object to train a model
    def build_train_eval_model(self, forward = False):
        input_dim = self.embedding.shape[1]
        output_dim = self.scaled_background_data.shape[1]
        self.model = self.build_model(input_dim, output_dim)
        
        # Train the Neural Network
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        self.history = self.model.fit(self.X_train, self.y_train, 
							validation_data=(self.X_val, self.y_val), 
							epochs=50, 
							batch_size=256, 
							callbacks=[early_stopping])

		# Evaluate the Neural Network
        test_loss = self.model.evaluate(self.X_test, self.y_test)
        print(f'Test Loss: {test_loss}')

		# Make Predictions
        y_pred = self.model.predict(self.X_test)
        for i in range(5):
            print(f"Predicted: {y_pred[i]}, Actual: {self.y_test[i]}")

		# Save the Model
        savestring = "NN" + str(self.n_neighbors) + "_" + str(self.n_components) + 'D_umap_inverse_model.h5'
        self.model.save(savestring)
        print("Model saved as " + savestring)


	#Load a pre-saved embedding
    def load_embedding(self, embedding_file, n_components = None, n_neighbors = None):
        with h5py.File(embedding_file, "r") as hf:
    		#print("Keys: %s" % hf.keys())
            self.embedding = hf["UMAP_background_embedding"][:, :]
            print(f"Embedding loaded from {embedding_file}")
            
        if (n_components is not None):
            self.n_components = n_components
            
        if (n_neighbors is not None):
            self.n_neightbors = n_neighbors