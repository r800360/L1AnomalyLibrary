#Methods for GPU Configuration
import os

class L1AnomalyGPU:

	@staticmethod
	def gpu_notebook_install(self):
	
		if (self.gpu_install_flag == True):
			print("GPU Install Already Completed")
			return
			
		os.system("pip install \
		--extra-index-url=https://pypi.nvidia.com \
		cudf-cu12==23.12.* dask-cudf-cu12==23.12.* cuml-cu12==23.12.* \
		cugraph-cu12==23.12.* cuspatial-cu12==23.12.* cuproj-cu12==23.12.* \
		cuxfilter-cu12==23.12.* cucim-cu12==23.12.* pylibraft-cu12==23.12.* \
		raft-dask-cu12==23.12.* --ignore-installed")

		os.system("pip install protobuf")
		os.system("cp /opt/conda/lib/python3.10/site-packages/google/protobuf/internal/builder.py .")
		os.system("pip install protobuf==3.9.11")
		os.system("cp builder.py /opt/conda/lib/python3.10/site-packages/google/protobuf/internal")
		os.system("pip install cupy-cuda11x")
		
		print("GPU Install Complete")
		self.gpu_install_flag = True
	
	@staticmethod
	def gpu_job_install(self):
	
		if (self.gpu_install_flag == True):
			print("GPU Install Already Completed")
			return
			
		os.system("pip install \
		--extra-index-url=https://pypi.nvidia.com \
		cudf-cu12==24.6.* dask-cudf-cu12==24.6.* cuml-cu12==24.6.* \
		cugraph-cu12==24.6.* cuspatial-cu12==24.6.* cuproj-cu12==24.6.* \
		cuxfilter-cu12==24.6.* cucim-cu12==24.6.* pylibraft-cu12==24.6.* \
		raft-dask-cu12==24.6.* cuvs-cu12==24.6.* --ignore-installed")
		
		os.system("pip install protobuf")
		os.system("pip install protobuf==3.9.11")
		os.system("pip install cupy-cuda11x")

		print("GPU Install Complete")
		self.gpu_install_flag = True