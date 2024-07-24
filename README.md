Contains Python scripts that achieve an object-oriented implementation of various explorations conducted for the L1Anomaly Project over the past year. Implementation prioritizes configurability.

L1AnomalyBase.py - class from which all other non-utility classes extend

L1AnomalyEncoderUtil.py - utility class for large methods used primarily to calculate encoder loss

L1AnomalyGPU.py - utility class with install commands for GPU configuration

L1AnomalyPlot.py - utility class containing some plotting tools and frameworks 

L1AnomalyTransform.py - new class not implemented yet - will contain transformations for data augmentation


All in all, there are three not implemented errors that are raised due to code migration in addition to the above L1AnomalyTransform.py error:

L1AnomalyPCA.py - plotting the ROC curve

L1AnomalyPlot.py - using BokehJS without the class variable

L1AnomalyEncoderUtil.py - in mod_make_mse_per_sample when classVar is True (not a high priority)

