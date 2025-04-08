**Predictive Analysis of Major Vascular-Related Events Using Deep Learning**

**Overview**

This library is designed for predicting whether a patient will experience major vascular-related events (such as myocardial infarction, stroke, and chronic kidney disease) within the next 10 years, based on fundus images and minimal demographic information (age and gender). The multimodal model is built using the PyTorch platform and takes fundus images along with structured data as input. 

**OS requirements**

This package was trained and tested under Ubuntu 18.04. Modifications may be necessary to run it on other platforms.

**Python Dependencies**

- numpy  
- pandas 
- sklearn 
- torch 
- pillow 
- argparse 
- tqdm

**Code structure**

- data
   - MyDataset.py: construction of dataset for input
- model 
   - model.py: architecture of the multimodal model
   - run.py: details about the implementation of model training and testing
- utils
   -  utils.py: necessary functions
   - metrics.py: evaluation of the predicted results
   - pytorchtools: earlystopping
- visualization
   - Calibration_plot.py: plotting calibration curves
   - Decision_curve.py: plotting decision curves
   - OR.py: calculation of odd ratios
   - ROC_curve.py: plotting ROC curves
