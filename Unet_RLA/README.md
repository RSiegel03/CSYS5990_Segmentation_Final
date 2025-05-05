# Unet/Reinforcement Learning Preprocessing Optimization

This .ipynb file contains code for the data loading, preprocessing, and implementation of a U-net used for articular cartilage segmentation. Also included is the exploratory testing of RL applications to preprocessing filter selection. Unfortunately, data for this project is not publicly available as it involves medical records and is proprietary to the MIOB lab at the University of Vermont. 

## Installation

To install, simply download the .ipynb notebook by either cloning the repository or directly downloading the notebook to your device. This notebook was developed and run in Google Colab.

## Usage

As data is not available, it is not possible to directly recreate the figures associated with the notebook. New data can be applied to the same pipeline by altering the data loader classes. 

This notebook currently relies on mounting the user's Google Drive for file access and storage of models. 

## Environment Setup
**Requirements:**
- PyTorch
- Numpy
- Matplotlib
- Scikit-Learn
- Google
- os
- Pandas
- Scipy
- TQDM
- Random

## Scripts
All classes and functions are contained within this single .ipynb. The first section is for the definition and training of a standard U-net. The next section applies reinforcement learning to the previously trained U-net to try to optimize filter parameters in order to improve the model's performance. Finally, an exhaustive search of the same space is done to verify the proper functionality of the RL pipeline. 
