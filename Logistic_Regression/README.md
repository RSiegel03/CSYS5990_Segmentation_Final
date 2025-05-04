# Logistic Regression Model for CSYS:5990

Ryan Siegel developed the following code and directories, with support from the UVM VACC, GitHub Copilot, ChatGPT, and through discussion with peers.

## Environment Setup

Ensure you have access to a virtual conda environment with the following dependencies
### Requirements:
- PyTorch
- Numpy
- Matplotlib
- Scikit-Learn
- h5py

### Included implementations for Debugging and Progress Checking:
- tqdm

## Usage

### On a Local Editor/IDE
To train a model run:

```train_model2_vacc.py``` -This file is optimized for UVM VACC HPC useage, but also runs alone.

To test a model run:

```test_model.py``` - Also optimized for the UVM VACC HPC, but runs alone

### On an HPC
To train a model run:```run_model.sh```

To test a model run: ```test_model.sh```

## Data
Data is stored within the data directory. To reduce the size of data files, computational time, and RAM useage, all 20 patients' scans were compiled and simplified into 2 files. The model state has also been saved in the 'data' directory as ```checkpoint_3.pth```
### scans.h5
Holds RSL values associated with each scan (2560 total)

### masks.h5
Holds classifications of each corresponding pixel from ```scans.h5``` into:
- 0: Not Cartilage
- 1: Femoral Cartilage
- 2: Tibial Cartilage

## Scripts
All classes and scripts have been stored in the ```~/LogisticRegression/scripts``` directory. Additional files include 4 progress tracking files for non-invasive progress tracking.

### pixel_patch_dataset.py
Holds 2 classes, the dataset class and the sampler class.
The dataset class is a custom PyTorch dataset class, used for easier batching and parallel processing, with the PyTorch dataloader, implemented in the ```train_model2_vacc.py```.

The sampler class is used specifically to allow shuffling of the dataset, but still still allow index tracking, for checkpointing of model training/testing.

### BCE_diceloss.py
Holds 1 class defining the loss metric/criterion for the model.

### cnn_extractor.py
Holds a custom PyTorch CNN. Holds only 8 filters, for various simple features. With only 1 true layer, and the rest maxpooling to reduce size to 8. No backpropagation in an attempt to isolate model training to logistic regression only.

There are 8 filters included in this class, specifically used to handle 

### .log + .out files
Allow easy tracking of progress across models.
