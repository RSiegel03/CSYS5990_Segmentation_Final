import cnn_extractor_ideal as cnn
import numpy as np
import torch
from pixel_patch_dataset import PixelPatchDataset as ppds
from pixel_patch_dataset import RandomSkipSampler
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
from BCE_diceloss import BCEDiceLoss
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import pickle
import os

# Parameters
patch_size = 9
batch_size = 256
num_workers = 14 
SAVE_EVERY = 100
scan_file, mask_file, info_file = "../data/scans.h5", "../data/masks.h5", "../data/slice_info.h5"
checkpoint_path = "../data/checkpoint_3.pth"
resume_path, sampler_path = "test_resume.pkl", "test_sampler.pkl"

# File paths for memmap
pred_path = "all_preds.npy"
label_path = "all_labels.npy"

# Load dataset and test indices
dataset = ppds(scan_file, mask_file, info_file, patch_size=patch_size, stride=1)
train_size = 0.7
train_n = int(train_size * len(dataset))
test_n = len(dataset) - train_n
train_dset, test_dset = random_split(dataset, [train_n, test_n])

# Load or initialize resume data
if os.path.exists(resume_path):
    with open(resume_path, 'rb') as f:
        resume_data = pickle.load(f)
    start_batch = resume_data['last_batch'] + 1
    dice_scores = resume_data.get('dice_scores', [])
    losses = resume_data.get('losses', [])
else:
    start_batch = 0
    dice_scores = []
    losses = []

# load previous sample metadata
if os.path.exists(sampler_path):
    with open(sampler_path, 'rb') as f:
        sampler = pickle.load(f)
    completed_indices = sampler.get('completed_idxs', [])
    remaining_indices = sampler.get("remaining_idxs", [])
else:
    remaining_indices = list(range(len(test_dset)))
    completed_indices = set([])

# create new sampler with old data
new_subset = torch.utils.data.Subset(test_dset, list(remaining_indices))
sampler = RandomSkipSampler(new_subset, 0, batch_size, start_batch=start_batch)
sampler.completed_indices = completed_indices

# create new dataloader from prev checkpoints
subloader = DataLoader(new_subset, batch_size=batch_size, 
                        num_workers=num_workers, pin_memory=False, 
                        shuffle=False, sampler=sampler)

# Setup model
cnn_model = cnn.CNNFeatureExtractor()
input_size = 8 * (patch_size // 8)**2 + 3  # features + position
output_size = 3
log_model = nn.Sequential(
    nn.Linear(in_features=input_size, out_features=output_size),
    nn.Sigmoid()
)

# Load trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
log_model.load_state_dict(torch.load(checkpoint_path, map_location=device)['model_state_dict'])
log_model.to(device)
log_model.eval()
cnn_model.to(device)
cnn_model.eval()

# Criterion
criterion = BCEDiceLoss()

# Setup memmap arrays
pred_memmap = np.memmap(pred_path, dtype='int64', mode='r+' if os.path.exists(pred_path) else 'w+', shape=(len(test_dset),))
label_memmap = np.memmap(label_path, dtype='int64', mode='r+' if os.path.exists(label_path) else 'w+', shape=(len(test_dset),))

# Testing loop
print("Start Batch:", start_batch , " |  Total Size:",test_n // batch_size, " |  Subset Size:", len(subloader))
print(len(completed_indices))
print(len(remaining_indices))
try:
    with torch.no_grad():
        for batch_i, batch in enumerate(tqdm(subloader, desc="Testing")):
            # load batch data
            patches = batch['patch'].to(device)
            positions = batch['position'].to(device)
            y_true = batch['label'].to(device).long()
            y_true_onehot = torch.nn.functional.one_hot(y_true, num_classes=3).float()

            # extract features
            features = cnn_model(patches)
            features = features.view(features.size(0), -1)
            combined = torch.cat((features, positions), dim=1)

            # make predictions and evaluate
            y_pred = log_model(combined)
            loss, dice = criterion(y_pred, y_true_onehot)

            # get classes
            preds = torch.argmax(y_pred, dim=1).cpu().numpy()
            labels = y_true.cpu().numpy()

            # Save to memmap
            start_idx = batch_i * batch_size
            end_idx = start_idx + len(preds)
            pred_memmap[start_idx:end_idx] = preds
            label_memmap[start_idx:end_idx] = labels

            dice_scores.append(dice.item())
            losses.append(loss.item())

            # Save resume info
            # Only save every N batches
            if (batch_i % SAVE_EVERY == 0 and batch_i > 0) or batch_i < len(subloader)-1 :
                pred_memmap.flush()
                label_memmap.flush()
                with open(resume_path, 'wb') as f:
                    pickle.dump({
                        'last_batch': batch_i + start_batch,
                        'dice_scores': dice_scores,
                        'losses': losses,
                        "completed_idx": sampler.completed_indices,
                        "remaining)idxs" : sampler.remaining_indices
                    }, f)
                
                with open("test_progress.log", "w") as f:
                    f.write(f"Batch: {batch_i + start_batch}, loss {loss}, dice {dice}\n\n=== Evaluation Metrics ===\n{classification_report(labels, preds, digits=4)}\nConfusion Matrix:\n{confusion_matrix(labels, preds)}\nMean Dice Score: {np.mean(dice_scores)}")

except KeyboardInterrupt:
    print("Interrupted. Saving...")
    pred_memmap.flush()
    label_memmap.flush()
    with open(resume_path, 'wb') as f:
        pickle.dump({
            'last_batch': batch_i + start_batch,
            'dice_scores': dice_scores,
            'losses': losses
        }, f)
finally:
    with open(sampler_path, "wb") as f:
        pickle.dump({"completed_idxs":sampler.completed_indices,
                     "remaining_idxs": sampler.remaining_indices}
            , f)
    print(f"Progress saved at batch {batch_i}.")

# Load full predictions and evaluate
all_preds = np.memmap(pred_path, dtype='int64', mode='r', shape=(len(test_dset),))
all_labels = np.memmap(label_path, dtype='int64', mode='r', shape=(len(test_dset),))

print("\n=== Evaluation Metrics ===")
print(classification_report(all_labels, all_preds, digits=4))
print("Confusion Matrix:")
print(confusion_matrix(all_labels, all_preds))
print(f"Mean Dice Score: {np.mean(dice_scores):.4f}")

"""

linear_layer = log_model[0]
weights = linear_layer.weight.data.cpu().numpy()  # shape: (output_size, input_size)
biases = linear_layer.bias.data.cpu().numpy()     # shape: (output_size,)

for i, class_weights in enumerate(weights):
    print(f"Class {i} weights:")
    print(class_weights)
    print(f"Bias: {biases[i]}")
    print()
"""
remaining_indices = sampler.remaining_indices
print(len(remaining_indices))