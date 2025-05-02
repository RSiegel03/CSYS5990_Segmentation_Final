import cnn_extractor_ideal as cnn
import numpy as np
import torch
import torch.optim as optim
from pixel_patch_dataset import PixelPatchDataset as ppds
from pixel_patch_dataset import RandomSkipSampler
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
from tqdm import tqdm
from BCE_diceloss import BCEDiceLoss
import os
import gc

# model details
batch_size = 64
num_workers = 8
patch_size = 9
lr = 0.01
epochs = 50
checkpoint_size = 200
train_size = 0.7
skip_prob = 0.4

# files
scan_file, mask_file, info_file = "../data/scans.h5", "../data/masks.h5", "../data/slice_info.h5"
test_idx_file, train_idx_file = "test_indices.pt", "train_indices.pt"
checkpoint_path = "../data/checkpoint_3.pth"
loss_file = "losses.npy"
sampler_path = "training_sampler.pkl"

#open classes/datasets
print("Opening Models and Datasets")
dataset = ppds(scan_file, mask_file, info_file, 
                patch_size=patch_size, stride=1)

# create feature extractor model
cnn_model = cnn.CNNFeatureExtractor()

# create multiclass logistic reg model
input_size = 8 * (patch_size // 8)**2 + 3 #  features, 2d maxpool2x2 3-layers = input size, plus 3 positional
output_size = 3
log_model = nn.Sequential(
    nn.Linear(in_features=input_size, out_features=output_size),
    nn.Sigmoid()) # for binary 
log_criterion = BCEDiceLoss() # loss function
optimizer = optim.Adam(log_model.parameters(), lr=lr) # grad descent

# run model in parallel if possible
print("Initializing GPU Usage")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("GPU Use:", torch.cuda.is_available())
log_model.to(device)
cnn_model.to(device)


# check and load previous model states
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    log_model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    start_batch = checkpoint['batch'] + 1
    print(f"Resuming from epoch {start_epoch}, batch {start_batch}")
else:
    start_batch = 0
    start_epoch = 0

# create dataloader
train_n = int(train_size * len(dataset))
test_n = len(dataset) - train_n
train_dset, test_dset = random_split(dataset, [train_n, test_n])
dataset = train_dset
del train_dset
del test_dset
sampler = RandomSkipSampler(dataset, skip_prob, batch_size, start_batch=start_batch)
dataloader = DataLoader(dataset, batch_size=batch_size, 
                        num_workers=num_workers,
                        shuffle=False, sampler=sampler)

# save losses over epochs
if os.path.exists(loss_file):
    losses = np.load(loss_file)
else:
    losses = np.zeros(epochs)
last_batch = 0

# allow checkpointing when cancelling
try:
    pbar_epoch = tqdm(total=epochs, desc=f"Epochs")
    pbar_epoch.update(start_epoch)

    for epoch_i in range(start_epoch, epochs):
            # show progress bar
        batch_pbar = tqdm(total=len(dataloader), desc="Batch Training")
        for batch_i, batch in enumerate(dataloader):
            # Move data to device
            patch = batch['patch'].to(device)
            position = batch['position'].to(device)
            y_true = batch['label'].to(device).long()
            y_true = torch.nn.functional.one_hot(y_true, num_classes=3).float()

            # Forward pass feature extractor
            with torch.no_grad():
                features = cnn_model(patch)
                features = features.view(features.size(0), -1)  # Flatten the features    
                features = torch.cat((features, position), dim=1) # Append position values to the features tensor
                
            # run log reg
            y_pred = log_model(features) # give softmax probs
            #y_class = torch.argmax(y_pred, dim=1, keepdim=True).squeeze()  # prediction is max 
            loss, dice = log_criterion(y_pred, y_true)

            # run grad descent on logistic model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_start_idx = batch_i * batch_size
            batch_end_idx = (batch_i + 1) * batch_size
        
                            # Save progress every 10 batches
            if (batch_i+start_batch )% checkpoint_size == 0 or batch_i == len(dataloader) - 1:
                print(f"Epoch [{batch_i+ (start_batch if epoch_i == start_epoch else 0)}/{len(dataloader)}], Loss: {loss.item():.4f}, Dice: {dice.item():.4f}")
                torch.save({
                        'model_state_dict': log_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        "epoch": epoch_i,
                        "batch": batch_i+start_batch
                }, checkpoint_path)

                torch.save({"completed_idxs":sampler.completed_indices,
                            "remaining_idxs":sampler.remaining_indices},
                        sampler_path)

                with open("progress.log", "w") as f:
                    f.write(f"Epoch: {epoch_i}, Batch: {batch_i + start_batch}, loss {loss}, dice {dice}\n")

                    #print(f"Checkpoint saved at batch {i + 1}")
                gc.collect()
                torch.cuda.empty_cache()
                batch_pbar.update(checkpoint_size)
            

            last_batch = batch_i
            del loss
            del dice

        start_batch=0
        # Loading
        torch.save({
                        'model_state_dict': log_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        "epoch": epoch_i,
                        "batch": start_batch
                }, checkpoint_path)
            
        sampler = RandomSkipSampler(dataset, skip_prob, batch_size, start_batch=start_batch)
        dataloader = DataLoader(dataset, batch_size=batch_size, 
                            num_workers=num_workers, 
                            shuffle=False, sampler=sampler)
            
        losses[epoch_i] = loss
        np.save("losses.npy", losses)

        checkpoint = torch.load(checkpoint_path)
        log_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        pbar_epoch.update(1)

except KeyboardInterrupt:
    print("Cancelling Feature Extraction")
    if epoch_i >1 and last_batch > checkpoint_size:
        torch.save({'model_state_dict': log_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    "epoch": epoch_i,
                    "batch": last_batch
                    }, checkpoint_path)
        np.save(loss_file, losses)
        print(f"Saved at batch index : {epoch_i}")

        torch.save({"completed_idxs":sampler.completed_indices,
                    "remaining_idxs":sampler.remaining_indices},
                    sampler_path)


    

