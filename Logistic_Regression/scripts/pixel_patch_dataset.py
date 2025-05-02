import h5py
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm
import random

class PixelPatchDataset(Dataset):
    def __init__(self, scan_path, mask_path, meta_path, patch_size=9, transform=None, stride=1, crop=128):
        self.scan_path = scan_path
        self.mask_path = mask_path
        self.meta_path = meta_path
        self.transform = transform
        self.stride = stride  # You can use this to downsample the number of pixels

        if patch_size//2 == 0:
            self.patch_size = patch_size+1
        else:
            self.patch_size = patch_size

        if crop is not None and crop is not False:
            if isinstance(crop, (int, float)):
                crop = min(int(crop), 512) + self.patch_size-1
            elif isinstance(crop, bool):
                crop = 128 + self.patch_size-1
        else:
            crop = 0 
        self.cropsize = crop
        self.center = 256

        self.scan_file = h5py.File(scan_path, 'r')
        self.mask_file = h5py.File(mask_path, 'r')
        self.meta_file = h5py.File(meta_path, 'r')
        scans = self.scan_file['scans/data']
        scans = scans[self.center - self.cropsize // 2:self.center + self.cropsize // 2, self.center - self.cropsize // 2:self.center + self.cropsize // 2, :]
        self.height, self.width, self.total_slices = scans.shape


        self.load_pixel_coords()
        self.get_cart_type()


    def load_pixel_coords(self):
        # Generate pixel coordinates
        coords = []
        pbar = tqdm(total=self.total_slices, desc="Loading Data")
        for z in range(self.total_slices):
            for y in range(0, self.height, self.stride):
                for x in range(0, self.width, self.stride):
                    # Patch bounds
                    if (x - self.patch_size // 2 < 0 or
                        x + self.patch_size // 2 >= self.width or
                        y - self.patch_size // 2 < 0 or
                        y + self.patch_size // 2 >= self.height):
                        continue
                    coords.append([x, y, z])
            pbar.update(1)
        coords = np.array(coords)
        
        self.pixel_coords = np.zeros((coords.shape[0], 5), dtype=int)
        self.pixel_coords[:, 0] = np.arange(len(coords))  # dummy index
        self.pixel_coords[:, 1:4] = coords  # x, y, z
    
    def get_cart_type(self):
        mask = self.mask_file['masks/data'][self.center - self.cropsize // 2:self.center + self.cropsize // 2, self.center - self.cropsize // 2: self.center + self.cropsize // 2, :].T
        """
        print("Mask shape:", mask.shape)
        print("X coords:", self.pixel_coords[:, 1], self.pixel_coords[:,1].shape)
        print("Y coords:", self.pixel_coords[:, 2], self.pixel_coords[:,2].shape)
        print("Slice index:", self.pixel_coords[:, 3], self.pixel_coords[:,3].shape)
        """
        self.pixel_coords[:, 4] = mask[ self.pixel_coords[:,3]-1, self.pixel_coords[:, 2], self.pixel_coords[:, 1]]


    def __len__(self):
        return len(self.pixel_coords)

    def __getitem__(self, index):
        _, x, y, slice_idx, cart_type = self.pixel_coords[index]

        # Get patch bounds
        patch_start_x = x - self.patch_size // 2
        patch_end_x   = x + self.patch_size // 2 + 1
        patch_start_y = y - self.patch_size // 2
        patch_end_y   = y + self.patch_size // 2 + 1

        scan = self.scan_file['scans/data'][:, :, slice_idx]
        patch = scan[patch_start_x:patch_end_x, patch_start_y:patch_end_y]

        # Normalize patch to [0, 1]
        patch = patch.astype(np.float32)
        patch = (patch - patch.min()) / (patch.max() - patch.min() + 1e-8)

        patch_tensor = torch.tensor(patch, dtype=torch.float32).unsqueeze(0)
        cart_type = torch.tensor(float(cart_type > 0), dtype=torch.float32)

        if self.transform:
            patch_tensor = self.transform(patch_tensor)

        return {
            'patch': patch_tensor,
            'position': torch.tensor([slice_idx, x, y]),
            'label': cart_type,
        }
    
    def visualize_pixel_patch(self, index):
        _, x, y, slice_idx, cart_type = self.pixel_coords[index]
        # Load the full slice from scan file
        with h5py.File(self.scan_path, 'r') as f_scan:
            scan_slice = f_scan['scans/data'][self.center - self.cropsize // 2:self.center + self.cropsize // 2,
                                            self.center - self.cropsize // 2:self.center + self.cropsize // 2,
                                            slice_idx]

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(scan_slice.T, cmap='gray', origin='lower')  # .T and origin='lower' for correct orientation

        # Add red rectangle (1x1) for the pixel
        pixel_type_color = ["red", "green", "yellow"]
        pixel = patches.Rectangle((x, y), 1, 1, linewidth=1.5, edgecolor=pixel_type_color[cart_type], facecolor='none')
        ax.add_patch(pixel)

        # Add blue square (patch_size x patch_size) centered at pixel
        patch_start_x = x - self.patch_size // 2
        patch_start_y = y - self.patch_size // 2
        blue_square = patches.Rectangle((patch_start_x, patch_start_y),
                                        self.patch_size, self.patch_size,
                                        linewidth=1.5, edgecolor='blue', facecolor='none')
        ax.add_patch(blue_square)

        ax.set_title(f"Dataset Index: {index} - Slice {slice_idx} - Pixel ({x}, {y})")
        plt.grid(False)
        plt.axis('off')
        plt.show()




"""
from torch.utils.data import DataLoader
dataset = PixelPatchDataset('scans.h5', 'masks.h5', 'slice_info.h5', stride = 1)

loader = DataLoader(dataset, batch_size=4, shuffle=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for i, batch in enumerate(loader):
    if i == 4:
        break 
    print(len(loader))
    batch['patch'] = batch['patch'].to(device)
    batch['position'] = batch['position'].to(device)
    batch['label'] = batch['label'].to(device)
    print("Patch shape:", batch['patch'].shape)
    print("Position shape:", batch['position'].shape)
    print("Label shape:", batch['label'].shape)
    print("Index:", batch['index'])
"""
class RandomSkipSampler(Sampler):
    def __init__(self, data_source, skip, batch_size, start_batch=0):
        self.data_source = data_source
        self.indices = list(range(len(data_source)))
        self.skip = skip
        random.shuffle(self.indices)
        self.start_batch = start_batch
        self.start_index = self.start_batch * batch_size

        # Track completed indices
        self.completed_indices = set()  # Set to track completed indices
        self.remaining_indices = set(self.indices)  # Start with all indices as remaining

    def __iter__(self):
        for idx in self.indices[self.start_index:]:
            # Decide randomly whether to skip this index based on the skip probability
            if random.random() > self.skip:  # Skip with the given probability
                yield idx

    def __len__(self):
        return len(self.indices)
    
    def get_sampled_indices(self):
        for idx in self.indices[self.start_index:]:
            # Decide randomly whether to skip this index based on the skip probability
            if random.random() > self.skip:
                self.completed_indices.add(idx)  # Mark this index as completed
                self.remaining_indices.discard(idx)  # Remove from remaining indices
                yield idx
    
    def get_completed_indices(self):
        """
        Return the indices that have been completed (processed).
        """
        return self.completed_indices

    def get_remaining_indices(self):
        """
        Return the indices that are remaining (not processed).
        """
        return self.remaining_indices