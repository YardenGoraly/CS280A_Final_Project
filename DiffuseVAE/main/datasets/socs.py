import torch
from torch.utils.data import Dataset
from joblib import load

import numpy as np
from PIL import Image
import os


class SOCSDataset(Dataset):
    def __init__(self, 
                 data_root,
                 norm=True,
                 transform=None,
                 sequence_length=8,
                 num_sequences=79935,
                 camera_choice=0,
                 ):
        
        self.data_root = data_root
        self.transform = transform
        self.norm = norm

        self.seq_len = sequence_length
        self.num_sequences = num_sequences
        self.camera_choice = camera_choice

    def __len__(self):
        return self.num_sequences * self.seq_len
    
    def __getitem__(self, idx):
        seq_num = idx // self.num_sequences
        seq_idx = idx % self.seq_len
        return self._loaditem(seq_num, seq_idx)

    # Overload for the specific dataset
    # Must return image sequence, viewpoint sequence, and optional time sequence
    def _loaditem(self, seq, idx):
        pass

    def _loaditem(self, seq, idx):
        data_root = self.data_root
        num_frames = self.seq_len
        # data_path = os.path.join(data_root, f'{seq}.npz')
        data_path = os.path.join(data_root, f'{seq}.npz')
        with open(data_path, 'rb') as f:
            data = np.load(f)
            img_seq = data['rgb'][:self.seq_len, self.camera_choice].astype('float32')
            # (Frames, H, W, C) 
            img = img_seq[idx]

            loaded_data = torch.from_numpy(img).permute(2, 0, 1).float()

        return loaded_data

class SOCSReconDataset(Dataset):
    def __init__(self, 
                 data_root,
                 norm=True,
                 transform=None,
                 sequence_length=8,
                 num_sequences=40,
                 camera_choice=0,
                 ):
        
        self.data_root = data_root
        self.transform = transform
        self.norm = norm

        self.seq_len = sequence_length
        self.num_sequences = num_sequences
        self.data = self.process_gif()
        # self.camera_choice = camera_choice

    def __len__(self):
        return len(self.data) * self.seq_len
    
    def __getitem__(self, idx):
        seq_num = idx // self.seq_len
        seq_idx = idx % self.seq_len
        # return self._loaditem(seq_num, seq_idx)
        img = self.data[seq_num][seq_idx]
        loaded_data = torch.from_numpy(img).permute(2, 0, 1).float()
        return loaded_data

    def process_gif(self):
        all_frames = []  # To store frames from all GIFs

        # Iterate over all files in the data_root directory
        filenames = sorted([f for f in os.listdir(self.data_root) if f.lower().endswith(".gif")])
        for filename in filenames:
            gif_path = os.path.join(self.data_root, filename)
            gif = Image.open(gif_path)

            # List to store frames for the current GIF
            frames = []

            # Iterate through the frames in the GIF
            for frame in range(gif.n_frames):
                gif.seek(frame)  # Go to the current frame
                frame_image = gif.convert("RGB")  # Ensure it's in RGB format
                frame_image = frame_image.resize((128, 128))  # Resize the frame to 128x128
                frame_array = np.array(frame_image)  # Convert to numpy array
                frames.append(frame_array)

            # Convert list of frames to a numpy array (e.g., (n_frames, 128, 128, 3))
            frames_np = np.array(frames) / 255.
            # Append the frames of the current GIF to the all_frames list
            all_frames.append(frames_np)
            # Print the shape of the frames (e.g., (8, 128, 128, 3))
            print(f"Processed {filename}: {frames_np.shape}, {len(all_frames)}")

        return all_frames

class SOCSReconSingleDataset(Dataset):
    def __init__(self, 
                 data_root,
                 norm=True,
                 transform=None,
                 sequence_length=8,
                 num_sequences=1,
                 camera_choice=0,
                 ):
        
        self.data_root = data_root
        self.transform = transform
        self.norm = norm

        self.seq_len = sequence_length
        self.num_sequences = num_sequences
        self.data = self.process_gif()
        # self.camera_choice = camera_choice

    def __len__(self):
        return len(self.data) * self.seq_len
    
    def __getitem__(self, idx):
        seq_num = idx // self.seq_len
        seq_idx = idx % self.seq_len
        # return self._loaditem(seq_num, seq_idx)
        img = self.data[seq_num][seq_idx]
        loaded_data = torch.from_numpy(img).permute(2, 0, 1).float()
        return loaded_data

    def process_seq(self):
        all_frames = []  # To store frames from all GIFs

        # Iterate over all files in the data_root directory
        # filenames = sorted([f for f in os.listdir(self.data_root) if f.lower().endswith(".gif")])
        # for filename in filenames:
        #     gif_path = os.path.join(self.data_root, filename)
        #     gif = Image.open(gif_path)

        # List to store frames for the current GIF
        frames = []

        # Iterate through the frames in the GIF
        for frame in range(self.seq_len):
            frames.append(self.data_root[frame])
            # gif.seek(frame)  # Go to the current frame
            # frame_image = gif.convert("RGB")  # Ensure it's in RGB format
            # frame_image = frame_image.resize((128, 128))  # Resize the frame to 128x128
            # frame_array = np.array(frame_image)  # Convert to numpy array
            # frames.append(frame_array)


        # Convert list of frames to a numpy array (e.g., (n_frames, 128, 128, 3))
        frames_np = np.array(frames) / 255.
        # Append the frames of the current GIF to the all_frames list
        all_frames.append(frames_np)
        # Print the shape of the frames (e.g., (8, 128, 128, 3))
        # print(f"Processed {filename}: {frames_np.shape}, {len(all_frames)}")

        return all_frames
