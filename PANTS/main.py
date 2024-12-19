import argparse
import numpy as np
import os
import torch
import torch.nn.functional as F

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import Dataset, DataLoader
import faulthandler
import matplotlib.pyplot as plt

from model import SOCS
from util import fourier_embeddings

class SOCSDataset(Dataset):
    def __init__(self, 
                 sequence_length,
                 spatial_patch_hw,
                 data_root,
                 num_sequences=1,
                 decode_pixel_downsample_factor=16,
                 img_dim_hw=(0,0),
                 camera_choice=[1],
                 add_instance_seg=False,
                 num_fourier_bands=10,
                 fourier_sampling_rate=60,
                 no_viewpoint=False):
        
        self.img_dim_hw = img_dim_hw
        self.seq_len = sequence_length
        self.decode_pixel_downsample_factor = decode_pixel_downsample_factor
        self.spatial_patch_hw = spatial_patch_hw
        self.data_root = data_root
        self.num_sequences = num_sequences
        self.camera_choice = camera_choice
        self.add_instance_seg = add_instance_seg
        self.num_fourier_bands = num_fourier_bands
        self.fourier_sampling_rate = fourier_sampling_rate
        self.provide_viewpoint = not no_viewpoint

    def __len__(self):
        return self.num_sequences
    
    def __getitem__(self, idx):
        (item, decode_mask) = self._set_pixels_to_decode(self._loaditem(idx))
        if self.add_instance_seg:
            self._load_instance_seg(idx, item, decode_mask)
        return item

    # Overload for the specific dataset
    # Must return image sequence, viewpoint sequence, and optional time sequence
    def _loaditem(self, idx):
        pass

    def _set_pixels_to_decode(self, item): # teacher_masks \in seq_len x H x W
        """
        Given a loaded sequence, find the positional embeddings for the transformer and the queries for
        the output decoder.
        """
        num_frames = self.seq_len*len(self.camera_choice)
        boolean_column = np.zeros((512))
        all_inds = np.array(np.meshgrid(range(num_frames), range(self.img_dim_hw[0]), range(self.img_dim_hw[1]), indexing='ij'))
        if self.decode_pixel_downsample_factor == 1 or not args.use_teacher_student: 
            # Running original SOCS

            # Choose strided pixels starting at a random value
            random_h_offset = np.random.randint(self.decode_pixel_downsample_factor)
            decode_pixel_h_inds = slice(random_h_offset, self.img_dim_hw[0], self.decode_pixel_downsample_factor)
            random_w_offset = np.random.randint(self.decode_pixel_downsample_factor)
            decode_pixel_w_inds = slice(random_w_offset, self.img_dim_hw[1], self.decode_pixel_downsample_factor)

            # Mask that determines which of the pixels in the input data will be decoded
            decode_mask = np.zeros((num_frames,) + self.img_dim_hw, dtype='bool')
            decode_mask[:, decode_pixel_h_inds, decode_pixel_w_inds] = True

            all_inds = np.array(np.meshgrid(range(num_frames), range(self.img_dim_hw[0]), range(self.img_dim_hw[1]), indexing='ij')) #SAM put this in model.py for choosing the pixel to decode

            decode_inds = all_inds[:, decode_mask].T # /in num_p x 3
        else: 
            # Running teacher-student model
            ground_truth_masks = item['ground_truth_segmentation']
            obj_id = int(np.max(ground_truth_masks))

            decode_mask = np.zeros((num_frames,) + self.img_dim_hw, dtype='bool')
            desired_num_pixels = 64

            # Choose strided pixels starting at a random value
            random_h_offset = np.random.randint(self.decode_pixel_downsample_factor)
            decode_pixel_h_inds = slice(random_h_offset, self.img_dim_hw[0], self.decode_pixel_downsample_factor)
            random_w_offset = np.random.randint(self.decode_pixel_downsample_factor)
            decode_pixel_w_inds = slice(random_w_offset, self.img_dim_hw[1], self.decode_pixel_downsample_factor)

            # Mask that determines which of the pixels in the input data will be decoded
            decode_mask = np.zeros((num_frames,) + self.img_dim_hw, dtype='bool')
            decode_mask[:, decode_pixel_h_inds, decode_pixel_w_inds] = True
            decode_inds = all_inds[:, decode_mask].T # /in num_p x 3
        
            teacher_masks = ground_truth_masks
            teacher_mask_boolean = teacher_masks == obj_id #true if pixel is obj_id, false otherwise
            object_inds = all_inds[:, teacher_mask_boolean].T

            # creates set of tuples where each tuple is an index inside an object 
            obj_set = set(map(tuple, object_inds))

            # array where value at index i is 1 if row i in decode_inds is in object, 0 otherwise
            boolean_column = np.array([tuple(row) in obj_set for row in decode_inds])

        img_seq = item['img_seq']
        viewpoint_seq = item['viewpoint_seq']
        # Remove the last row of the transform matrix if it's stored
        if viewpoint_seq.shape[1] > 12:
            viewpoint_seq = viewpoint_seq[:, :-4]

        if self.provide_viewpoint:
            viewpoint_size = viewpoint_seq.shape[1]
        else:
            viewpoint_size = 1

        # 3D positional information to be used as a query for the decoder
        # The columns are time, y, x, viewpoint_transform
        base_decoder_queries = np.zeros((decode_inds.shape[0], 3 + viewpoint_size))
        base_decoder_queries[:,0] = item['time_seq'][decode_inds[:,0]] # time
        base_decoder_queries[:,1] = (2*decode_inds[:,1] - (self.img_dim_hw[0] - 1)).flatten() / self.img_dim_hw[0] # y
        base_decoder_queries[:,2] = (2*decode_inds[:,2] - (self.img_dim_hw[1] - 1)).flatten() / self.img_dim_hw[1] # x
        if self.provide_viewpoint:
            base_decoder_queries[:,3:] = viewpoint_seq[decode_inds[:,0]] # viewpoint transform
        # If not providing the camera transform matrix, recover the index of the camera to provide instead
        else:
            num_viewpoints = len(self.camera_choice)
            num_timepoints = self.seq_len
            camera_inds = np.unravel_index(decode_inds[:,0], (num_timepoints, num_viewpoints))[1]
            base_decoder_queries[:,3] = np.ones(decode_inds.shape[0]) * camera_inds

        decoder_queries = fourier_embeddings(base_decoder_queries, self.num_fourier_bands, self.fourier_sampling_rate)

        # Prepare the chunk labels for the first transformer
        base_patch_embeddings = np.zeros((num_frames, self.spatial_patch_hw[0], self.spatial_patch_hw[1], 3 + viewpoint_size))

        for i in range(num_frames): # frames
            time_offset = item['time_seq'][i]
            if self.provide_viewpoint:
                view_offset = viewpoint_seq[i]
            else:
                view_offset = [np.unravel_index(i, (num_timepoints, num_viewpoints))[1]]
            for j in range(self.spatial_patch_hw[0]): # height
                patch_y_offset = ((2*j) / (self.spatial_patch_hw[0] - 1)) - 1
                for k in range(self.spatial_patch_hw[1]): # width
                    patch_x_offset = ((2*k) / (self.spatial_patch_hw[1] - 1)) - 1
                    base_patch_embeddings[i,j,k] = np.array([time_offset, patch_y_offset, patch_x_offset, *view_offset])

        patch_positional_embeddings = fourier_embeddings(base_patch_embeddings, self.num_fourier_bands, self.fourier_sampling_rate)

        # If we have access to the full ground truth, we can include it
        if 'full_segmentation' in item.keys():
            instance_mask = item['full_segmentation']
        else:
            instance_mask = np.zeros_like(img_seq)

        data = dict(
            img_seq = img_seq.astype('float32'),
            decode_dims = np.array([num_frames, 
                           self.img_dim_hw[0] // self.decode_pixel_downsample_factor,
                           self.img_dim_hw[1] // self.decode_pixel_downsample_factor]),
            ground_truth_rgb = img_seq[decode_mask],
            patch_positional_embeddings = patch_positional_embeddings.astype('float32'),
            decoder_queries = decoder_queries.astype('float32'),
            in_object_array = boolean_column,
            instance_mask = instance_mask,
        )

        if 'bc_waypoints' in item:
            data['bc_waypoints'] = item['bc_waypoints']
        if 'bc_mask' in item:
            data['bc_mask'] = item['bc_mask']

        return (data, decode_mask)

class LocalDataset(SOCSDataset):
    def _loaditem(self, idx, data_root=None):
        data_root = data_root if data_root is not None else self.data_root
        num_frames = self.seq_len*len(self.camera_choice)
        data_path = os.path.join(data_root, f'{idx}.npz')
        with open(data_path, 'rb') as f:
            data = np.load(f, allow_pickle=True)
            img_seq = data['rgb'][:self.seq_len, self.camera_choice]
            img_seq = img_seq.reshape((num_frames,) + img_seq.shape[2:]).astype('float32')
            viewpoint_seq = data['viewpoint_transform'][:self.seq_len, self.camera_choice]
            viewpoint_seq = viewpoint_seq.reshape((num_frames,) + viewpoint_seq.shape[2:])
            time_seq = data['time'][:self.seq_len].flatten()
            if 'segmentation' in data.keys():
                ground_truth_segmentation = data['segmentation']
            else:
                 ground_truth_segmentation = None
            print('HEEEEEERE', data.keys())
            if 'full_segmentation' in data.keys():
                full_segmentation = (data['full_segmentation'].sum(3) > 0).astype(int)
            else:
                full_segmentation = np.zeros_like(img_seq)
            if 'teacher_masks' in data.keys():
                teacher_masks_seq = np.expand_dims(data['teacher_masks'], axis=1)
            else:
                teacher_masks_seq = None

            loaded_data = dict(img_seq=img_seq,
                               ground_truth_segmentation=ground_truth_segmentation,
                               viewpoint_seq=viewpoint_seq,
                               time_seq=time_seq,
                               teacher_masks_seq=teacher_masks_seq,
                               full_segmentation=full_segmentation
                               )

        return loaded_data

    def _load_instance_seg(self, idx, item, decode_mask, data_root=None):
        num_frames = len(self.camera_choice)*self.seq_len
        data_root = data_root if data_root is not None else self.data_root
        data_path = os.path.join(data_root, f'{idx}.npz')
        with open(data_path, 'rb') as f:
            data = np.load(f)
            if 'instance_seg' in data:
                instance_segs = data['instance_seg']
                instance_segs = instance_segs.reshape((num_frames,) + instance_segs.shape[2:])[decode_mask]
            else:
                instance_segs = np.zeros(item['img_seq'].shape[:-1])

        instance_masks = np.zeros(instance_segs.shape, dtype='bool')
        instance_masks[np.where(instance_segs != 0)[0]] = True

        instances = np.unique(instance_segs[instance_masks])
        num_instances = len(instances)
        if num_instances > 0:
            instance_oh = np.zeros(instance_masks.shape + (num_instances,))
            for i in range(num_instances):
                single_mask = np.where(instance_segs == instances[i])[0]
                instance_oh[:, i][single_mask] = 1
        else:
            instance_oh = np.zeros(0)

        item['instance_oh'] = instance_oh
        item['instance_mask'] = instance_masks

# A dataset class designed to only load a specified subset of the full dataset
class InferenceDataset(LocalDataset):
    def set_indices(self, indices):
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        (item, decode_mask) = self._set_pixels_to_decode(self._loaditem(self.indices[idx]))
        if self.add_instance_seg:
            self._load_instance_seg(self.indices[idx], item, decode_mask)
        return item

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    faulthandler.enable()

    # Basic training parameters
    parser.add_argument('--name', default='SOCS')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--gpu', type=int, default=[0], nargs='+')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--num_train_seq', type=int, default=40000)
    parser.add_argument('--num_epochs', type=int, default=1000) # -1 for infinite epochs
    parser.add_argument('--data_root', default='data_raw')
    parser.add_argument('--dataset', default='waymo', choices=['waymo', 'movi'])
    parser.add_argument('--lr', type=float, default=1e-4)
    
    # Network hyperparameters
    parser.add_argument('--no_viewpoint', action='store_true')
    parser.add_argument('--num_gaussian_heads', type=int, default=3)
    parser.add_argument('--behavioral_cloning_task', action='store_true')
    parser.add_argument('--sequence_length', type=int, default=8)
    parser.add_argument('--beta', type=float, default=5e-7)
    parser.add_argument('--bc_loss_weight')
    parser.add_argument('--sigma', type=float, default=0.08)
    parser.add_argument('--downsample_factor', type=int, default=16)
    parser.add_argument('--num_patches_height', type=int, default=None) #should be 16
    parser.add_argument('--num_patches_width', type=int, default=None)  #should be 16
    parser.add_argument('--checkpoint_path', default=None)
    parser.add_argument('--decoder_layers', type=int, default=3)
    parser.add_argument('--decoder_size', type=int, default=1536)
    parser.add_argument('--transformer_heads', type=int, default=4)
    parser.add_argument('--transformer_head_size', type=int, default=128)
    parser.add_argument('--transformer_ff_size', type=int, default=1024)
    parser.add_argument('--transformer_layers', type=int, default=3)
    parser.add_argument('--num_object_slots', type=int, default=None)
    parser.add_argument('--object_latent_size', type=int, default=32)
    parser.add_argument('--cameras', type=int, default=[0, 1, 2], nargs='+')
    parser.add_argument('--num_fourier_bands', type=int, default=10)
    parser.add_argument('--fourier_sampling_rate', type=int, default=60)
    parser.add_argument('--use_teacher_student', default=False)
    parser.add_argument('--full_ground_truth_included', default=False)

    args = parser.parse_args()
    batch_size = args.batch_size // len(args.gpu)
    num_frames = len(args.cameras) * args.sequence_length

    torch.manual_seed(args.seed)

    if args.dataset == 'movi':
        #set num patches width and height
        img_dim_hw = (128, 128)
        default_patches_hw = (8, 8) #size of image dimension / 16
        viewpoint_size = 4
        default_num_object_slots = 16

    viewpoint_size *= (1 + 2*args.num_fourier_bands)
    nph = args.num_patches_height if args.num_patches_height is not None else default_patches_hw[0]
    npw = args.num_patches_width if args.num_patches_width is not None else default_patches_hw[1]
    spatial_patch_hw = (nph, npw)
    num_objects = args.num_object_slots if args.num_object_slots is not None else default_num_object_slots

    train_dataloader = DataLoader(LocalDataset(args.sequence_length, 
                                                spatial_patch_hw,
                                                os.path.join(args.data_root, 'train'), 
                                                num_sequences=args.num_train_seq,
                                                img_dim_hw=img_dim_hw,
                                                decode_pixel_downsample_factor=args.downsample_factor,
                                                camera_choice=args.cameras,
                                                no_viewpoint=args.no_viewpoint), 
                                    batch_size=batch_size, shuffle=True, num_workers=32)

    model = SOCS(img_dim_hw=img_dim_hw,
                 embed_dim=args.object_latent_size,
                 beta=args.beta,
                 sigma_x=args.sigma, 
                 viewpoint_size=viewpoint_size,
                 learning_rate=args.lr,
                 num_transformer_layers=args.transformer_layers,
                 num_transformer_heads=args.transformer_heads,
                 transformer_head_dim=args.transformer_head_size,
                 transformer_hidden_dim=args.transformer_ff_size,
                 num_decoder_layers=args.decoder_layers,
                 decoder_hidden_dim=args.decoder_size,
                 num_object_slots=num_objects,
                 spatial_patch_hw=spatial_patch_hw,
                 pixel_downsample_factor=args.downsample_factor,
                 num_fourier_bands=args.num_fourier_bands,
                 fourier_sampling_rate=args.fourier_sampling_rate,
                 cameras=args.cameras,
                 provide_viewpoint=not args.no_viewpoint,
                 num_gaussian_heads=args.num_gaussian_heads,
                 bc_task=args.behavioral_cloning_task,
                 seed=args.seed,
                 dataset_name=args.dataset,
                 dataset_root=args.data_root,
                 sequence_len=args.sequence_length)
    

    logger = TensorBoardLogger(save_dir=os.getcwd(), name=os.path.join('logs', args.name))

    recent_checkpoint_callback = ModelCheckpoint(
        filename="last_{step}",
        every_n_train_steps=100
    )
    historical_checkpoint_callback = ModelCheckpoint(
        save_top_k=-1,
        every_n_train_steps=100000,
        filename="{step}"
    )
    trainer = Trainer(accelerator='gpu',
                      devices=args.gpu,
                      strategy="ddp" if len(args.gpu) > 1 else None,
                      check_val_every_n_epoch=1, 
                      logger=logger, 
                      max_epochs=args.num_epochs,
                      precision=16,
                      callbacks=[recent_checkpoint_callback, historical_checkpoint_callback])
    
    trainer.fit(model, train_dataloader, ckpt_path=args.checkpoint_path)
