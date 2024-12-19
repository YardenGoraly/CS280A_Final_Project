# Fun with SOCS
## (Unsupervised Object Representations)
## CS 280A Final Project

Here, we include more visual results from our project:

![image](assets/image.png)

### Latent Perturbation

#### Movi-A Data
<img src="assets/perturb/movi/orig.gif" alt="perturb_0" width="128" />
<img src="assets/perturb/movi/y-x.gif" alt="perturb_0" width="128" />
<img src="assets/perturb/movi/yx.gif" alt="perturb_0" width="128" />
<img src="assets/perturb/movi/ytrans.gif" alt="perturb_0" width="128" />
<img src="assets/perturb/movi/glossy.gif" alt="perturb_0" width="128" />
<img src="assets/perturb/movi/opacity.gif" alt="perturb_0" width="128" />
<img src="assets/perturb/movi/ytransobj.gif" alt="perturb_0" width="128" />
<img src="assets/perturb/movi/ytransobj2.gif" alt="perturb_0" width="128" />

#### Real World
<img src="assets/perturb/real/orig.gif" alt="perturb_0" width="128" />
<img src="assets/perturb/real/-ytrans.gif" alt="perturb_0" width="128" />
<img src="assets/perturb/real/xtrans.gif" alt="perturb_0" width="128" />
<img src="assets/perturb/real/ytrans.gif" alt="perturb_0" width="128" />
<img src="assets/perturb/real/size_dec.gif" alt="perturb_0" width="128" />
<img src="assets/perturb/real/size_inc.gif" alt="perturb_0" width="128" />
<img src="assets/perturb/real/opacity.gif" alt="perturb_0" width="128" />
<img src="assets/perturb/real/color.gif" alt="perturb_0" width="128" />

### Diffusion

#### DDPM-A Reconstructions (Movi-A Data)
Base Reconstruction \
<img src="assets/movi_a_recons/0.gif" alt="ddpm_c_0" width="128" />
<img src="assets/movi_a_recons/1.gif" alt="ddpm_c_1" width="128" />
<img src="assets/movi_a_recons/2.gif" alt="ddpm_c_2" width="128" />
<img src="assets/movi_a_recons/3.gif" alt="ddpm_c_3" width="128" />
<img src="assets/movi_a_recons/4.gif" alt="ddpm_c_4" width="128" />
<img src="assets/movi_a_recons/5.gif" alt="ddpm_c_5" width="128" />
<img src="assets/movi_a_recons/6.gif" alt="ddpm_c_6" width="128" />
<img src="assets/movi_a_recons/7.gif" alt="ddpm_c_7" width="128" />

T=8 Denoising (more lossy) \
![ddpm_a_0](assets/trial28/0.gif)
![ddpm_a_1](assets/trial28/1.gif)
![ddpm_a_2](assets/trial28/2.gif)
![ddpm_a_3](assets/trial28/3.gif)
![ddpm_a_4](assets/trial28/4.gif)
![ddpm_a_5](assets/trial28/5.gif)
![ddpm_a_6](assets/trial28/6.gif)
![ddpm_a_7](assets/trial28/7.gif)

T=6 Denoising (closer reconstruction) \
![ddpm_a_0](assets/trial29/0.gif)
![ddpm_a_1](assets/trial29/1.gif)
![ddpm_a_2](assets/trial29/2.gif)
![ddpm_a_3](assets/trial29/3.gif)
![ddpm_a_4](assets/trial29/4.gif)
![ddpm_a_5](assets/trial29/5.gif)
![ddpm_a_6](assets/trial29/6.gif)
![ddpm_a_7](assets/trial29/7.gif)

T=4 Denoising (clearest) \
![ddpm_a_0](assets/trial30/0.gif)
![ddpm_a_1](assets/trial30/1.gif)
![ddpm_a_2](assets/trial30/2.gif)
![ddpm_a_3](assets/trial30/3.gif)
![ddpm_a_4](assets/trial30/4.gif)
![ddpm_a_5](assets/trial30/5.gif)
![ddpm_a_6](assets/trial30/6.gif)
![ddpm_a_7](assets/trial30/7.gif)

#### DDPM-C Reconstructions (Movi-A Data)
T=4 Denoising (better than DDPM-A) \
![ddpm_c_0](assets/trial1c/0.gif)
![ddpm_c_1](assets/trial1c/1.gif)
![ddpm_c_2](assets/trial1c/2.gif)
![ddpm_c_3](assets/trial1c/3.gif)
![ddpm_c_4](assets/trial1c/4.gif)
![ddpm_c_5](assets/trial1c/5.gif)
![ddpm_c_6](assets/trial1c/6.gif)
![ddpm_c_7](assets/trial1c/7.gif)

T=8 Denoising (minor changes to objects) \
![ddpm_c_0](assets/trial2c/0.gif)
![ddpm_c_1](assets/trial2c/1.gif)
![ddpm_c_2](assets/trial2c/2.gif)
![ddpm_c_3](assets/trial2c/3.gif)
![ddpm_c_4](assets/trial2c/4.gif)
![ddpm_c_5](assets/trial2c/5.gif)
![ddpm_c_6](assets/trial2c/6.gif)
![ddpm_c_7](assets/trial2c/7.gif)

#### DDPM-C Reconstructions (Real World)
Ground Truth \
![real_gt_0](assets/real_gt/0.gif)
![real_gt_1](assets/real_gt/1.gif)
![real_gt_2](assets/real_gt/2.gif)
![real_gt_3](assets/real_gt/3.gif)
![real_gt_4](assets/real_gt/4.gif)
![real_gt_5](assets/real_gt/5.gif)
![real_gt_6](assets/real_gt/6.gif)
![real_gt_7](assets/real_gt/7.gif)

Base Reconstruction \
<img src="assets/real_recons/val_0_375700.gif" alt="ddpm_c_0" width="128" />
<img src="assets/real_recons/val_14_375700.gif" alt="ddpm_c_3" width="128" />
<img src="assets/real_recons/val_3_375700.gif" alt="ddpm_c_4" width="128" />
<img src="assets/real_recons/val_12_375700.gif" alt="ddpm_c_5" width="128" />
<img src="assets/real_recons/val_13_375700.gif" alt="ddpm_c_7" width="128" />
<img src="assets/real_recons/val_2_375700.gif" alt="ddpm_c_2" width="128" />
<img src="assets/real_recons/val_15_375700.gif" alt="ddpm_c_6" width="128" />
<img src="assets/real_recons/val_16_375700.gif" alt="ddpm_c_1" width="128" />

T=4 Denoising (pretty blurry) \
![ddpm_c_0](assets/trial4c/0.gif)
![ddpm_c_1](assets/trial4c/1.gif)
![ddpm_c_2](assets/trial4c/2.gif)
![ddpm_c_3](assets/trial4c/3.gif)
![ddpm_c_4](assets/trial4c/4.gif)
![ddpm_c_5](assets/trial4c/5.gif)
![ddpm_c_6](assets/trial4c/6.gif)
![ddpm_c_7](assets/trial4c/7.gif)

T=40 Denoising (complete loss) \
![ddpm_c_0](assets/trial8c/0.gif)
![ddpm_c_1](assets/trial8c/1.gif)
![ddpm_c_2](assets/trial8c/2.gif)
![ddpm_c_3](assets/trial8c/3.gif)
![ddpm_c_4](assets/trial8c/4.gif)
![ddpm_c_5](assets/trial8c/5.gif)
![ddpm_c_6](assets/trial8c/6.gif)
![ddpm_c_7](assets/trial8c/7.gif)

#### DeepFloyd Reconstructions (Real World)
T=24 (loss of object) \
<img src="assets/trial_df/0.gif" alt="df_0" width="100" />
<img src="assets/trial_df/1.gif" alt="df_1" width="100" />
<img src="assets/trial_df/2.gif" alt="df_2" width="100" />
<img src="assets/trial_df/4.gif" alt="df_4" width="100" />
<img src="assets/trial_df/7.gif" alt="df_7" width="100" />


### Teacher-Student Model

#### With vs. Without Mask Loss
<img src="assets/teacher/mask.gif" alt="teacher_0" width="128" />
<img src="assets/teacher/no_mask.gif" alt="teacher_0" width="128" />
