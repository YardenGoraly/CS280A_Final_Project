# CS280A_Final_Project

To generate dataset for main.py and analysis.py, run dataset_generation_movi.py on data with the following structure:
example_folder/train
    video_0
        frame0.png
        frame1.png
        ...
        frame8.png
    video_1
    ...

Here are example scripts: 
python main.py --num_train_seq=80000 --num_epochs=-1 --dataset=movi --data_root=[path to npz folder] --cameras=0 --no_viewpoint --batch_size 8 --lr=1e-4

python analysis.py logs/SOCS/version_5 --split=train --gpu 2 --idx 2832 --video_format=none --plot_latent_perturbations=True --string_append 0 


