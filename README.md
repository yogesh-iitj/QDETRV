
## To setup environment
```
  # create new env fsrr
  $ conda create -n qdetr python=3.10.4

  # activate qdetr
  $ conda activate qdetr

  # install pytorch, torchvision
  $ conda install -c pytorch pytorch torchvision
  $ conda install cython scipy

  # install other dependencies
  $ pip install -r requirements.txt
```

## Pre-training
```
# download IMGENET and UCF-101 dataset
# To create the synthetic data for pre-training:
$ python ./dataset/syn_trajectory.py path/to/video_folder path/to/output_csv_annotations path/to/save/process_videos

# To pre-train the model
# set config_pre.py
# set CUDA devices
$ export CUDA_VISIBLE_DEVICES=0,1
# Image-level pretraining
$ python train_qdetr_pre.py
# video-level pertaining
$ python train_qdetrv_pre.py
```
## Training 
```
# To download the images
$ python ./dataset/1_download_images.py

# To filter queries into main categories
$ python ./dataset/2_filter_queries.py

# To create query images and target video pairs for training and testing
$ python ./dataset/3_generate_pairs.py

# To train the model
# set config.py
# set CUDA devices
$ export CUDA_VISIBLE_DEVICES=0,1
# training image-level QDETR
$ python train_qdetr.py
# training video-level QDETR
$ python train_qdetrv.py
```
## Evaluation
```
 # set the paths in config
 $ python eval.py 
```
