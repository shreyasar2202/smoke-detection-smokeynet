# pytorch-lightning-smoke-detection

Created: 2021 Anshuman Dewangan

This repository uses [Pytorch Lightning](https://www.pytorchlightning.ai/) for wildfire smoke detection. Supports image classification and object detection models.

Visualization of model performance:

![](wildfire-smoke-detection.mp4)

# Quick Setup & Usage
## Initial Setup
1. Clone directory:
```bash
git clone https://gitlab.nrp-nautilus.io/anshumand/pytorch-lightning-smoke-detection.git
cd pytorch-lightning-smoke-detection
```

2. Edit ```torch-gpu.yaml``` to replace instances of ```anshumand``` to ```<your_username>```

## Usage
1. Create Kubernetes container: ```kubectl create -f torch-gpu.yaml```
2. Confirm pod is running: ```kubectl get pods```
3. Forward port to local machine: ```kubectl port-forward deployment/torch-gpu-anshumand 8888:8888```
4. Open Jupyter Lab on browser (password = ```digits```): ```http://127.0.0.1:8888/```
5. (Optional) To SSH into virtual server from local terminal, use: ```kubectl exec -it deployment/torch-gpu-anshumand -- /bin/bash```
6. (Optional) If Jupyter Lab server goes down for some reason, restart it with: ```jupyter lab --port=8888 --no-browser --ip=0.0.0.0 --allow-root --NotebookApp.password="$(echo digits | python3 -c 'from notebook.auth import passwd;print(passwd(input()))')"  --ContentsManager.allow_hidden=True```
7. Once Jupyter Lab is opened on your browser, it is recommended to use tmux by entering the following command in Terminal: ```tmux```
8. In tmux, run the following command to copy files to the root directory: ```./setup_files.sh```
9. In a different terminal window, it is recommend to start Tensorboard to access logs: ```tensorboard --logdir ./lightning_logs```
10. Once files are setup, run the following command to run an experiment: ```./run_train.sh```

# Documentation

## Kubernetes
**Relevant Files:**
- ```Dockerfile```: Code that is run when the container is set up. Uses ```nvidia/cuda``` module as base. Installs system packages and necessary Python packages. Starts Jupyter Lab server at port 8888
- ```torch-gpu.yaml```: Parameters with which the container is initialized
- ```.gitlab-ci.yml```: YAML file that allows CI/CD with GitLab to automatically build new container with Dockerfile changes


## Data Setup
**Relevant Files:**
- ```dynamic_dataloader.py```: Includes datamodule and dataloader for training. ```DynamicDataModule.prepare_data()``` needs only to be run once.
- ```./data/metadata.pkl```: Dictionary generated by ```DynamicDataModule.prepare_data()``` that includes helpful information about the data. See ```dynamic_dataloader.py``` for the full list of keys.

**Relevant Directories:**
- ```/userdata/kerasData/data/new_data/raw_images```: location of raw images
- ```/userdata/kerasData/data/new_data/drive_clone```: location of raw XML labels
- ```/userdata/kerasData/data/new_data/drive_clone_labels```: location of preprocessed labels from ```prepare_data()```
- ```./data/final_split/```: data split where train = all the labeled fires and val/test is a random split of unlabeled fires (with night fires removed).
- ```./data/split1/``` and ```./data/split1/```: random train/val/test split of all labeled fires only
- ```./data/night_fires.txt```: list of fires that occur during the night (so they can be removed)
- ```./data/omit_mislabeled.txt```: list of images that are supposed to be labeled but do not have bbox labels


## Model Setup
**Relevant Files:**
- ```model_components.py```: Different torch models to use with ```main_model.py```. Each model has its own forward pass and loss function.
- ```main_model.py```: Main model to use with ```lightning_module.py```. Chains forward passes and sums loss functions from individual model_components

**Models:**
Models are created with model_components that can be chained together using the ```--model-type-list``` command line argument. Intermediate supervision from tile_labels or image_labels provides additional feedback to each model_component. Models can be one of five types:
1. RawToTile: Raw inputs -> tile predictions
2. RawToImage: Raw inputs -> image predictions
3. TileToTile: Tile predictions -> tile predictions
4. TileToImage: Tile predictins -> image predictions
5. ImageToImage: Image predictions -> image predictions


## Training
**Relevant Files:**
- ```main.py```: Kicks off training and evaluation. Contains many command line arguments for hyperparameters. 
- ```lightning_module.py```: PyTorch Lightning LightningModule that defines optimizers, training step and metrics.
- ```run_train.sh```: Used to easily start training from main.py with command line arguments.

**Steps to Run:**
To run training, use ```./run_train.sh```. You can check ```main.py``` for a full list of tunable hyperparameters as command line arguments.


## Logging
**Relevant Directories:**
- ```./lightning_logs/``` (currently not pushed to repo): Automatically generated each run where logs & checkpoints are saved
- ```./saved_logs/``` (currently not pushed to repo): It is suggested to move logs you want to save long-term in this directory
- ```visual_analysis.ipynb```: iPython notebook to visualize errors

**Steps to Access:**
Logs can be accessed using Tensorboard: ```tensorboard --logdir ./lightning_logs```

# Other Stuff

## Training Tricks
The following steps can increase training speed by 2-5x:
1. Copy the dataset to your home directory (```/root/``` for admin and ```/home/``` for user)
2. Add ```dshm``` volume to YAML file. Also recommended to specify GPU type to ```1080Ti``` or ```2080Ti```, use 4 CPUs, aand set memory=12GB. See torch-gpu.yaml for an example.
3. Set num_workers=4 (equal to the # of CPUs and 4x # of GPUs) in dataloader

# Data Download & Preparation

## Download Data
1. Run ```./scripts/download_raw_data.sh``` to download raw images from the [HPWREN website](http://hpwren.ucsd.edu/HPWREN-FIgLib/HPWREN-FIgLib-Data/) to ```/userdata/kerasData/data/new_data/raw_images_new/``` directory
2. Follow the instructions at the bottom of this [Google Doc](https://docs.google.com/document/d/14cnRoZ9VkYk8y0Wf3__uGAvDXWaGv1dwGrWtTYe75q0/edit?usp=sharing) to download the bounding box and contour annotation labels