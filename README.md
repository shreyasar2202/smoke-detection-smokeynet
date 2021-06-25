# pytorch-lightning-smoke-detection

Created: 2021 Anshuman Dewangan

This repository uses [Pytorch Lightning](https://www.pytorchlightning.ai/) for wildfire smoke detection.

# Setup & Usage
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

# Documentation

## Kubernetes
**Relevant Files:**
- ```Dockerfile```: Code that is run when the container is set up. Uses ```nvidia/cuda``` module as base. Installs system packages and necessary Python packages. Starts Jupyter Lab server at port 8888
- ```torch-gpu.yaml```: Parameters with which the container is initialized
- ```.gitlab-ci.yml```: YAML file that allows CI/CD with GitLab to automatically build new container with Dockerfile changes


## Data Setup
**Relevant Files:**
- ```dynamic_dataloader.py```: Includes datamodule and dataloader for training. ```DynamicDataModule.prepare_data()``` needs only to be run once.
- ```./data/metadata.pkl```: Generated by ```DynamicDataModule.prepare_data()```. See full info below.

**Relevant Directories:**
- ```/userdata/kerasData/data/new_data/raw_images```: location of raw images
- ```/userdata/kerasData/data/new_data/drive_clone```: location of raw XML labels
- ```/userdata/kerasData/data/new_data/drive_clone_labels```: location of preprocessed labels from ```prepare_data()```

**metadata.pkl:**
```metadata.pkl``` is a dictionary generated by ```DynamicDataModule.prepare_data()``` that includes helpful information about the files. Params:
- ```fire_to_images``` (dict): dictionary with fires as keys and list of corresponding images as values
- ```num_fires``` (int): total number of fires in dataset
- ```num_images``` (int): total number of images in dataset
- ```ground_truth_label``` (dict): dictionary with fires as keys and 1 if fire has "+" in its file name
- ```has_xml_label``` (dict): dictionary with fires as keys and 1 if fire has a .xml file associated with it
- ```omit_no_xml``` (list of str): list of images that erroneously do not have XML files for labels
- ```omit_no_bbox``` (list of str): list of images that erroneously do not have loaded bboxes for labels
- ```omit_images_list``` (list of str): union of omit_no_xml and omit_no_bbox


## Training
**Relevant Files:**
- ```main.py```: Kicks off training and evaluation. Contains many command line arguments for hyperparameters. 
- ```lightning_model.py```: Main file with PyTorch Lightning LightningModule. Defines model, forward pass, and training.
- ```models.py```: Different torch models to use with lightning_model.py
- ```run_train.sh```: Used to easily start training from main.py with command line arguments.

**Steps to Run:**
To run training, use ```python3 main.py``` in the command line. You can optionally use ```./run_train.sh```. You can check ```main.py``` for a full list of tunable hyperparameters as command line arguments, but the defaults will be set to give good performance.

**Models:**
Each model has three parts:
1. backbone: Input = [batch_size/arbitrary_value, num_channels=3, height=224, width=224]. Output = [batch_size, num_tiles=54, 1]. This allows for intermediate supervision using tile_labels.
2. middle_layer: Input = [batch_size, num_tiles=54, 1]. Output = [batch_size, num_tiles=54, 1]. This allows to use backbone as input, chain multiple middle layers, and use intermediate supervision using tile_labels.
3. final_layer: Input = [batch_size, num_tiles=54, 1]. Output = [batch_size, 1]. This gives a final prediction per image. 


## Logging
**Relevant Directories:**
- ```./lightning_logs/``` (currently not pushed to repo): Automatically generated each run where logs & checkpoints are saved
- ```./saved_logs/``` (currently not pushed to repo): It is suggested to move logs you want to save long-term in this directory

**Steps to Access:**
Logs can be accessed using Tensorboard: ```tensorboard --logdir ./lightning_logs```


## Archive
**Relevant Files:**
- ```./archive/generate_batched_tiled_data.py```: Code to create tiled & batched data. Useful if you want overlap between tiles
- ```./batched_tiled_dataloader.py```: Dataloader to be used with data generated from ```generate_batched_tiled_data.py```