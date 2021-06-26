"""
Created by: Anshuman Dewangan
Date: 2021

Description: Kicks off training and evaluation. Contains many command line arguments for hyperparameters. 
"""

# Torch imports
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

# Other package imports
from argparse import ArgumentParser
import datetime

# File imports
from lightning_module import LightningModule
from main_model import MainModel
from dynamic_dataloader import DynamicDataModule, DynamicDataloader
import util_fns

IS_DEBUG = False

#####################
## Argument Parser
#####################

# All args recorded as hyperparams. Recommended not to use unnecessary args
parser = ArgumentParser(description='Takes raw wildfire images and saves tiled images')

# Experiment args
parser.add_argument('--experiment-name', type=str, default=None,
                    help='(Optional) Name for experiment')
parser.add_argument('--experiment-description', type=str, default=None,
                    help='(Optional) Short description of experiment that will be saved as a hyperparam')

# Path args
parser.add_argument('--raw-data-path', type=str, default='/userdata/kerasData/data/new_data/raw_images',
                    help='Path to raw images.')
parser.add_argument('--labels-path', type=str, default='/userdata/kerasData/data/new_data/drive_clone_labels',
                    help='Path to processed XML labels.')
parser.add_argument('--raw-labels-path', type=str, default='/userdata/kerasData/data/new_data/drive_clone',
                    help='Path to raw XML labels.')
parser.add_argument('--metadata-path', type=str, default='./data/metadata.pkl',
                    help='Path to metadata.pkl.')
parser.add_argument('--train-split-path', type=str, default=None,
                    help='(Optional) Path to txt file with train image paths. Only works if train, val, and test paths are provided.')
parser.add_argument('--val-split-path', type=str, default=None,
                    help='(Optional) Path to txt file with val image paths. Only works if train, val, and test paths are provided.')
parser.add_argument('--test-split-path', type=str, default=None,
                    help='(Optional) Path to txt file with test image paths. Only works if train, val, and test paths are provided.')
parser.add_argument('--checkpoint-path', type=str, default=None,
                    help='(Optional) Path to checkpoint to load.')

# Dataloader args
parser.add_argument('--train-split-size', type=int, default=0.7,
                    help='% of data to split for train.')
parser.add_argument('--test-split-size', type=int, default=0.15,
                    help='% of data to split for test.')
parser.add_argument('--batch-size', type=int, default=1,
                    help='Batch size for training.')
parser.add_argument('--num-workers', type=int, default=0,
                    help='Number of workers for dataloader.')
parser.add_argument('--series-length', type=int, default=1,
                    help='Number of sequential video frames to process during training.')
parser.add_argument('--time-range-min', type=int, default=-2400,
                    help='Start time of fire images to consider during training. ')
parser.add_argument('--time-range-max', type=int, default=2400,
                    help='End time of fire images to consider during training (inclusive).')

parser.add_argument('--image-height', type=int, default=1536,
                    help='Desired resize height of image.')
parser.add_argument('--image-width', type=int, default=2016,
                    help='Desired resize width of image.')
parser.add_argument('--crop-height', type=int, default=1120,
                    help='Desired height after cropping.')
parser.add_argument('--tile-size', type=int, default=224,
                    help='Height and width of tile.')
parser.add_argument('--smoke-threshold', type=int, default=10,
                    help='Number of pixels of smoke to consider tile positive.')

parser.add_argument('--flip-augment', action='store_true',
                    help='Enables data augmentation with horizontal flip.')
parser.add_argument('--blur-augment', action='store_true',
                    help='Enables data augmentation with Gaussian blur.')

# Model args
parser.add_argument('--model-type', type=str, default='ResNet50',
                    help='Type of model to use for training.')
parser.add_argument('--learning-rate', type=float, default=0.001,
                    help='Learning rate for training.')
parser.add_argument('--no-lr-schedule', action='store_true',
                    help='Disables ReduceLROnPlateau learning rate scheduler. See PyTorch Lightning docs for more details.')
parser.add_argument('--no-pretrain-backbone', action='store_true',
                    help='Disables pretraining of backbone.')
parser.add_argument('--no-freeze-backbone', action='store_true',
                    help='Disables freezing of layers on pre-trained backbone.')

parser.add_argument('--bce-pos-weight', type=float, default=10,
                    help='Weight for positive class for BCE loss for tiles.')
parser.add_argument('--focal-alpha', type=float, default=0.25,
                    help='Alpha for focal loss.')
parser.add_argument('--focal-gamma', type=float, default=2,
                    help='Gamma for focal loss.')

# Training args
parser.add_argument('--min-epochs', type=int, default=10,
                    help='Min number of epochs to train for.')
parser.add_argument('--max-epochs', type=int, default=50,
                    help='Max number of epochs to train for.')
parser.add_argument('--no-auto-lr-find', action='store_true',
                    help='Disables auto learning rate finder. See PyTorch Lightning docs for more details.')
parser.add_argument('--no-early-stopping', action='store_true',
                    help='Disables early stopping based on validation loss. See PyTorch Lightning docs for more details.')
parser.add_argument('--no-sixteen-bit', action='store_true',
                    help='Disables use of 16-bit training to reduce memory. See PyTorch Lightning docs for more details.')
parser.add_argument('--no-stochastic-weight-avg', action='store_true',
                    help='Disables stochastic weight averaging. See PyTorch Lightning docs for more details.')
parser.add_argument('--gradient-clip-val', type=float, default=0,
                    help='Clips gradients to prevent vanishing or exploding gradients. See PyTorch Lightning docs for more details.')
parser.add_argument('--accumulate-grad-batches', type=int, default=1,
                    help='Accumulate multiple batches before calling loss.backward() to increase effective batch size. See PyTorch Lightning docs for more details.')

    
#####################
## Main
#####################
    
def main(# Path args
        raw_data_path, 
        labels_path, 
        raw_labels_path=None,
        metadata_path=None,
        train_split_path=None, 
        val_split_path=None, 
        test_split_path=None,
    
        # Experiment args
        experiment_name=None,
        experiment_description=None,
        parsed_args=None,

        # Dataloader args
        train_split_size=0.7,
        test_split_size=0.15,
        batch_size=1, 
        num_workers=0, 
        series_length=1, 
        time_range=(-2400,2400), 
        image_dimensions=(1536, 2016),
        crop_height=1344,
        tile_dimensions=(224,224),
        smoke_threshold=10,
        flip_augment=False,
        blur_augment=False,

        # Model args
        model_type='ResNet50',
        learning_rate=0.001,
        lr_schedule=True,
        pretrain_backbone=True,
        freeze_backbone=True,
    
        bce_pos_weight=10,
        focal_alpha=0.25,
        focal_gamma=2,

        # Trainer args 
        min_epochs=10,
        max_epochs=50,
        auto_lr_find=True,
        early_stopping=True,
        sixteen_bit=True,
        stochastic_weight_avg=True,
        gradient_clip_val=0,
        accumulate_grad_batches=1,

        # Checkpoint args
        checkpoint_path=None,
        checkpoint=None):
        
    try:
        if not IS_DEBUG: util_fns.send_fb_message(f'Experiment {experiment_name} Started...')
        
        ### Initialize data_module ###
        data_module = DynamicDataModule(
            # Path args
            raw_data_path=raw_data_path,
            labels_path=labels_path,
            raw_labels_path=raw_labels_path,
            metadata_path=metadata_path,
            train_split_path=train_split_path,
            val_split_path=val_split_path,
            test_split_path=test_split_path,

            # Dataloader args
            train_split_size=train_split_size,
            test_split_size=test_split_size,
            batch_size=batch_size,
            num_workers=num_workers,
            series_length=series_length,
            time_range=time_range,
        
            image_dimensions=image_dimensions,
            crop_height=crop_height,
            tile_dimensions=tile_dimensions,
        
            flip_augment=flip_augment,
            blur_augment=blur_augment)
        
        ### Initialize Model ###
        main_model = MainModel(
                         model_type=model_type,
                         series_length=series_length, 
                         freeze_backbone=freeze_backbone, 
                         pretrain_backbone=pretrain_backbone,
                         bce_pos_weight=bce_pos_weight,
                         focal_alpha=focal_alpha, 
                         focal_gamma=focal_gamma)
        
        ### Initialize LightningModule ###
        if checkpoint_path and checkpoint:
            lightning_module = LightningModule.load_from_checkpoint(
                                   checkpoint_path,
                                   model=main_model,
                                   learning_rate=learning_rate,
                                   lr_schedule=lr_schedule,
                                   series_length=series_length,
                                   parsed_args=parsed_args)
        else:
            lightning_module = LightningModule(
                                   model=main_model,
                                   learning_rate=learning_rate,
                                   lr_schedule=lr_schedule,
                                   series_length=series_length,
                                   parsed_args=parsed_args)
            
        ### Implement EarlyStopping & Other Callbacks ###
        early_stop_callback = EarlyStopping(
           monitor='val/loss',
           min_delta=0.00,
           patience=4,
           verbose=True)

        checkpoint_callback = ModelCheckpoint(monitor='val/loss', save_last=True)
        
        callbacks = []
        if early_stopping: 
            callbacks.append(early_stop_callback)
        if not IS_DEBUG: 
            callbacks.append(checkpoint_callback)

        ### Initialize Trainer ###

        # Initialize logger 
        logger = TensorBoardLogger("lightning_logs/", 
                                   name=experiment_name, 
                                   log_graph=True,
                                   version=None)
        
        # Set up data_module and save train/val/test splits
        data_module.setup(log_dir=logger.log_dir if not IS_DEBUG else None)

        trainer = pl.Trainer(
            # Trainer args
            min_epochs=min_epochs,
            max_epochs=max_epochs,
            auto_lr_find=auto_lr_find,
            callbacks=callbacks,
            precision=16 if sixteen_bit else 32,
            stochastic_weight_avg=stochastic_weight_avg,
            gradient_clip_val=gradient_clip_val,
            accumulate_grad_batches=accumulate_grad_batches,
            
            # Other args
            resume_from_checkpoint=checkpoint_path,
            logger=logger if not IS_DEBUG else False,
            log_every_n_steps=1,
#             val_check_interval=0.5,

            # Dev args
#             fast_dev_run=True, 
#             overfit_batches=1,
#             limit_train_batches=1,
#             limit_val_batches=1,
#             limit_test_batches=0.25,
#             track_grad_norm=2,
#             weights_summary='full',
#             profiler="simple", # "advanced" "pytorch"
#             log_gpu_memory=True,
            gpus=1)
        
        ### Training & Evaluation ###
        # Auto find learning rate
        if auto_lr_find:
            trainer.tune(lightning_module, datamodule=data_module)

        # Train the model
        trainer.fit(lightning_module, datamodule=data_module)

        # Evaluate the best model on the test set
        trainer.test(lightning_module, datamodule=data_module)

        if not IS_DEBUG: util_fns.send_fb_message(f'Experiment {experiment_name} Complete')
    except Exception as e:
        if not IS_DEBUG: util_fns.send_fb_message(f'Experiment {args.experiment_name} Failed. Error: ' + str(e))
        raise(e) 
    
    
if __name__ == '__main__':
    args = parser.parse_args()
    
    if not args.checkpoint_path:
        main(# Path args
            raw_data_path=args.raw_data_path, 
            labels_path=args.labels_path, 
            raw_labels_path=args.raw_labels_path,
            metadata_path=args.metadata_path,
            train_split_path=args.train_split_path, 
            val_split_path=args.val_split_path, 
            test_split_path=args.test_split_path,

            # Experiment args
            experiment_name=args.experiment_name,
            experiment_description=args.experiment_description,
            parsed_args=args,

            # Dataloader args
            train_split_size=args.train_split_size,
            test_split_size=args.test_split_size,
            batch_size=args.batch_size, 
            num_workers=args.num_workers, 
            series_length=args.series_length, 
            time_range=(args.time_range_min,args.time_range_max), 

            image_dimensions=(args.image_height, args.image_width),
            crop_height=args.crop_height,
            tile_dimensions=(args.tile_size, args.tile_size),
            smoke_threshold=args.smoke_threshold,

            flip_augment=args.flip_augment,
            blur_augment=args.blur_augment,

            # Model args
            model_type=args.model_type,
            learning_rate=args.learning_rate,
            lr_schedule=not args.no_lr_schedule,
            pretrain_backbone=not args.no_pretrain_backbone,
            freeze_backbone=not args.no_freeze_backbone,
            focal_alpha=args.focal_alpha,
            focal_gamma=args.focal_gamma,

            # Trainer args
            min_epochs=args.min_epochs,
            max_epochs=args.max_epochs,
            auto_lr_find=not args.no_auto_lr_find,
            early_stopping=not args.no_early_stopping,
            sixteen_bit=not args.no_sixteen_bit,
            stochastic_weight_avg=not args.no_stochastic_weight_avg,
            gradient_clip_val=args.gradient_clip_val,
            accumulate_grad_batches=args.accumulate_grad_batches)
    else:
        # Load from checkpoint
        checkpoint = torch.load(args.checkpoint_path)

        main(# Path args
            raw_data_path=args.raw_data_path, 
            labels_path=args.labels_path, 
            raw_labels_path=args.raw_labels_path,
            metadata_path=args.metadata_path,
            train_split_path=args.train_split_path, 
            val_split_path=args.val_split_path, 
            test_split_path=args.test_split_path,

            # Experiment args
            experiment_name=checkpoint['hyper_parameters']['experiment_name'],
            experiment_description=checkpoint['hyper_parameters']['experiment_description'],
            parsed_args=checkpoint['hyper_parameters'],

            # Dataloader args
            train_split_size=checkpoint['hyper_parameters']['train_split_size'],
            test_split_size=checkpoint['hyper_parameters']['test_split_size'],
            batch_size=checkpoint['hyper_parameters']['batch_size'], 
            num_workers=checkpoint['hyper_parameters']['num_workers'], 
            series_length=checkpoint['hyper_parameters']['series_length'], 
            time_range=(checkpoint['hyper_parameters']['time_range_min'],checkpoint['hyper_parameters']['time_range_max']), 

            image_dimensions=(checkpoint['hyper_parameters']['image_height'], checkpoint['hyper_parameters']['image_width']),
            crop_height=checkpoint['hyper_parameters']['crop_height'],
            tile_dimensions=(checkpoint['hyper_parameters']['tile_size'], checkpoint['hyper_parameters']['tile_size']),
            smoke_threshold=checkpoint['hyper_parameters']['smoke_threshold'],

            flip_augment=checkpoint['hyper_parameters']['flip_augment'],
            blur_augment=checkpoint['hyper_parameters']['blur_augment'],
            
            # Model args
            model_type=checkpoint['hyper_parameters']['model_type'],
            learning_rate=checkpoint['hyper_parameters']['learning_rate'],
            lr_schedule=not checkpoint['hyper_parameters']['no_lr_schedule'],
            pretrain_backbone=not checkpoint['hyper_parameters']['no_pretrain_backbone'],
            freeze_backbone=not checkpoint['hyper_parameters']['no_freeze_backbone'],
            focal_alpha=checkpoint['hyper_parameters']['focal_alpha'],
            focal_gamma=checkpoint['hyper_parameters']['focal_gamma'],
            
            # Trainer args
            min_epochs=checkpoint['hyper_parameters']['min_epochs'],
            max_epochs=checkpoint['hyper_parameters']['max_epochs'],
            auto_lr_find=not checkpoint['hyper_parameters']['no_auto_lr_find'],
            early_stopping=not checkpoint['hyper_parameters']['no_early_stopping'],
            sixteen_bit=not checkpoint['hyper_parameters']['no_sixteen_bit'],
            stochastic_weight_avg=not checkpoint['hyper_parameters']['no_stochastic_weight_avg'],
            gradient_clip_val=checkpoint['hyper_parameters']['gradient_clip_val'],
            accumulate_grad_batches=checkpoint['hyper_parameters']['accumulate_grad_batches'],
        
            # Checkpoint args
            checkpoint_path=args.checkpoint_path,
            checkpoint=checkpoint)
 
    