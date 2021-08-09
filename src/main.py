"""
Created by: Anshuman Dewangan
Date: 2021

Description: Kicks off training and evaluation. Contains many command line arguments for hyperparameters. 
"""

# Torch imports
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

# Other package imports
from argparse import ArgumentParser, Namespace
import datetime

# File imports
from lightning_module import LightningModule
from main_model import MainModel
from dynamic_dataloader import DynamicDataModule, DynamicDataloader
import util_fns



#####################
## Argument Parser
#####################

# All args recorded as hyperparams
parser = ArgumentParser(description='Takes raw wildfire images and saves tiled images')

# Debug args = 5
parser.add_argument('--is-debug', action='store_true',
                    help='Turns off logging and checkpointing.')
parser.add_argument('--is-test-only', action='store_true',
                    help='Skips training for testing only. Useful when checkpoint loading.')
parser.add_argument('--is-hem-training', action='store_true',
                    help='Enables hard example mining training. Prevents loading Trainer from checkpoint and loads train set exactly as is.')
parser.add_argument('--omit-list', nargs='*',
                    help='List of metadata keys to omit from train/val sets. Options: [omit_mislabeled] [omit_no_xml] [omit_no_bbox] [omit_no_contour]')
parser.add_argument('--omit-images-from-test', action='store_true',
                    help='Omits omit_list_images from the test set.')
parser.add_argument('--mask-omit-images', action='store_true',
                    help='Masks tile predictions for images in omit_list_images.')
parser.add_argument('--is-object-detection', action='store_true',
                    help='Specifies data loader for object detection models.')

# Experiment args = 2
parser.add_argument('--experiment-name', type=str, default=None,
                    help='(Optional) Name for experiment')
parser.add_argument('--experiment-description', type=str, default=None,
                    help='(Optional) Short description of experiment that will be saved as a hyperparam')

# Path args = 4 + 4
parser.add_argument('--raw-data-path', type=str, default='/root/raw_images',
                    help='Path to raw images.')
parser.add_argument('--labels-path', type=str, default='/root/pytorch_lightning_data/drive_clone_numpy',
                    help='Path to processed XML labels.')
parser.add_argument('--raw-labels-path', type=str, default='/userdata/kerasData/data/new_data/drive_clone',
                    help='Path to raw XML labels. Only used to generated processed XML labels.')
parser.add_argument('--metadata-path', type=str, default='./data/metadata.pkl',
                    help='Path to metadata.pkl.')

parser.add_argument('--train-split-path', type=str, default=None,
                    help='(Optional) Path to txt file with train image paths. Only works if train, val, and test paths are provided.')
parser.add_argument('--val-split-path', type=str, default=None,
                    help='(Optional) Path to txt file with val image paths. Only works if train, val, and test paths are provided.')
parser.add_argument('--test-split-path', type=str, default=None,
                    help='(Optional) Path to txt file with test image paths. Only works if train, val, and test paths are provided.')
parser.add_argument('--load-images-from-split', action='store_true',
                    help='If images should be loaded exactly from split (as opposed to fires)')

# Dataloader args = 4 + 4 + 5 + 5 + 5
parser.add_argument('--train-split-size', type=int, default=0.7,
                    help='% of data to split for train.')
parser.add_argument('--test-split-size', type=int, default=0.15,
                    help='% of data to split for test.')
parser.add_argument('--batch-size', type=int, default=1,
                    help='Batch size for training.')
parser.add_argument('--num-workers', type=int, default=4,
                    help='Number of workers for dataloader.')

parser.add_argument('--series-length', type=int, default=4,
                    help='Number of sequential video frames to process during training.')
parser.add_argument('--add-base-flow', action='store_true',
                    help='If series_length > 1, sets the first image as the negative sample at time step -2400.')
parser.add_argument('--time-range-min', type=int, default=-2400,
                    help='Start time of fire images to consider during training. ')
parser.add_argument('--time-range-max', type=int, default=2400,
                    help='End time of fire images to consider during training (inclusive).')

parser.add_argument('--original-height', type=int, default=1536,
                    help='Original height of image.')
parser.add_argument('--original-width', type=int, default=2048,
                    help='Original width of image.')
parser.add_argument('--resize-height', type=int, default=1392,
                    help='Desired resize height of image.')
parser.add_argument('--resize-width', type=int, default=1856,
                    help='Desired resize width of image.')
parser.add_argument('--crop-height', type=int, default=1040,
                    help='Desired height after cropping.')

parser.add_argument('--tile-size', type=int, default=224,
                    help='Height and width of tile.')
parser.add_argument('--tile-overlap', type=int, default=20,
                    help='Amount to overlap each tile.')
parser.add_argument('--no-pre-tile', action='store_false',
                    help='Disables tiling of image in dataloader.')
parser.add_argument('--smoke-threshold', type=int, default=250,
                    help='Number of pixels of smoke to consider tile positive.')
parser.add_argument('--num-tile-samples', type=int, default=0,
                    help='Number of random tile samples per batch for upsampling positives and downsampling negatives. If < 1, then turned off. Recommended: 30.')

parser.add_argument('--no-flip-augment', action='store_false',
                    help='Disables data augmentation with horizontal flip.')
parser.add_argument('--no-resize-crop-augment', action='store_false',
                    help='Disables data augmentation with random resize cropping.')
parser.add_argument('--no-blur-augment', action='store_false',
                    help='Disables data augmentation with Gaussian blur.')
parser.add_argument('--no-color-augment', action='store_false',
                    help='Disables data augmentation with color jitter.')
parser.add_argument('--no-brightness-contrast-augment', action='store_false',
                    help='Disables data augmentation with brightness and contrast jitter.')

# Model args = 5 + 4 + 4
parser.add_argument('--model-type-list', nargs='*',
                    help='Specify the model type through multiple model components.')
parser.add_argument('--pretrain-epochs', nargs='*',
                    help='Specify the number of epochs to pretrain each model component.')
parser.add_argument('--no-intermediate-supervision', action='store_false',
                    help='Disables intermediate supervision for chained models.')
parser.add_argument('--use-image-preds', action='store_true',
                    help='Uses image predictions from linear layers instead of tile preds.')
parser.add_argument('--tile-embedding-size', type=int, default=960,
                    help='Target embedding size to use for tile predictions.')

parser.add_argument('--no-pretrain-backbone', action='store_false',
                    help='Disables pretraining of backbone.')
parser.add_argument('--freeze-backbone', action='store_true',
                    help='Freezes layers on pre-trained backbone.')
parser.add_argument('--backbone-size',  type=str, default='small',
                    help='how big a model to train. Options: [small] [medium] [large]')
parser.add_argument('--backbone-checkpoint-path', type=str, default=None,
                    help='Loads pretrained weights for the backbone from a checkpoint.')

parser.add_argument('--tile-loss-type', type=str, default='bce',
                    help='Type of loss to use for training. Options: [bce], [focal]')
parser.add_argument('--bce-pos-weight', type=float, default=40,
                    help='Weight for positive class for BCE loss for tiles.')
parser.add_argument('--focal-alpha', type=float, default=0.25,
                    help='Alpha for focal loss.')
parser.add_argument('--focal-gamma', type=float, default=2,
                    help='Gamma for focal loss.')

# Optimizer args = 4
parser.add_argument('--optimizer-type', type=str, default='SGD',
                    help='Type of optimizer to use for training. Options: [AdamW] [SGD]')
parser.add_argument('--optimizer-weight-decay', type=float, default=1e-3,
                    help='Weight decay of optimizer.')
parser.add_argument('--learning-rate', type=float, default=1e-3,
                    help='Learning rate for training.')
parser.add_argument('--use-lr-schedule', action='store_true',
                    help='Enables ReduceLROnPlateau learning rate scheduler. See PyTorch Lightning docs for more details.')

# Training args = 8
parser.add_argument('--min-epochs', type=int, default=3,
                    help='Min number of epochs to train for.')
parser.add_argument('--max-epochs', type=int, default=25,
                    help='Max number of epochs to train for.')
parser.add_argument('--no-early-stopping', action='store_false',
                    help='Disables early stopping based on validation loss. See PyTorch Lightning docs for more details.')
parser.add_argument('--early-stopping-patience', type=int, default=3,
                    help='Number of epochs to wait for val loss to increase before early stopping.')
parser.add_argument('--no-sixteen-bit', action='store_false',
                    help='Disables use of 16-bit training to reduce memory. See PyTorch Lightning docs for more details.')
parser.add_argument('--no-stochastic-weight-avg', action='store_false',
                    help='Disables stochastic weight averaging. See PyTorch Lightning docs for more details.')
parser.add_argument('--gradient-clip-val', type=float, default=0,
                    help='Clips gradients to prevent vanishing or exploding gradients. See PyTorch Lightning docs for more details.')
parser.add_argument('--accumulate-grad-batches', type=int, default=1,
                    help='Accumulate multiple batches before calling loss.backward() to increase effective batch size. See PyTorch Lightning docs for more details.')

# Checkpoint args
parser.add_argument('--checkpoint-path', type=str, default=None,
                    help='(Optional) Path to checkpoint to load.')

    
#####################
## Main
#####################
    
def main(# Debug args
        is_debug=False,
        is_test_only=False,
        is_hem_training=False,
        omit_list=None,
        omit_images_from_test=False,
        mask_omit_images=False,
        is_object_detection=False,
        
        # Path args
        raw_data_path=None, 
        labels_path=None, 
        raw_labels_path=None,
        metadata_path=None,
    
        train_split_path=None, 
        val_split_path=None, 
        test_split_path=None,
        load_images_from_split=False,
    
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
        add_base_flow=False, 
        time_range=(-2400,2400), 
    
        original_dimensions=(1536, 2016),
        resize_dimensions=(1536, 2016),
        crop_height=1120,
    
        tile_dimensions=(224,224),
        tile_overlap=0,
        pre_tile=True,
        smoke_threshold=250,
        num_tile_samples=0,
    
        flip_augment=True,
        resize_crop_augment=True,
        blur_augment=True,
        color_augment=True,
        brightness_contrast_augment=True,

        # Model args
        model_type_list=['RawToTile_MobileNet'],
        pretrain_epochs=None,
        intermediate_supervision=True,
        use_image_preds=False,
        tile_embedding_size=960,

        pretrain_backbone=True,
        freeze_backbone=False,
        backbone_size='small',
        backbone_checkpoint_path=None,
    
        tile_loss_type='bce',
        bce_pos_weight=36,
        focal_alpha=0.25,
        focal_gamma=2,

        # Optimizer args
        optimizer_type='SGD',
        optimizer_weight_decay=0.0001,
        learning_rate=0.0001,
        lr_schedule=True,
    
        # Trainer args 
        min_epochs=3,
        max_epochs=50,
        early_stopping=True,
        early_stopping_patience=4,
        sixteen_bit=True,
        stochastic_weight_avg=True,
        gradient_clip_val=0,
        accumulate_grad_batches=1,

        # Checkpoint args
        checkpoint_path=None,
        checkpoint=None):
    
    print("Experiment: ", experiment_name)
    print("- IS_DEBUG: ",  is_debug)
    print("- IS_TEST_ONLY: ", is_test_only)

    ### Initialize data_module ###
    data_module = DynamicDataModule(
        is_hem_training=is_hem_training,
        omit_list=omit_list,
        omit_images_from_test=omit_images_from_test,
        mask_omit_images=mask_omit_images,
        is_object_detection=is_object_detection,
        
        # Path args
        raw_data_path=raw_data_path,
        labels_path=labels_path,
        raw_labels_path=raw_labels_path,
        metadata_path=metadata_path,
        
        train_split_path=train_split_path,
        val_split_path=val_split_path,
        test_split_path=test_split_path,
        load_images_from_split=load_images_from_split,

        # Dataloader args
        train_split_size=train_split_size,
        test_split_size=test_split_size,
        batch_size=batch_size,
        num_workers=num_workers,
        
        series_length=series_length,
        add_base_flow=add_base_flow,
        time_range=time_range,

        original_dimensions=original_dimensions,
        resize_dimensions=resize_dimensions,
        crop_height=crop_height,
        
        tile_dimensions=tile_dimensions,
        tile_overlap=tile_overlap,
        pre_tile=pre_tile,
        smoke_threshold=smoke_threshold,
        num_tile_samples=num_tile_samples,

        flip_augment=flip_augment,
        resize_crop_augment=resize_crop_augment,
        blur_augment=blur_augment,
        color_augment=color_augment,
        brightness_contrast_augment=brightness_contrast_augment)

    ### Initialize MainModel ###
    num_tiles_height, num_tiles_width = util_fns.calculate_num_tiles(resize_dimensions, crop_height, tile_dimensions, tile_overlap)

    main_model = MainModel(
                     # Model args
                     model_type_list=model_type_list,
                     pretrain_epochs=pretrain_epochs,
                     intermediate_supervision=intermediate_supervision,
                     use_image_preds=use_image_preds,
                     tile_embedding_size=tile_embedding_size,

                     tile_loss_type=tile_loss_type,
                     bce_pos_weight=bce_pos_weight,
                     focal_alpha=focal_alpha, 
                     focal_gamma=focal_gamma,

                     freeze_backbone=freeze_backbone, 
                     pretrain_backbone=pretrain_backbone,
                     backbone_size=backbone_size,
                     backbone_checkpoint_path=backbone_checkpoint_path,

                     num_tiles=num_tiles_height * num_tiles_width,
                     num_tiles_height=num_tiles_height,
                     num_tiles_width=num_tiles_width,
                     series_length=series_length)

    ### Initialize LightningModule ###
    if checkpoint_path and checkpoint:
        # Load from checkpoint
        lightning_module = LightningModule.load_from_checkpoint(
                               checkpoint_path,
                               model=main_model,
                               omit_images_from_test=omit_images_from_test,

                               optimizer_type=optimizer_type,
                               optimizer_weight_decay=optimizer_weight_decay,
                               learning_rate=learning_rate,
                               lr_schedule=lr_schedule,

                               series_length=series_length,
                               parsed_args=parsed_args)
    else:
        lightning_module = LightningModule(
                               model=main_model,
                               omit_images_from_test=omit_images_from_test,

                               optimizer_type=optimizer_type,
                               optimizer_weight_decay=optimizer_weight_decay,
                               learning_rate=learning_rate,
                               lr_schedule=lr_schedule,

                               series_length=series_length,
                               parsed_args=parsed_args)

    ### Implement EarlyStopping & Other Callbacks ###
    callbacks = []

    if early_stopping: 
        early_stop_callback = EarlyStopping(
                               monitor='val/loss',
                               min_delta=0.00,
                               patience=early_stopping_patience,
                               verbose=True)
        callbacks.append(early_stop_callback)
    if not is_debug: 
        checkpoint_callback = ModelCheckpoint(monitor='val/loss', save_last=True)
        callbacks.append(checkpoint_callback)

        lr_monitor = LearningRateMonitor(logging_interval='epoch')
        callbacks.append(lr_monitor)

    ### Initialize Trainer ###

    # Initialize logger 
    logger = TensorBoardLogger("./lightning_logs/", 
                               name=experiment_name, 
                               log_graph=False,
                               version=None)

    # Set up data_module and save train/val/test splits
    data_module.setup(log_dir=logger.log_dir if not is_debug else None)

    trainer = pl.Trainer(
        # Trainer args
        min_epochs=min_epochs,
        max_epochs=max_epochs,
        callbacks=callbacks,
        precision=16 if sixteen_bit else 32,
        stochastic_weight_avg=stochastic_weight_avg,
        gradient_clip_val=gradient_clip_val,
        accumulate_grad_batches=accumulate_grad_batches,

        # Other args
        resume_from_checkpoint=checkpoint_path if not is_hem_training else None,
        logger=logger if not is_debug else False,
        log_every_n_steps=512/(batch_size*accumulate_grad_batches),
#             val_check_interval=0.5,

        # Dev args
        num_sanity_val_steps=0,
#             fast_dev_run=True, 
#             overfit_batches=100,
#             limit_train_batches=2,
#             limit_val_batches=2,
#             limit_test_batches=2,
#             track_grad_norm=2,
#             weights_summary='full',
#             profiler="simple", # "advanced" "pytorch"
#             log_gpu_memory=True,
        gpus=1)

    ### Training & Evaluation ###
    if is_test_only:
        trainer.test(lightning_module, datamodule=data_module)
    else:
        trainer.fit(lightning_module, datamodule=data_module)
        trainer.test(lightning_module, datamodule=data_module)
    
    
if __name__ == '__main__':
    args = vars(parser.parse_args())
    
    # Load checkpoint if it exists
    if args['checkpoint_path'] is not None:
        checkpoint = torch.load(args['checkpoint_path'])
    else:
        checkpoint = None
    
    # Load args from checkpoint if it exists and not hem_training
    if args['checkpoint_path'] is None or args['is_hem_training']:
        parsed_args = args
    else:
        parsed_args = checkpoint['hyper_parameters']
        
    main(# Debug args
        is_debug=args['is_debug'],
        is_test_only=args['is_test_only'],
        is_hem_training=args['is_hem_training'],
        omit_list=parsed_args['omit_list'],
        omit_images_from_test=args['omit_images_from_test'],
        mask_omit_images=parsed_args['mask_omit_images'],
        is_object_detection=parsed_args['is_object_detection'],
        
        # Path args - always used command line args for these
        raw_data_path=args['raw_data_path'],
        labels_path=args['labels_path'], 
        raw_labels_path=args['raw_labels_path'],
        metadata_path=args['metadata_path'],
        
        train_split_path=args['train_split_path'], 
        val_split_path=args['val_split_path'], 
        test_split_path=args['test_split_path'],
        load_images_from_split=args['load_images_from_split'],

        # Experiment args
        experiment_name=args['experiment_name'],
        experiment_description=args['experiment_description'],
        parsed_args=parsed_args,

        # Dataloader args
        train_split_size=parsed_args['train_split_size'],
        test_split_size=parsed_args['test_split_size'],
        batch_size=parsed_args['batch_size'], 
        num_workers=parsed_args['num_workers'], 
        
        series_length=parsed_args['series_length'], 
        add_base_flow=parsed_args['add_base_flow'], 
        time_range=(parsed_args['time_range_min'],parsed_args['time_range_max']), 

        original_dimensions=(parsed_args['original_height'], parsed_args['original_width']),
        resize_dimensions=(parsed_args['resize_height'], parsed_args['resize_width']),
        crop_height=parsed_args['crop_height'],
        tile_dimensions=(parsed_args['tile_size'], parsed_args['tile_size']),
        tile_overlap=parsed_args['tile_overlap'],
        pre_tile=parsed_args['no_pre_tile'],
        smoke_threshold=parsed_args['smoke_threshold'],
        num_tile_samples=parsed_args['num_tile_samples'],

        flip_augment=parsed_args['no_flip_augment'],
        resize_crop_augment=parsed_args['no_resize_crop_augment'],
        blur_augment=parsed_args['no_blur_augment'],
        color_augment=parsed_args['no_color_augment'],
        brightness_contrast_augment=parsed_args['no_brightness_contrast_augment'],

        # Model args
        model_type_list=parsed_args['model_type_list'],
        pretrain_epochs=parsed_args['pretrain_epochs'],
        intermediate_supervision=parsed_args['no_intermediate_supervision'],
        use_image_preds=parsed_args['use_image_preds'],
        tile_embedding_size=parsed_args['tile_embedding_size'],
        
        pretrain_backbone=parsed_args['no_pretrain_backbone'],
        freeze_backbone=parsed_args['freeze_backbone'],
        backbone_size=parsed_args['backbone_size'],
        backbone_checkpoint_path=parsed_args['backbone_checkpoint_path'],
        
        tile_loss_type=parsed_args['tile_loss_type'],
        bce_pos_weight=parsed_args['bce_pos_weight'],
        focal_alpha=parsed_args['focal_alpha'],
        focal_gamma=parsed_args['focal_gamma'],
        
        # Optimizer args
        optimizer_type=parsed_args['optimizer_type'],
        optimizer_weight_decay=parsed_args['optimizer_weight_decay'],
        learning_rate=parsed_args['learning_rate'],
        lr_schedule=parsed_args['use_lr_schedule'],

        # Trainer args
        min_epochs=parsed_args['min_epochs'],
        max_epochs=parsed_args['max_epochs'],
        early_stopping=parsed_args['no_early_stopping'],
        early_stopping_patience=parsed_args['early_stopping_patience'],
        
        sixteen_bit=parsed_args['no_sixteen_bit'],
        stochastic_weight_avg=parsed_args['no_stochastic_weight_avg'],
        gradient_clip_val=parsed_args['gradient_clip_val'],
        accumulate_grad_batches=parsed_args['accumulate_grad_batches'],
    
        # Checkpoint args
        checkpoint_path=args['checkpoint_path'],
        checkpoint=checkpoint)
