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

# Debug args = 2
parser.add_argument('--is-debug', action='store_true',
                    help='Turns off logging and checkpointing.')
parser.add_argument('--is-test-only', action='store_true',
                    help='Skips training for testing only. Useful when checkpoint loading.')

# Experiment args = 2
parser.add_argument('--experiment-name', type=str, default=None,
                    help='(Optional) Name for experiment')
parser.add_argument('--experiment-description', type=str, default=None,
                    help='(Optional) Short description of experiment that will be saved as a hyperparam')

# Path args = 5 + 4
parser.add_argument('--raw-data-path', type=str, default='/userdata/kerasData/data/new_data/raw_images',
                    help='Path to raw images.')
parser.add_argument('--embeddings-path', type=str, default=None,
                    help='(Optional) Path to pretrained embeddings. Default: /userdata/kerasData/data/new_data/pytorch_lightning_data/embeddings')
parser.add_argument('--labels-path', type=str, default='/userdata/kerasData/data/new_data/pytorch_lightning_data/drive_clone_labels',
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
parser.add_argument('--save-embeddings-path', type=str, default=None,
                    help='If not None, where to save test embeddings. Use with --is-test-only.')

# Dataloader args = 4 + 4 + 6 + 2
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

parser.add_argument('--resize-height', type=int, default=1536,
                    help='Desired resize height of image.')
parser.add_argument('--resize-width', type=int, default=2016,
                    help='Desired resize width of image.')
parser.add_argument('--crop-height', type=int, default=1120,
                    help='Desired height after cropping.')
parser.add_argument('--tile-size', type=int, default=224,
                    help='Height and width of tile.')
parser.add_argument('--tile-overlap', type=int, default=0,
                    help='Amount to overlap each tile.')
parser.add_argument('--smoke-threshold', type=int, default=250,
                    help='Number of pixels of smoke to consider tile positive.')
parser.add_argument('--num-tile-samples', type=int, default=0,
                    help='Number of random tile samples per batch for upsampling positives and downsampling negatives. If < 1, then turned off. Recommended: 30.')

parser.add_argument('--no-flip-augment', action='store_false',
                    help='Disables data augmentation with horizontal flip.')
parser.add_argument('--no-blur-augment', action='store_false',
                    help='Disables data augmentation with Gaussian blur.')
parser.add_argument('--no-jitter-augment', action='store_false',
                    help='Disables data augmentation with slightly displaced cropping.')


# Model args = 3 + 2 + 4
parser.add_argument('--model-type-list', nargs='*',
                    help='Specify the model type through multiple model components.')
parser.add_argument('--pretrain-epochs', nargs='*',
                    help='Specify the number of epochs to pretrain each model component.')
parser.add_argument('--no-intermediate-supervision', action='store_false',
                    help='Disables intermediate supervision for chained models.')

parser.add_argument('--no-pretrain-backbone', action='store_false',
                    help='Disables pretraining of backbone.')
parser.add_argument('--freeze-backbone', action='store_true',
                    help='Freezes layers on pre-trained backbone.')
parser.add_argument('--backbone-checkpoint-path', type=str, default=None,
                    help='Loads pretrained weights for the backbone from a checkpoint.')

parser.add_argument('--tile-loss-type', type=str, default='bce',
                    help='Type of loss to use for training. Options: [bce], [focal]')
parser.add_argument('--bce-pos-weight', type=float, default=36,
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
parser.add_argument('--no-lr-schedule', action='store_false',
                    help='Disables ReduceLROnPlateau learning rate scheduler. See PyTorch Lightning docs for more details.')

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
        
        # Path args
        raw_data_path=None, 
        embeddings_path=None,
        labels_path=None, 
        raw_labels_path=None,
        metadata_path=None,
    
        train_split_path=None, 
        val_split_path=None, 
        test_split_path=None,
        load_images_from_split=False,
        save_embeddings_path=None,
    
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
    
        resize_dimensions=(1536, 2016),
        crop_height=1120,
        tile_dimensions=(224,224),
        tile_overlap=0,
        smoke_threshold=250,
        num_tile_samples=0,
    
        flip_augment=True,
        blur_augment=True,
        jitter_augment=True,

        # Model args
        model_type_list=['RawToTile_MobileNetV3Large'],
        pretrain_epochs=None,
        intermediate_supervision=True,

        pretrain_backbone=True,
        freeze_backbone=False,
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
        
    print("IS_DEBUG: ",  is_debug)
    print("IS_TEST_ONLY: ", is_test_only)

    ### Initialize data_module ###
    data_module = DynamicDataModule(
        # Path args
        raw_data_path=raw_data_path,
        embeddings_path=embeddings_path,
        labels_path=labels_path,
        raw_labels_path=raw_labels_path,
        metadata_path=metadata_path,
        
        train_split_path=train_split_path,
        val_split_path=val_split_path,
        test_split_path=test_split_path,
        load_images_from_split=load_images_from_split,
        save_embeddings_path=save_embeddings_path,

        # Dataloader args
        train_split_size=train_split_size,
        test_split_size=test_split_size,
        batch_size=batch_size,
        num_workers=num_workers,
        
        series_length=series_length,
        add_base_flow=add_base_flow,
        time_range=time_range,

        resize_dimensions=resize_dimensions,
        crop_height=crop_height,
        tile_dimensions=tile_dimensions,
        tile_overlap=tile_overlap,
        smoke_threshold=smoke_threshold,
        num_tile_samples=num_tile_samples,

        flip_augment=flip_augment,
        blur_augment=blur_augment,
        jitter_augment=jitter_augment)

    ### Initialize MainModel ###
    num_tiles_height, num_tiles_width = util_fns.calculate_num_tiles(resize_dimensions, crop_height, tile_dimensions, tile_overlap)

    main_model = MainModel(
                     # Model args
                     model_type_list=model_type_list,
                     pretrain_epochs=pretrain_epochs,
                     intermediate_supervision=intermediate_supervision,

                     tile_loss_type=tile_loss_type,
                     bce_pos_weight=bce_pos_weight,
                     focal_alpha=focal_alpha, 
                     focal_gamma=focal_gamma,

                     freeze_backbone=freeze_backbone, 
                     pretrain_backbone=pretrain_backbone,
                     backbone_checkpoint_path=backbone_checkpoint_path,

                     num_tiles=num_tiles_height * num_tiles_width,
                     num_tiles_height=num_tiles_height,
                     num_tiles_width=num_tiles_width,

                     image_size=(crop_height, resize_dimensions[1]),
                     tile_size=tile_dimensions[0],
                     series_length=series_length)

    ### Initialize LightningModule ###
    if checkpoint_path and checkpoint:
        # Load from checkpoint
        lightning_module = LightningModule.load_from_checkpoint(
                               checkpoint_path,
                               model=main_model,

                               optimizer_type=optimizer_type,
                               optimizer_weight_decay=optimizer_weight_decay,
                               learning_rate=learning_rate,
                               lr_schedule=lr_schedule,

                               series_length=series_length,
                               parsed_args=parsed_args,
                               save_embeddings_path=save_embeddings_path)
    else:
        lightning_module = LightningModule(
                               model=main_model,

                               optimizer_type=optimizer_type,
                               optimizer_weight_decay=optimizer_weight_decay,
                               learning_rate=learning_rate,
                               lr_schedule=lr_schedule,

                               series_length=series_length,
                               parsed_args=parsed_args,
                               save_embeddings_path=save_embeddings_path)

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
        resume_from_checkpoint=checkpoint_path,
        logger=logger if not is_debug else False,
        log_every_n_steps=512/(batch_size*accumulate_grad_batches),
#             val_check_interval=0.5,

        # Dev args
#             fast_dev_run=True, 
#             overfit_batches=65,
#             limit_train_batches=1,
#             limit_val_batches=1,
#             limit_test_batches=1,
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
    args = parser.parse_args()
    
    # Load hyperparameters from checkpoint if it exists
    if args.checkpoint_path is not None:
        checkpoint = torch.load(args.checkpoint_path)
        parsed_args = Namespace(**checkpoint['hyper_parameters'])
    else:
        checkpoint = None
        parsed_args = args
        
    main(# Debug args
        is_debug=args.is_debug,
        is_test_only=args.is_test_only,
        
        # Path args - always used command line args for these
        raw_data_path=args.raw_data_path,
        embeddings_path=args.embeddings_path,
        labels_path=args.labels_path, 
        raw_labels_path=args.raw_labels_path,
        metadata_path=args.metadata_path,
        
        train_split_path=args.train_split_path, 
        val_split_path=args.val_split_path, 
        test_split_path=args.test_split_path,
        load_images_from_split=args.load_images_from_split,
        save_embeddings_path=args.save_embeddings_path,

        # Experiment args
        experiment_name=parsed_args.experiment_name,
        experiment_description=parsed_args.experiment_description,
        parsed_args=parsed_args,

        # Dataloader args
        train_split_size=parsed_args.train_split_size,
        test_split_size=parsed_args.test_split_size,
        batch_size=parsed_args.batch_size, 
        num_workers=parsed_args.num_workers, 
        
        series_length=parsed_args.series_length, 
        add_base_flow=parsed_args.add_base_flow, 
        time_range=(parsed_args.time_range_min,args.time_range_max), 

        resize_dimensions=(parsed_args.resize_height, args.resize_width),
        crop_height=parsed_args.crop_height,
        tile_dimensions=(parsed_args.tile_size, args.tile_size),
        tile_overlap=parsed_args.tile_overlap,
        smoke_threshold=parsed_args.smoke_threshold,
        num_tile_samples=parsed_args.num_tile_samples,

        flip_augment=parsed_args.no_flip_augment,
        blur_augment=parsed_args.no_blur_augment,
        jitter_augment=parsed_args.no_jitter_augment,

        # Model args
        model_type_list=parsed_args.model_type_list,
        pretrain_epochs=parsed_args.pretrain_epochs,
        intermediate_supervision=parsed_args.no_intermediate_supervision,
        
        pretrain_backbone=parsed_args.no_pretrain_backbone,
        freeze_backbone=parsed_args.freeze_backbone,
        backbone_checkpoint_path=parsed_args.backbone_checkpoint_path,
        
        tile_loss_type=parsed_args.tile_loss_type,
        bce_pos_weight=parsed_args.bce_pos_weight,
        focal_alpha=parsed_args.focal_alpha,
        focal_gamma=parsed_args.focal_gamma,
        
        # Optimizer args
        optimizer_type=parsed_args.optimizer_type,
        optimizer_weight_decay=parsed_args.optimizer_weight_decay,
        learning_rate=parsed_args.learning_rate,
        lr_schedule=parsed_args.no_lr_schedule,

        # Trainer args
        min_epochs=parsed_args.min_epochs,
        max_epochs=parsed_args.max_epochs,
        early_stopping=parsed_args.no_early_stopping,
        early_stopping_patience=parsed_args.early_stopping_patience,
        
        sixteen_bit=parsed_args.no_sixteen_bit,
        stochastic_weight_avg=parsed_args.no_stochastic_weight_avg,
        gradient_clip_val=parsed_args.gradient_clip_val,
        accumulate_grad_batches=parsed_args.accumulate_grad_batches,
    
        # Checkpoint args
        checkpoint_path=args.checkpoint_path,
        checkpoint=checkpoint)
