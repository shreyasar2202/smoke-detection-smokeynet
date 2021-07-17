#############################################
# Created by: Anshuman Dewangan
# Date: 2021
#
# Description: Used to easily start training from main.py with command line arguments.
#############################################


python3.9 src/main.py \
    --experiment-name "MobileNet" \
    --model-type-list "RawToTile_MobileNetV3Large" \
    --min-epochs 3 \
    --max-epochs 50 \
    --batch-size 4 \
    --series-length 1 \
    --accumulate-grad-batches 8 \
    --no-lr-schedule \
    --labels-path '/root/pytorch_lightning_data/drive_clone_labels' \
    --raw-data-path '/root/raw_images' \
    --is-debug

# python3.9 src/main.py \
#     --experiment-name "RawToTile_DeiT" \
#     --model-type-list "RawToTile_DeiT" \
#     --min-epochs 3 \
#     --max-epochs 50 \
#     --batch-size 8 \
#     --series-length 1 \
#     --accumulate-grad-batches 4 \
#     --no-lr-schedule \
#     --labels-path '/root/pytorch_lightning_data/drive_clone_labels' \
#     --raw-data-path '/root/raw_images' \

# python3.9 src/main.py \
#     --experiment-name "RawToTile_ViDeiT" \
#     --model-type-list "RawToTile_DeiT" \
#     --min-epochs 3 \
#     --max-epochs 50 \
#     --batch-size 2 \
#     --series-length 4 \
#     --accumulate-grad-batches 16 \
#     --no-lr-schedule \
#     --labels-path '/root/pytorch_lightning_data/drive_clone_labels' \
#     --raw-data-path '/root/raw_images' \

# python3.9 src/main.py \
#     --experiment-name "RawToTile_ViDeiT_BaseFlow" \
#     --model-type-list "RawToTile_DeiT" \
#     --min-epochs 3 \
#     --max-epochs 50 \
#     --batch-size 2 \
#     --series-length 4 \
#     --accumulate-grad-batches 16 \
#     --no-lr-schedule \
#     --add-base-flow \
#     --labels-path '/root/pytorch_lightning_data/drive_clone_labels' \
#     --raw-data-path '/root/raw_images' \
    
# python3.9 src/main.py \
#     --experiment-name "EfficientNetB6" \
#     --model-type-list "RawToTile_EfficientNetB6" \
#     --min-epochs 3 \
#     --max-epochs 50 \
#     --batch-size 1 \
#     --series-length 1 \
#     --accumulate-grad-batches 32 \
#     --num-workers 0 \
#     --no-lr-schedule \
#     --time-range-min 0 \
#     --num-tile-samples 30 \
#     --bce-pos-weight 1 \
#     --smoke-threshold 0 \
#     --labels-path '/root/pytorch_lightning_data/drive_clone_labels' \
#     --raw-data-path '/root/raw_images' \
    
# python3.9 src/main.py \
#     --experiment-name "RawToTile_DeiT_LinearOutputs" \
#     --model-type-list "RawToTile_DeiT" "TileToImage_LinearOutputs" \
#     --min-epochs 3 \
#     --max-epochs 50 \
#     --batch-size 8 \
#     --series-length 1 \
#     --accumulate-grad-batches 4 \
#     --no-lr-schedule \
#     --labels-path '/root/pytorch_lightning_data/drive_clone_labels' \
#     --raw-data-path '/root/raw_images' \
    
# python3.9 src/main.py \
#     --experiment-name "RawToTile_DeiT_LinearEmbeddings" \
#     --model-type-list "RawToTile_DeiT" "TileToImage_LinearEmbeddings" \
#     --min-epochs 3 \
#     --max-epochs 50 \
#     --batch-size 8 \
#     --series-length 1 \
#     --accumulate-grad-batches 4 \
#     --no-lr-schedule \
#     --labels-path '/root/pytorch_lightning_data/drive_clone_labels' \
#     --raw-data-path '/root/raw_images' \

#####################
## Normal Run
#####################

# python3.9 src/main.py \
#     --experiment-name "MobileNet" \
#     --model-type-list "RawToTile_MobileNetV3Large" \
#     --min-epochs 3 \
#     --max-epochs 50 \
#     --batch-size 8 \
#     --series-length 1 \
#     --accumulate-grad-batches 4 \
#     --no-lr-schedule \
#     --labels-path '/root/pytorch_lightning_data/drive_clone_labels' \
#     --raw-data-path '/root/raw_images' \

#########################
## With Overlapping Tiles
#########################

#     --resize-height 1536 \
#     --resize-width 2060 \
#     --crop-height 1244 \
#     --tile-overlap 20 \
    
#####################
## Create Embeddings
#####################

# python3.9 src/main.py \
#     --test-split-path './data/all_fires.txt' \
#     --train-split-path './data/all_fires.txt' \
#     --val-split-path './data/all_fires.txt' \
#     --checkpoint-path './saved_logs/version_45/checkpoints/epoch=1-step=449.ckpt' \
#     --raw-data-path '/root/raw_images' \
#     --labels-path '/root/drive_clone_labels'


#####################
## Random Upsampling
#####################

#     --time-range-min 0 \
#     --num-tile-samples 30 \
#     --bce-pos-weight 1 \
#     --smoke-threshold 1 \

    
#########################
## Load from Checkpoint
#########################

#     --checkpoint-path './lightning_logs/resnet50/version_9/checkpoints/last.ckpt' \
#     --train-split-path './lightning_logs/resnet50/version_9/train_images.txt' \
#     --val-split-path './lightning_logs/resnet50/version_9/val_images.txt' \
#     --test-split-path './lightning_logs/resnet50/version_9/test_images.txt' \


#########################
## Load MaskRCNN train/test split
#########################

#     --train-split-path '/userdata/kerasData/data/new_data/mask_rcnn_preprocessed/20210420/train_list.txt' \
#     --val-split-path '/userdata/kerasData/data/new_data/mask_rcnn_preprocessed/20210420/val_list.txt' \
#     --test-split-path '/userdata/kerasData/data/new_data/mask_rcnn_preprocessed/20210420/test_list.txt' \


#########################
## Load from Embeddings
#########################

# python3.9 src/main.py \
#     --experiment-name "MobileEmbeddings_Transformer" \
#     --model-type-list "TileToTile_Transformer" \
#     --min-epochs 3 \
#     --max-epochs 50 \
#     --batch-size 32 \
#     --series-length 4 \
#     --accumulate-grad-batches 1 \
#     --no-lr-schedule \
#     --labels-path '/root/pytorch_lightning_data/drive_clone_labels' \
#     --embeddings-path '/root/pytorch_lightning_data/embeddings' \
#     --train-split-path './saved_logs/version_80/train_images.txt' \
#     --val-split-path './saved_logs/version_80/val_images.txt' \
#     --test-split-path './saved_logs/version_80/test_images.txt' \


#########################
## Best Model
#########################

# python3.9 src/main.py \
#     --experiment-name "MobileNet" \
#     --model-type-list "RawToTile_MobileNetV3Large" \
#     --min-epochs 3 \
#     --max-epochs 50 \
#     --batch-size 8 \
#     --series-length 1 \
#     --accumulate-grad-batches 4 \
#     --no-lr-schedule \
#     --labels-path '/root/pytorch_lightning_data/drive_clone_labels' \
#     --raw-data-path '/root/raw_images' \