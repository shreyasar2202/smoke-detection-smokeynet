#############################################
# Created by: Anshuman Dewangan
# Date: 2021
#
# Description: Used to easily start training from main.py with command line arguments.
#############################################

    
python3.9 src/main.py \
    --experiment-name "MobileNet_OverlapJitter" \
    --model-type-list "RawToTile_MobileNetV3Large" \
    --min-epochs 3 \
    --max-epochs 50 \
    --batch-size 4 \
    --series-length 1 \
    --accumulate-grad-batches 8 \
    --resize-width 2060 \
    --crop-height 1244 \
    --tile-overlap 20 \
    --no-lr-schedule \
    --labels-path '/root/pytorch_lightning_data/drive_clone_labels' \
    --raw-data-path '/root/raw_images' \
    --checkpoint-path './lightning_logs/MobileNet_OverlapJitter/version_1/checkpoints/epoch=8-step=2015.ckpt' \
    --train-split-path './lightning_logs/MobileNet_OverlapJitter/version_1/train_images.txt' \
    --val-split-path './lightning_logs/MobileNet_OverlapJitter/version_1/val_images.txt' \
    --test-split-path './lightning_logs/MobileNet_OverlapJitter/version_1/test_images.txt' \
    --is-test-only \
    
python3.9 src/main.py \
    --experiment-name "MobileNet_LinearOutputs_TilePreds" \
    --model-type-list "RawToTile_MobileNetV3Large" "TileToImage_LinearOutputs" \
    --min-epochs 3 \
    --max-epochs 50 \
    --batch-size 4 \
    --series-length 1 \
    --accumulate-grad-batches 8 \
    --no-lr-schedule \
    --no-jitter-augment \
    --labels-path '/root/pytorch_lightning_data/drive_clone_labels' \
    --raw-data-path '/root/raw_images' \
    --checkpoint-path './lightning_logs/MobileNet_LinearOutputs_TilePreds/version_0/checkpoints/epoch=6-step=1581.ckpt' \
    --train-split-path './lightning_logs/MobileNet_LinearOutputs_TilePreds/version_0/train_images.txt' \
    --val-split-path './lightning_logs/MobileNet_LinearOutputs_TilePreds/version_0/val_images.txt' \
    --test-split-path './lightning_logs/MobileNet_LinearOutputs_TilePreds/version_0/test_images.txt' \
    --is-test-only \

python3.9 src/main.py \
    --experiment-name "MobileNet_LinearOutputs_ImagePreds" \
    --model-type-list "RawToTile_MobileNetV3Large" "TileToImage_LinearOutputs" \
    --min-epochs 3 \
    --max-epochs 50 \
    --batch-size 4 \
    --series-length 1 \
    --accumulate-grad-batches 8 \
    --no-lr-schedule \
    --no-jitter-augment \
    --use-image-preds \
    --labels-path '/root/pytorch_lightning_data/drive_clone_labels' \
    --raw-data-path '/root/raw_images' \
    --checkpoint-path './lightning_logs/MobileNet_LinearOutputs_ImagePreds/version_0/checkpoints/epoch=6-step=1595.ckpt' \
    --train-split-path './lightning_logs/MobileNet_LinearOutputs_ImagePreds/version_0/train_images.txt' \
    --val-split-path './lightning_logs/MobileNet_LinearOutputs_ImagePreds/version_0/val_images.txt' \
    --test-split-path './lightning_logs/MobileNet_LinearOutputs_ImagePreds/version_0/test_images.txt' \
    --is-test-only \
    
python3.9 src/main.py \
    --experiment-name "MobileNet_LinearEmbeddings_TilePreds" \
    --model-type-list "RawToTile_MobileNetV3Large" "TileToImage_LinearEmbeddings" \
    --min-epochs 3 \
    --max-epochs 50 \
    --batch-size 4 \
    --series-length 1 \
    --accumulate-grad-batches 8 \
    --no-lr-schedule \
    --no-jitter-augment \
    --labels-path '/root/pytorch_lightning_data/drive_clone_labels' \
    --raw-data-path '/root/raw_images' \
    --checkpoint-path './lightning_logs/MobileNet_LinearEmbeddings_TilePreds/version_0/checkpoints/epoch=6-step=1567.ckpt' \
    --train-split-path './lightning_logs/MobileNet_LinearEmbeddings_TilePreds/version_0/train_images.txt' \
    --val-split-path './lightning_logs/MobileNet_LinearEmbeddings_TilePreds/version_0/val_images.txt' \
    --test-split-path './lightning_logs/MobileNet_LinearEmbeddings_TilePreds/version_0/test_images.txt' \
    --is-test-only \
    
python3.9 src/main.py \
    --experiment-name "MobileNet_LinearEmbeddings_ImagePreds" \
    --model-type-list "RawToTile_MobileNetV3Large" "TileToImage_LinearEmbeddings" \
    --min-epochs 3 \
    --max-epochs 50 \
    --batch-size 4 \
    --series-length 1 \
    --accumulate-grad-batches 8 \
    --no-lr-schedule \
    --no-jitter-augment \
    --use-image-preds \
    --labels-path '/root/pytorch_lightning_data/drive_clone_labels' \
    --raw-data-path '/root/raw_images' \
    --checkpoint-path './lightning_logs/MobileNet_LinearEmbeddings_ImagePreds/version_0/checkpoints/epoch=5-step=1367.ckpt' \
    --train-split-path './lightning_logs/MobileNet_LinearEmbeddings_ImagePreds/version_0/train_images.txt' \
    --val-split-path './lightning_logs/MobileNet_LinearEmbeddings_ImagePreds/version_0/val_images.txt' \
    --test-split-path './lightning_logs/MobileNet_LinearEmbeddings_ImagePreds/version_0/test_images.txt' \
    --is-test-only \
    
    

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