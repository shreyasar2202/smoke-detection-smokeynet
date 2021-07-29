#############################################
# Created by: Anshuman Dewangan
# Date: 2021
#
# Description: Used to easily start training from main.py with command line arguments.
#############################################
    
python3.9 src/main.py \
    --experiment-name "1MobileNet_LSTM_ViT_LinearOutputs_50Resize" \
    --model-type-list "RawToTile_MobileNetV3Large" "TileToTile_LSTM" "TileToTile_ViT" "TileToImage_LinearOutputs" \
    --omit-list "omit_no_xml" "omit_no_contour" \
    --batch-size 2 \
    --series-length 2 \
    --accumulate-grad-batches 16 \
    --no-lr-schedule \
    --resize-height 780 \
    --resize-width 1040 \
    --crop-height 632 \
    --train-split-path './data/train_images1.txt' \
    --val-split-path './data/val_images1.txt' \
    --test-split-path './data/test_images1.txt' \
    
python3.9 src/main.py \
    --experiment-name "1MobileNet_LSTM_ViT_LinearOutputs_80Resize" \
    --model-type-list "RawToTile_MobileNetV3Large" "TileToTile_LSTM" "TileToTile_ViT" "TileToImage_LinearOutputs" \
    --omit-list "omit_no_xml" "omit_no_contour" \
    --batch-size 2 \
    --series-length 2 \
    --accumulate-grad-batches 16 \
    --no-lr-schedule \
    --resize-height 1239 \
    --resize-width 1652 \
    --crop-height 1040 \
    --train-split-path './data/train_images1.txt' \
    --val-split-path './data/val_images1.txt' \
    --test-split-path './data/test_images1.txt' \

    

#####################
## Normal Run
#####################

# python3.9 src/main.py \
#     --experiment-name "MobileNet" \
#     --model-type-list "RawToTile_MobileNetV3Large" \
#     --omit-list "omit_no_xml" "omit_no_contour" \
#     --batch-size 4 \
#     --series-length 1 \
#     --accumulate-grad-batches 8 \
#     --no-lr-schedule \

# python3.9 src/main.py \
#     --experiment-name "MobileNet_LSTM_DeiT_LinearOutputs" \
#     --model-type-list "RawToTile_MobileNetV3Large" "TileToTile_LSTM" "TileToTile_DeiT" "TileToImage_LinearOutputs" \
#     --omit-list "omit_no_xml" \
#     --batch-size 2 \
#     --series-length 2 \
#     --accumulate-grad-batches 16 \
#     --no-lr-schedule \


#     --labels-path '/root/pytorch_lightning_data/drive_clone_labels' \
#     --raw-data-path '/root/raw_images' \

#########################
## Without Overlapping Tiles
#########################

#     --resize-width 2016 \
#     --crop-height 1120 \
#     --tile-overlap 0 \
    
#####################
## Create Embeddings
#####################

# python3.9 src/main.py \
#     --test-split-path './data/all_fires.txt' \
#     --train-split-path './data/all_fires.txt' \
#     --val-split-path './data/all_fires.txt' \
#     --load-images-from-split \
#     --checkpoint-path './saved_logs/version_90/checkpoints/epoch=13-step=3177.ckpt' \
#     --raw-data-path '/root/raw_images' \
#     --labels-path '/root/drive_clone_labels' \
#     --is-debug \
#     --is-test-only \
#     --save-embeddings-path '/userdata/kerasData/data/new_data/pytorch_lightning_data/embeddings_version_90' 

# tar -czf /userdata/kerasData/data/new_data/pytorch_lightning_data/embeddings_version_90.tar.gz /userdata/kerasData/data/new_data/pytorch_lightning_data/embeddings_version_90


#####################
## Random Upsampling
#####################

#     --time-range-min 0 \
#     --num-tile-samples 30 \
#     --bce-pos-weight 1 \
#     --smoke-threshold -1 \

    
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
#     --experiment-name "TileToTile_ViDeiT" \
#     --model-type-list "TileToTile_ViDeiT" \
#     --min-epochs 3 \
#     --max-epochs 50 \
#     --batch-size 32 \
#     --series-length 4 \
#     --accumulate-grad-batches 1 \
#     --no-lr-schedule \
#     --tile-embedding-size 1792 \
#     --labels-path '/root/pytorch_lightning_data/drive_clone_labels' \
#     --embeddings-path '/root/pytorch_lightning_data/embeddings_version_90' \
#     --train-split-path './saved_logs/version_90/train_images.txt' \
#     --val-split-path './saved_logs/version_90/val_images.txt' \
#     --test-split-path './saved_logs/version_90/test_images.txt' \


#########################
## Load Backbone Checkpoint
#########################

#     --backbone-checkpoint-path './saved_logs/version_85/epoch=8-step=2015.ckpt' \
#     --train-split-path './saved_logs/version_85/train_images.txt' \
#     --val-split-path './saved_logs/version_85/val_images.txt' \
#     --test-split-path './saved_logs/version_85/test_images.txt' \

#########################
## Best Model
#########################

# python3.9 src/main.py \
#     --experiment-name "MobileNet_DeiT_IntSupervision_NoCheckpoint" \
#     --model-type-list "RawToTile_MobileNetV3Large" "TileToTile_DeiT" \
#     --min-epochs 3 \
#     --max-epochs 25 \
#     --batch-size 4 \
#     --series-length 1 \
#     --accumulate-grad-batches 8 \
#     --no-lr-schedule \
#     --labels-path '/root/pytorch_lightning_data/drive_clone_labels' \
#     --raw-data-path '/root/raw_images' \