#############################################
# Created by: Anshuman Dewangan
# Date: 2021
#
# Description: Used to easily start training from main.py with command line arguments.
#############################################
    
# python3.9 src/main.py \
#     --experiment-name "2MobileNetFPN_NoPreTile_LSTM_SpatialViT_ImagePreds" \
#     --model-type-list "RawToTile_MobileNetFPN_NoPreTile" "TileToTile_LSTM" "TileToTileImage_SpatialViT" \
#     --omit-list "omit_no_xml" "omit_no_contour" \
#     --mask-omit-images \
#     --use-image-preds \
#     --batch-size 2 \
#     --series-length 2 \
#     --accumulate-grad-batches 16 \
#     --num-workers 6 \
#     --train-split-path './data/split2/train_images2.txt' \
#     --val-split-path './data/split2/val_images2.txt' \
#     --test-split-path './data/split2/test_images2.txt' \
#     --no-pre-tile \
#     --resize-height 1344 \
#     --resize-width 1792 \
#     --crop-height 1120 \
#     --tile-overlap 0 \
    
python3.9 src/main.py \
    --experiment-name "2MobileNetFPNV2_NoPreTile_LSTM_SpatialViT_ImagePreds" \
    --model-type-list "RawToTile_MobileNetFPNV2_NoPreTile" "TileToTile_LSTM" "TileToTileImage_SpatialViT" \
    --omit-list "omit_no_xml" "omit_no_contour" \
    --mask-omit-images \
    --use-image-preds \
    --batch-size 2 \
    --series-length 2 \
    --accumulate-grad-batches 16 \
    --num-workers 6 \
    --train-split-path './data/split2/train_images2.txt' \
    --val-split-path './data/split2/val_images2.txt' \
    --test-split-path './data/split2/test_images2.txt' \
    --no-pre-tile \
    --resize-height 1344 \
    --resize-width 1792 \
    --crop-height 1120 \
    --tile-overlap 0 \
    --tile-embedding-size 768 \
    --is-debug



#####################
## Best Run
#####################

# python3.9 src/main.py \
#     --experiment-name "1MobileNetFPN_LSTM_SpatialViT_ImagePreds" \
#     --model-type-list "RawToTile_MobileNetFPN" "TileToTile_LSTM" "TileToTileImage_SpatialViT" \
#     --omit-list "omit_no_xml" "omit_no_contour" \
#     --mask-omit-images \
#     --use-image-preds \
#     --batch-size 2 \
#     --series-length 2 \
#     --accumulate-grad-batches 16 \
#     --num-workers 6 \
#     --train-split-path './data/split1/hem-train_images1.txt' \
#     --val-split-path './data/split1/val_images1.txt' \
#     --test-split-path './data/split1/test_images1.txt' \

#########################
## Load from Checkpoint
#########################

#     --checkpoint-path './lightning_logs/resnet50/version_9/checkpoints/last.ckpt' \
#     --train-split-path './lightning_logs/resnet50/version_9/train_images.txt' \
#     --val-split-path './lightning_logs/resnet50/version_9/val_images.txt' \
#     --test-split-path './lightning_logs/resnet50/version_9/test_images.txt' \

#########################
## Without Overlapping Tiles
#########################

#     --no-pre-tile
#     --resize-height 1344 \
#     --resize-width 1792 \
#     --crop-height 1344 \
#     --tile-overlap 0 \

#########################
## HEM
#########################

# python3.9 src/main.py \
#     --experiment-name "2MobileNet_LSTM_SpatialViT_90Resize_DataAugment_ImagePreds_HEM-Test" \
#     --model-type-list "RawToTile_MobileNet" "TileToTile_LSTM" "TileToTileImage_SpatialViT" \
#     --omit-list "omit_no_xml" "omit_no_contour" \
#     --mask-omit-images \
#     --use-image-preds \
#     --batch-size 4 \
#     --series-length 2 \
#     --accumulate-grad-batches 8 \
#     --no-lr-schedule \
#     --num-workers 6 \
#     --train-split-path './data/split2/train_images2.txt' \
#     --val-split-path './data/split2/val_images2.txt' \
#     --test-split-path './data/split2/train_images2.txt' \
#     --is-test-only \
#     --checkpoint-path './lightning_logs/2MobileNet_LSTM_SpatialViT_90Resize_DataAugment_ImagePreds/version_0/checkpoints/epoch=16-step=4317.ckpt' \

# python3.9 src/main.py \
#     --experiment-name "2MobileNet_LSTM_SpatialViT_ImagePreds_HEM-Train" \
#     --model-type-list "RawToTile_MobileNet" "TileToTile_LSTM" "TileToTileImage_SpatialViT" \
#     --omit-list "omit_no_xml" "omit_no_contour" \
#     --mask-omit-images \
#     --use-image-preds \
#     --batch-size 4 \
#     --series-length 2 \
#     --accumulate-grad-batches 8 \
#     --no-lr-schedule \
#     --learning-rate 1e-3 \
#     --num-workers 6 \
#     --train-split-path './data/split2/hem-train_images2.txt' \
#     --val-split-path './data/split2/val_images2.txt' \
#     --test-split-path './data/split2/test_images2.txt' \
#     --is-hem-training \
#     --checkpoint-path './saved_logs_195-209/2MobileNet_LSTM_SpatialViT_90Resize_DataAugment_ImagePreds/version_0/checkpoints/epoch=16-step=4317.ckpt' \

### Old Stuff ###

#########################
## Load Backbone Checkpoint
#########################

#     --backbone-checkpoint-path './saved_logs/version_85/epoch=8-step=2015.ckpt' \
#     --train-split-path './saved_logs/version_85/train_images.txt' \
#     --val-split-path './saved_logs/version_85/val_images.txt' \
#     --test-split-path './saved_logs/version_85/test_images.txt' \
    
#####################
## Random Upsampling
#####################

#     --time-range-min 0 \
#     --num-tile-samples 30 \
#     --bce-pos-weight 1 \
#     --smoke-threshold -1 \

#########################
## Load MaskRCNN train/test split
#########################

#     --train-split-path '/userdata/kerasData/data/new_data/mask_rcnn_preprocessed/20210420/train_list.txt' \
#     --val-split-path '/userdata/kerasData/data/new_data/mask_rcnn_preprocessed/20210420/val_list.txt' \
#     --test-split-path '/userdata/kerasData/data/new_data/mask_rcnn_preprocessed/20210420/test_list.txt' \

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





