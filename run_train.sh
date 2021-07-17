#############################################
# Created by: Anshuman Dewangan
# Date: 2021
#
# Description: Used to easily start training from main.py with command line arguments.
#############################################

# Check flags in main.py before starting!

# python3 src/main.py \
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
#     --is-debug

python3 src/main.py \
    --experiment-name "RawToTile_MobileNetV3Large" \
    --model-type-list "RawToTile_MobileNetV3Large" \
    --min-epochs 3 \
    --max-epochs 50 \
    --batch-size 2 \
    --series-length 1 \
    --accumulate-grad-batches 16 \
    --no-lr-schedule \
    --labels-path '/root/pytorch_lightning_data/drive_clone_labels' \
    --raw-data-path '/root/raw_images' \
    --backbone-checkpoint-path '/userdata/kerasData/src/pytorch-lightning-smoke-detection/saved_logs/version_80/checkpoints/last.ckpt' \
    --is-debug
    
# python3 src/main.py \
#     --experiment-name "MobileEmbeddings_LSTM_DeiT_LinearEmbeddings" \
#     --model-type-list "TileToTile_LSTM" "TileToTile_DeiT" "TileToImage_LinearEmbeddings" \
#     --min-epochs 3 \
#     --max-epochs 50 \
#     --batch-size 32 \
#     --series-length 4 \
#     --accumulate-grad-batches 1 \
#     --no-freeze-backbone \
#     --flip-augment \
#     --blur-augment \
#     --no-lr-schedule \
#     --labels-path '/root/pytorch_lightning_data/drive_clone_labels' \
#     --embeddings-path '/root/pytorch_lightning_data/embeddings' \
    
# python3 src/main.py \
#     --experiment-name "MobileEmbeddings_LSTM_DeiT_LinearEmbeddings_TilePreds" \
#     --model-type-list "TileToTile_LSTM" "TileToTile_DeiT" "TileToImage_LinearEmbeddings" \
#     --min-epochs 3 \
#     --max-epochs 50 \
#     --batch-size 32 \
#     --series-length 4 \
#     --add-base-flow \
#     --accumulate-grad-batches 1 \
#     --no-freeze-backbone \
#     --flip-augment \
#     --blur-augment \
#     --no-lr-schedule \
#     --labels-path '/root/pytorch_lightning_data/drive_clone_labels' \
#     --embeddings-path '/root/pytorch_lightning_data/embeddings' \
    
# python3 src/main.py \
#     --experiment-name "MobileEmbeddings_LSTM_DeiT" \
#     --model-type-list "TileToTile_LSTM" "TileToTile_DeiT" \
#     --min-epochs 3 \
#     --max-epochs 50 \
#     --batch-size 32 \
#     --series-length 5 \
#     --add-base-flow \
#     --accumulate-grad-batches 1 \
#     --no-freeze-backbone \
#     --flip-augment \
#     --blur-augment \
#     --no-lr-schedule \
#     --labels-path '/root/pytorch_lightning_data/drive_clone_labels' \
#     --embeddings-path '/root/pytorch_lightning_data/embeddings' \
    
# python3 src/main.py \
#     --experiment-name "MobileNet_LSTM_DeiT" \
#     --model-type-list "RawToTile_MobileNetV3Large" "TileToTile_LSTM" "TileToTile_DeiT" \
#     --min-epochs 3 \
#     --max-epochs 50 \
#     --batch-size 1 \
#     --series-length 4 \
#     --add-base-flow \
#     --accumulate-grad-batches 32 \
#     --no-freeze-backbone \
#     --model-pretrain-epochs 5 0 0 \
#     --flip-augment \
#     --blur-augment \
#     --no-lr-schedule \
#     --labels-path '/root/pytorch_lightning_data/drive_clone_labels' \
#     --raw-data-path '/root/raw_images' \





#####################
## Normal Run
#####################

# python3 src/main.py \
#     --experiment-name "MobileNet" \
#     --model-type-list "RawToTile_MobileNetV3Large" \
#     --min-epochs 3 \
#     --max-epochs 10 \
#     --batch-size 8 \
#     --series-length 1 \
#     --accumulate-grad-batches 4 \
#     --no-freeze-backbone \
#     --flip-augment \
#     --blur-augment \
#     --no-lr-schedule \
#     --labels-path '/root/pytorch_lightning_data/drive_clone_labels' \
#     --raw-data-path '/root/raw_images' \


#####################
## Debug Run
#####################

# python3 main.py \
#     --experiment-name "Debug" \
#     --experiment-description "Debug" \
#     --min-epochs 1 \
#     --max-epochs 1 \
#     --batch-size 1 \
#     --flip-augment \
#     --blur-augment \
#     --raw-data-path '/root/raw_images' \
#     --labels-path '/root/drive_clone_labels'
    
    
#####################
## Create Embeddings
#####################

# python3 src/main.py \
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
## Best Model
#########################

# python3 src/main.py \
#     --experiment-name "MobileNet" \
#     --model-type-list "RawToTile_MobileNetV3Large" \
#     --min-epochs 3 \
#     --max-epochs 50 \
#     --batch-size 8 \
#     --series-length 1 \
#     --accumulate-grad-batches 4 \
#     --no-freeze-backbone \
#     --flip-augment \
#     --blur-augment \
#     --no-lr-schedule \
#     --labels-path '/root/pytorch_lightning_data/drive_clone_labels' \
#     --raw-data-path '/root/raw_images' \

# python3 src/main.py \
#     --experiment-name "MobileEmbeddings_LSTM_DeiT" \
#     --model-type-list "TileToTile_LSTM" "TileToTile_DeiT" \
#     --min-epochs 3 \
#     --max-epochs 50 \
#     --batch-size 32 \
#     --series-length 4 \
#     --accumulate-grad-batches 1 \
#     --no-freeze-backbone \
#     --flip-augment \
#     --blur-augment \
#     --no-lr-schedule \
#     --labels-path '/root/pytorch_lightning_data/drive_clone_labels' \
#     --embeddings-path '/root/pytorch_lightning_data/embeddings' \