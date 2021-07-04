#############################################
# Created by: Anshuman Dewangan
# Date: 2021
#
# Description: Used to easily start training from main.py with command line arguments.
#############################################


python3 src/main.py \
    --experiment-name "Intermediate_LSTM" \
    --model-type-list "RawToTile_MobileNetV3Large" "TileToTile_LSTM" \
    --min-epochs 1 \
    --max-epochs 4 \
    --batch-size 1 \
    --num-workers 4 \
    --series-length 4 \
    --accumulate-grad-batches 32 \
    --flip-augment \
    --no-freeze-backbone \
    --learning-rate 1e-2 \
    --time-range-min 0 \
    --num-tile-samples 20 \
    --bce-pos-weight 1 \
    --raw-data-path '/root/raw_images' \
    --labels-path '/root/drive_clone_labels'

python3 src/main.py \
    --experiment-name "Intermediate_LSTM" \
    --model-type-list "RawToTile_MobileNetV3Large" "TileToTile_LSTM" \
    --min-epochs 1 \
    --max-epochs 4 \
    --batch-size 1 \
    --num-workers 4 \
    --series-length 4 \
    --accumulate-grad-batches 32 \
    --flip-augment \
    --no-freeze-backbone \
    --learning-rate 1e-3 \
    --time-range-min 0 \
    --num-tile-samples 20 \
    --bce-pos-weight 1 \
    --raw-data-path '/root/raw_images' \
    --labels-path '/root/drive_clone_labels'
    
python3 src/main.py \
    --experiment-name "Intermediate_LSTM" \
    --model-type-list "RawToTile_MobileNetV3Large" "TileToTile_LSTM" \
    --min-epochs 1 \
    --max-epochs 4 \
    --batch-size 1 \
    --num-workers 4 \
    --series-length 4 \
    --accumulate-grad-batches 32 \
    --flip-augment \
    --no-freeze-backbone \
    --learning-rate 1e-4 \
    --time-range-min 0 \
    --num-tile-samples 20 \
    --bce-pos-weight 1 \
    --raw-data-path '/root/raw_images' \
    --labels-path '/root/drive_clone_labels'
    
python3 src/main.py \
    --experiment-name "Intermediate_LSTM" \
    --model-type-list "RawToTile_MobileNetV3Large" "TileToTile_LSTM" \
    --min-epochs 1 \
    --max-epochs 4 \
    --batch-size 1 \
    --num-workers 4 \
    --series-length 4 \
    --accumulate-grad-batches 32 \
    --flip-augment \
    --no-freeze-backbone \
    --learning-rate 1e-5 \
    --time-range-min 0 \
    --num-tile-samples 20 \
    --bce-pos-weight 1 \
    --raw-data-path '/root/raw_images' \
    --labels-path '/root/drive_clone_labels'


# Debug Run
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
    
    
# Command Line Options    
#     --experiment-name "resnet50" \
#     --experiment-description "ResNet50 vanilla run" \

#     --raw-data-path '/root/raw_images' \
#     --labels-path '/root/drive_clone_labels'

#     --train-split-path '/userdata/kerasData/data/new_data/mask_rcnn_preprocessed/20210420/train_list.txt' \
#     --val-split-path '/userdata/kerasData/data/new_data/mask_rcnn_preprocessed/20210420/val_list.txt' \
#     --test-split-path '/userdata/kerasData/data/new_data/mask_rcnn_preprocessed/20210420/test_list.txt' \

#     --checkpoint-path './lightning_logs/resnet50/version_9/checkpoints/last.ckpt' \
#     --train-split-path './lightning_logs/resnet50/version_9/train_images.txt' \
#     --val-split-path './lightning_logs/resnet50/version_9/val_images.txt' \
#     --test-split-path './lightning_logs/resnet50/version_9/test_images.txt' \

#     --no-lr-schedule \
#     --no-early-stopping \
#     --no-sixteen-bit \
#     --no-stochastic-weight-avg \
#     --gradient-clip-val 0 \
#     --accumulate-grad-batches 1 \

#     --min-epochs 4 \
#     --max-epochs 8 \
#     --batch-size 8 \
#     --learning-rate 0.01 \
