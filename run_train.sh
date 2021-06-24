#############################################
# Created by: Anshuman Dewangan
# Date: 2021
#
# Description: Used to easily start training from main.py with command line arguments.
#############################################

# Vanilla run
# python3 main.py \
#     --experiment-name "resnet50" \
#     --experiment-description "ResNet50 vanilla run" \
#     --no-lr-schedule \
#     --no-auto-lr-find \
#     --no-early-stopping \
#     --no-sixteen-bit \
#     --no-stochastic-weight-avg \
#     --gradient-clip-val 0 \
#     --accumulate-grad-batches 1 \
#     --min-epochs 4 \
#     --max-epochs 8 \
#     --batch-size 8 

# Real Run
# python3 main.py \
#     --experiment-name "ResNet50Focal" \
#     --experiment-description "ResNet50Focal full data" \
#     --min-epochs 1 \
#     --max-epochs 50 \
#     --batch-size 8 \
#     --accumulate-grad-batches 10 \
#     --flip-augment \
#     --blur-augment \
#     --no-auto-lr-find \
#     --raw-data-path '/root/raw_images' \
#     --labels-path '/root/drive_clone_labels'

# Debug run
python3 main.py \
    --experiment-name "Debug" \
    --experiment-description "Debug" \
    --min-epochs 1 \
    --max-epochs 1 \
    --batch-size 8 \
    --no-auto-lr-find \
    --raw-data-path '/root/raw_images' \
    --labels-path '/root/drive_clone_labels'
    
    
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
#     --no-auto-lr-find \
#     --no-early-stopping \
#     --no-sixteen-bit \
#     --no-stochastic-weight-avg \
#     --gradient-clip-val 0 \
#     --accumulate-grad-batches 1 \

#     --min-epochs 4 \
#     --max-epochs 8 \
#     --batch-size 8 \
#     --learning-rate 0.01 \
