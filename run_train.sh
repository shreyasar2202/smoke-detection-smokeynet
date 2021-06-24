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


# Debug run
python3 main.py \
    --experiment-name "ResnetFocal" \
    --experiment-description "ResNet50 w Focal Loss" \
    --min-epochs 1 \
    --max-epochs 100 \
    --batch-size 8 \
    --accumulate-grad-batches 5
    
#     --raw-data-path '~/raw_images'
#     --labels-path '~/drive_clone_labels'
    
#     --train-split-path '/userdata/kerasData/data/new_data/mask_rcnn_preprocessed/20210420/train_list.txt' \
#     --val-split-path '/userdata/kerasData/data/new_data/mask_rcnn_preprocessed/20210420/val_list.txt' \
#     --test-split-path '/userdata/kerasData/data/new_data/mask_rcnn_preprocessed/20210420/test_list.txt' \

#     --checkpoint-path './lightning_logs/resnet50/version_9/checkpoints/last.ckpt' \
#     --train-split-path './lightning_logs/resnet50/version_9/train_images.txt' \
#     --val-split-path './lightning_logs/resnet50/version_9/val_images.txt' \
#     --test-split-path './lightning_logs/resnet50/version_9/test_images.txt' \

#     --is-test 