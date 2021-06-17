#############################################
# Created by: Anshuman Dewangan
# Date: 2021
#
# Description: Used to easily start training from main.py with command line arguments.
#############################################

# Vanilla run
# python3 main.py \
#     --no-lr-schedule \
#     --no-auto-lr-find \
#     --no-early-stopping \
#     --no-sixteen-bit \
#     --no-stochastic-weight-avg \
#     --gradient-clip-val 0 \
#     --accumulate-grad-batches 1

# Debug run
python3 main.py \
    --experiment-name "debug" \
    --min-epochs 1 \
    --max-epochs 1 \
    --no-lr-schedule \
    --no-auto-lr-find \
    --no-early-stopping \
    --no-sixteen-bit \
    --no-stochastic-weight-avg \
    --gradient-clip-val 0 \
    --accumulate-grad-batches 1 
#     --experiment-description "this is a test" \
#     --train-split-path '/userdata/kerasData/data/new_data/mask_rcnn_preprocessed/20210420/train_list.txt' \
#     --val-split-path '/userdata/kerasData/data/new_data/mask_rcnn_preprocessed/20210420/val_list.txt' \
#     --test-split-path '/userdata/kerasData/data/new_data/mask_rcnn_preprocessed/20210420/test_list.txt' 