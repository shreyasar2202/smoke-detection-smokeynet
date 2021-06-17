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
    --omit-images-path '/userdata/kerasData/data/new_data/batched_tiled_data/omit_images.npy' \
    --min-epochs 1 \
    --max-epochs 4 \
    --no-lr-schedule \
    --no-auto-lr-find \
    --no-early-stopping \
    --no-sixteen-bit \
    --no-stochastic-weight-avg \
    --gradient-clip-val 0 \
    --accumulate-grad-batches 1 
#     --train-split-path '/userdata/kerasData/data/new_data/mask_rcnn_preprocessed/20210420/train_list.txt' \
#     --val-split-path '/userdata/kerasData/data/new_data/mask_rcnn_preprocessed/20210420/val_list.txt' \
#     --test-split-path '/userdata/kerasData/data/new_data/mask_rcnn_preprocessed/20210420/test_list.txt' 