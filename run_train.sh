#############################################
# Created by: Anshuman Dewangan
# Date: 2021
#
# Description: Used to easily start training from main.py with command line arguments.
#############################################

python3 src/main.py \
    --experiment-name "MobileToDeiT_NoSupervision" \
    --model-type-list "RawToTile_MobileNetV3Large" "TileToTile_DeiT" \
    --min-epochs 3 \
    --max-epochs 5 \
    --batch-size 2 \
    --series-length 1 \
    --accumulate-grad-batches 16 \
    --no-freeze-backbone \
    --no-pretrain-backbone \
    --flip-augment \
    --blur-augment \
    --no-early-stopping \
    --no-lr-schedule \
    --learning-rate 0.001 \
    --no-intermediate-supervision \
    --labels-path '/root/pytorch_lightning_data/drive_clone_labels' \
    --raw-data-path '/root/raw_images'
    
python3 src/main.py \
    --experiment-name "MobileToDeiT_PretrainedNoSupervision" \
    --model-type-list "RawToTile_MobileNetV3Large" "TileToTile_DeiT" \
    --min-epochs 3 \
    --max-epochs 5 \
    --batch-size 2 \
    --series-length 1 \
    --accumulate-grad-batches 16 \
    --no-freeze-backbone \
    --no-pretrain-backbone \
    --flip-augment \
    --blur-augment \
    --no-early-stopping \
    --no-lr-schedule \
    --learning-rate 0.001 \
    --no-intermediate-supervision \
    --model-pretrain-epochs 2 0 \
    --labels-path '/root/pytorch_lightning_data/drive_clone_labels' \
    --raw-data-path '/root/raw_images'
    
python3 src/main.py \
    --experiment-name "MobileToDeiT_Pretrained" \
    --model-type-list "RawToTile_MobileNetV3Large" "TileToTile_DeiT" \
    --min-epochs 3 \
    --max-epochs 5 \
    --batch-size 2 \
    --series-length 1 \
    --accumulate-grad-batches 16 \
    --no-freeze-backbone \
    --no-pretrain-backbone \
    --flip-augment \
    --blur-augment \
    --no-early-stopping \
    --no-lr-schedule \
    --learning-rate 0.001 \
    --model-pretrain-epochs 2 0 \
    --labels-path '/root/pytorch_lightning_data/drive_clone_labels' \
    --raw-data-path '/root/raw_images'
    
python3 src/main.py \
    --experiment-name "MobileToDeiT" \
    --model-type-list "RawToTile_MobileNetV3Large" "TileToTile_DeiT" \
    --min-epochs 3 \
    --max-epochs 5 \
    --batch-size 2 \
    --series-length 1 \
    --accumulate-grad-batches 16 \
    --no-freeze-backbone \
    --no-pretrain-backbone \
    --flip-augment \
    --blur-augment \
    --no-early-stopping \
    --no-lr-schedule \
    --learning-rate 0.001 \
    --labels-path '/root/pytorch_lightning_data/drive_clone_labels' \
    --raw-data-path '/root/raw_images'


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

    
#########################
## Command Line Options
#########################

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
