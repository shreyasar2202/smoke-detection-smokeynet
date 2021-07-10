#############################################
# Created by: Anshuman Dewangan
# Date: 2021
#
# Description: Used to easily start training from main.py with command line arguments.
#############################################

python3 src/main.py \
    --experiment-name "RawToTile_MobileNetV3LargeRUS" \
    --model-type-list "RawToTile_MobileNetV3Large" \
    --min-epochs 3 \
    --max-epochs 10 \
    --batch-size 8 \
    --series-length 1 \
    --accumulate-grad-batches 4 \
    --no-freeze-backbone \
    --no-pretrain-backbone \
    --flip-augment \
    --blur-augment \
    --bce-pos-weight 1 \
    --num-tile-samples 30 \
    --time-range-min 0 \
    --labels-path '/root/pytorch_lightning_data/drive_clone_labels' \
    --raw-data-path '/root/raw_images'
    
python3 src/main.py \
    --experiment-name "RawToTile_MobileNetV3LargeNoRUS" \
    --model-type-list "RawToTile_MobileNetV3Large" \
    --min-epochs 3 \
    --max-epochs 10 \
    --batch-size 8 \
    --series-length 1 \
    --accumulate-grad-batches 4 \
    --no-freeze-backbone \
    --no-pretrain-backbone \
    --flip-augment \
    --blur-augment \
    --labels-path '/root/pytorch_lightning_data/drive_clone_labels' \
    --raw-data-path '/root/raw_images'

# python3 src/main.py \
#     --test-split-path './data/all_fires.txt' \
#     --train-split-path './data/all_fires.txt' \
#     --val-split-path './data/all_fires.txt' \
#     --checkpoint-path './saved_logs/version_49/checkpoints/epoch=1-step=449.ckpt' \
#     --raw-data-path '/root/raw_images' \
#     --labels-path '/root/drive_clone_labels'


# python3 src/main.py \
#     --experiment-name "MobileToDeiT_PretrainedNoSupervision" \
#     --model-type-list "RawToTile_MobileNetV3Large" "TileToTile_DeiT" \
#     --min-epochs 3 \
#     --max-epochs 5 \
#     --batch-size 2 \
#     --series-length 1 \
#     --accumulate-grad-batches 16 \
#     --no-freeze-backbone \
#     --no-pretrain-backbone \
#     --flip-augment \
#     --blur-augment \
#     --no-early-stopping \
#     --no-lr-schedule \
#     --no-intermediate-supervision \
#     --model-pretrain-epochs 3 0 \
#     --labels-path '/root/pytorch_lightning_data/drive_clone_labels' \
#     --raw-data-path '/root/raw_images'


# python3 src/main.py \
#     --experiment-name "MobileEmbeddings_DeiT" \
#     --model-type-list "TileToTile_DeiT" \
#     --min-epochs 3 \
#     --max-epochs 5 \
#     --batch-size 1 \
#     --series-length 1 \
#     --accumulate-grad-batches 32 \
#     --no-freeze-backbone \
#     --no-pretrain-backbone \
#     --flip-augment \
#     --blur-augment \
#     --no-early-stopping \
#     --no-lr-schedule \
#     --labels-path '/root/pytorch_lightning_data/drive_clone_labels' \
#     --embeddings-path '/root/pytorch_lightning_data/embeddings'

# python3 src/main.py \
#     --experiment-name "MobileEmbeddings_LSTM" \
#     --model-type-list "TileToTile_LSTM" \
#     --min-epochs 3 \
#     --max-epochs 5 \
#     --batch-size 1 \
#     --series-length 1 \
#     --accumulate-grad-batches 32 \
#     --no-freeze-backbone \
#     --no-pretrain-backbone \
#     --flip-augment \
#     --blur-augment \
#     --no-early-stopping \
#     --no-lr-schedule \
#     --labels-path '/root/pytorch_lightning_data/drive_clone_labels' \
#     --embeddings-path '/root/pytorch_lightning_data/embeddings'

# python3 src/main.py \
#     --experiment-name "MobileEmbeddings_DeiTLSTM" \
#     --model-type-list "TileToTile_DeiT" "TileToTile_LSTM" \
#     --min-epochs 3 \
#     --max-epochs 5 \
#     --batch-size 1 \
#     --series-length 1 \
#     --accumulate-grad-batches 32 \
#     --no-freeze-backbone \
#     --no-pretrain-backbone \
#     --flip-augment \
#     --blur-augment \
#     --no-early-stopping \
#     --no-lr-schedule \
#     --labels-path '/root/pytorch_lightning_data/drive_clone_labels' \
#     --embeddings-path '/root/pytorch_lightning_data/embeddings'

# python3 src/main.py \
#     --experiment-name "MobileEmbeddings_LSTMDeiT" \
#     --model-type-list "TileToTile_LSTM" "TileToTile_DeiT" \
#     --min-epochs 3 \
#     --max-epochs 5 \
#     --batch-size 1 \
#     --series-length 1 \
#     --accumulate-grad-batches 32 \
#     --no-freeze-backbone \
#     --no-pretrain-backbone \
#     --flip-augment \
#     --blur-augment \
#     --no-early-stopping \
#     --no-lr-schedule \
#     --labels-path '/root/pytorch_lightning_data/drive_clone_labels' \
#     --embeddings-path '/root/pytorch_lightning_data/embeddings'

# python3 src/main.py \
#     --experiment-name "MobileEmbeddings_DeiTLinear" \
#     --model-type-list "TileToTile_DeiT" "TileToImage_Linear" \
#     --min-epochs 3 \
#     --max-epochs 5 \
#     --batch-size 1 \
#     --series-length 1 \
#     --accumulate-grad-batches 32 \
#     --no-freeze-backbone \
#     --no-pretrain-backbone \
#     --flip-augment \
#     --blur-augment \
#     --no-early-stopping \
#     --no-lr-schedule \
#     --labels-path '/root/pytorch_lightning_data/drive_clone_labels' \
#     --embeddings-path '/root/pytorch_lightning_data/embeddings'
    


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
