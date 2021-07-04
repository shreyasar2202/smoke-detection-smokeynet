python3 src/main.py \
    --experiment-name 'Focal' \
    --model-type-list "RawToTile_MobileNetV3Large" \
    --min-epochs 1 \
    --max-epochs 4 \
    --batch-size 1 \
    --accumulate-grad-batches 512 \
    --flip-augment \
    --no-freeze-backbone \
    --tile-loss-type 'focal' \
    --optimizer-type 'SGD' \
    --learning-rate 1e-2 \
    --raw-data-path '/root/raw_images' \
    --labels-path '/root/drive_clone_labels'
    
python3 src/main.py \
    --experiment-name 'Focal' \
    --model-type-list "RawToTile_MobileNetV3Large" \
    --min-epochs 1 \
    --max-epochs 4 \
    --batch-size 1 \
    --accumulate-grad-batches 512 \
    --flip-augment \
    --no-freeze-backbone \
    --tile-loss-type 'focal' \
    --optimizer-type 'SGD' \
    --learning-rate 1e-3 \
    --raw-data-path '/root/raw_images' \
    --labels-path '/root/drive_clone_labels'