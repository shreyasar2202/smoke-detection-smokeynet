# Vanilla run
python3 train.py \
    --no-lr-schedule \
    --no-auto-lr-find \
    --no-early-stopping \
    --no-sixteen-bit \
    --no-stochastic-weight-avg \
    --gradient_clip_val 0 \
    --accumulate-grad-batches 1
