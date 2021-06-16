# Vanilla run
# python3 train.py \
#     --no-lr-schedule \
#     --no-auto-lr-find \
#     --no-early-stopping \
#     --no-sixteen-bit \
#     --no-stochastic-weight-avg \
#     --gradient-clip-val 0 \
#     --accumulate-grad-batches 1

# Debug run
python3 train.py \
    --min-epochs 1 \
    --max-epochs 2 \
    --no-lr-schedule \
    --no-auto-lr-find \
    --no-early-stopping \
    --no-sixteen-bit \
    --no-stochastic-weight-avg \
    --gradient-clip-val 0 \
    --accumulate-grad-batches 1