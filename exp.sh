#!/bin/bash
which python
# Define model variants
MODELS=("bilstm" "bilstm_crf" "transformer" "tcn" "mamba")

# Define hidden sizes to try
HIDDENS=(64 128 256)

# Training parameters
EPOCHS=30
BATCH_SIZE=32
LR=5e-3

# Loop over models and hidden dims
for MODEL in "${MODELS[@]}"; do
  for H in "${HIDDENS[@]}"; do
    echo "Running model: $MODEL, hidden_dim: $H"
    python train.py \
      --model "$MODEL" \
      --hidden_dim "$H" \
      --epochs "$EPOCHS" \
      --batch_size "$BATCH_SIZE" \
      --learning_rate "$LR" \
      --output_dir "checkpoints/${MODEL}_h${H}"
  done
done