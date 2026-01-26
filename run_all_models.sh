#!/bin/bash

#SBATCH --job-name all_models
#SBATCH --partition dgx2,dgx
#SBATCH --gres=gpu:1
#SBATCH --output outputs/all_models.txt
#SBATCH --error outputs/error/all_models.txt
#SBATCH --mem 50000
#SBATCH --mail-type=ALL
#SBATCH --mail-user=e.ivangarzon98@go.ugr.es

export PATH="/opt/anaconda/anaconda3/bin:$PATH"
export PATH="/opt/anaconda/bin:$PATH"
eval "$(conda shell.bash hook)"

conda activate /mnt/homeGPU/igarzon/Meta-Learning/metaenv_prueba

# ==============================================================================
# Hyperparameters from JSON configuration
# ==============================================================================
# Common:
#   - Spectral preprocessing: Savitzky-Golay, derivative_order=1, window_size=15, poly_order=2
#   - batch_size=32, meta_batch_size=25
#   - k_spt=25, k_qry=25
#
# Dataset name mapping (defined in each training script):
#   TRIP      -> data/TRIP
#   Mango_y   -> data/MangoDataset_by_year
#   Mango_yr  -> data/MangoDataset_by_year-region
#   Soil_MIR  -> data/SoilDataset_MIR
#   Soil_NIR  -> data/SoilDataset_NIR
# ==============================================================================

DATASET="TRIP"  # Use dataset name instead of path
K_SPT=25
K_QRY=25
REPEATS=3
OUTPUT_DIR="results/benchmark"

# ==============================================================================
# FT (Fine-Tuning) - train_tf.py
# ==============================================================================
# Training: optimizer=Adam, lr=0.09, epochs=500
# Test-time adaptation: enabled=true, optimizer=Adam, lr=0.09, epochs=10
# ==============================================================================
echo "=========================================="
echo "Running FT (Fine-Tuning)..."
echo "=========================================="
python train_tf.py \
    --output ${OUTPUT_DIR}/FT \
    --epochs 500 \
    --lr 0.09 \
    --lr_adapt 0.09 \
    --epochs_adapt 10 \
    --k_spt ${K_SPT} \
    --k_qry ${K_QRY} \
    --repeats ${REPEATS} \
    --dataset ${DATASET}

# ==============================================================================
# MAML - train_maml.py
# ==============================================================================
# Training: optimizer=Adam, outer_lr=0.001, episodes=50000
# Test-time adaptation: enabled=true, optimizer=SGD, inner_lr=0.1, epochs=10
# Note: --update_lr is the inner learning rate used during adaptation
#       --meta_lr is the outer learning rate for meta-optimization
# ==============================================================================
echo "=========================================="
echo "Running MAML..."
echo "=========================================="
python train_maml.py \
    --output ${OUTPUT_DIR}/MAML \
    --episodes 50000 \
    --update_lr 0.1 \
    --meta_lr 0.001 \
    --update_step 5 \
    --update_step_test 10 \
    --k_spt ${K_SPT} \
    --k_qry ${K_QRY} \
    --grad_clip 1.0 \
    --second_order \
    --savgol \
    --repeats ${REPEATS} \
    --dataset ${DATASET}

# ==============================================================================
# ProtoNet - train_protonet.py
# ==============================================================================
# Training: optimizer=Adam, lr=0.005, episodes=5000
# Test-time adaptation: enabled=false
# ==============================================================================
echo "=========================================="
echo "Running ProtoNet..."
echo "=========================================="
python train_protonet.py \
    --output ${OUTPUT_DIR}/ProtoNet \
    --episodes 5000 \
    --lr 0.005 \
    --k_spt ${K_SPT} \
    --k_qry ${K_QRY} \
    --embedding after \
    --savgol \
    --repeats ${REPEATS} \
    --dataset ${DATASET}

# ==============================================================================
# SNAIL - train_snail.py
# ==============================================================================
# Training: optimizer=Adam, lr=0.0001, epochs=10000
# Test-time adaptation: enabled=false
# Note: SNAIL uses --epochs for training iterations
# ==============================================================================
echo "=========================================="
echo "Running SNAIL..."
echo "=========================================="
python train_snail.py \
    --exp ${OUTPUT_DIR}/SNAIL \
    --epochs 10000 \
    --lr 0.0001 \
    --shots ${K_SPT} \
    --shots_test ${K_SPT} \
    --queries 1 \
    --queries_test 1 \
    --batch_size 32 \
    --task_batch 25 \
    --cuda \
    --repeats ${REPEATS} \
    --dataset ${DATASET}

# ==============================================================================
# Base/Individual - train_individual.py
# ==============================================================================
# Training: None (only test-time adaptation)
# Test-time adaptation: enabled=true, optimizer=Adam, lr=0.00001, epochs=10
# Note: This model trains from scratch on each test task
# ==============================================================================
echo "=========================================="
echo "Running Base (Individual)..."
echo "=========================================="
python train_individual.py \
    --output ${OUTPUT_DIR}/Base \
    --epochs 10 \
    --lr 0.00001 \
    --k_spt ${K_SPT} \
    --k_qry ${K_QRY} \
    --repeats ${REPEATS} \
    --dataset ${DATASET}

echo "=========================================="
echo "All models completed!"
echo "=========================================="
