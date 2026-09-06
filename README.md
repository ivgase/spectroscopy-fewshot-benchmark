# A Task-Oriented Few-Shot Spectroscopy Benchmark: Empirical Evaluation of Meta-Learning Approaches
A unified, task-oriented benchmark for few-shot regression in NIR/MIR spectroscopy, providing standardized preprocessing, task definitions, and reference implementations for transfer learning and meta-learning methods.

## Dataset Generation

To generate the complete benchmark dataset, follow these steps in order:

### 1. Download raw data

Run the `fetch_data.py` script to download all original datasets from their public sources:

```bash
python fetch_data.py
```

This script will automatically download the following datasets:
- **Mango**: Mango quality prediction data
- **Melamine**: Melamine-formaldehyde resin dataset
- **Corn**: Corn properties (moisture, oil, protein, starch)
- **Diesel**: Diesel properties (BP50, CN, density, flash point, freeze point, total content, viscosity)
- **Eggs**: Eggs dataset
- **Wheat kernel**: Wheat kernel dataset
- **CGL**: CGL NIR dataset
- **NIR Shootout 2002**: NIR Shootout 2002 dataset
- **OSSL**: Open Soil Spectral Library (MIR and NIR soil data)

Files will be saved in the `data_tmp/` directory. Some compressed files (.zip, .gz) are automatically decompressed. Failed or incomplete downloads do not replace existing destination files.

### Download and prepare only openly licensed sources

Run these commands from the repository root:

```bash
# Preview selection, licenses and source URLs without downloading or writing files
python fetch_data.py --open-only --dry-run

python fetch_data.py --open-only
python transform_soil_data.py
python generate_partitions.py --open-only
```

This includes **Mango, Melamine, Eggs, Wheat Kernel and OSSL** and excludes
**Diesel, Corn, CGL and NIR Shootout**. “Open” means a reviewed open license
allowing commercial reuse subject to its conditions, including attribution;
it does not mean license-free. See [dataset licenses and source evidence](DATA_LICENSES.md).

Raw files still go to `data_tmp/`. Existing raw files are not removed by the
filter. Partition generation applies the same selection even when excluded raw
files are present. OSSL transformation uses L0; both original OSSL downloads are
retained for compatibility.

Open-only partitions default to `data_open/`, and TRIP's `splits.csv` contains only
selected tasks with their original split assignments. The output directory must
be empty to prevent mixing previous tasks; choose a new directory for a rerun:

```bash
python generate_partitions.py --open-only --output-dir data_open_run2
# A single eligible dataset can also be prepared in a fresh directory
python generate_partitions.py --open-only --dataset eggs --output-dir data_open_eggs
# Training accepts the generated dataset path
python train_protonet.py --dataset data_open/TRIP
```

An excluded `--dataset` combined with `--open-only` is an error. `--cleanup` is
available only for a complete, unfiltered run, since the shared raw directory
may contain unprocessed datasets. Missing required input files cause a nonzero
exit status. The filtered dataset is a **reduced benchmark**, not a reproduction
of the complete benchmark's results or sample counts.

Without `--open-only`, the scripts continue to select all sources and generate
partitions in `data/` by default.

### 2. Transform soil data

OSSL soil data requires special processing. Run:

```bash
python transform_soil_data.py
```

This script:
- Reads the raw OSSL data (`data_tmp/ossl_all_L0_v1.2.csv`)
- Applies transformations and mappings defined in `data_base/`
- Generates two clean datasets:
  - `data_tmp/soil_data_transformed_nir.csv` (NIR soil data)
  - `data_tmp/soil_data_transformed_mir.csv` (MIR soil data)

### 3. Generate partitions

Finally, generate the train/validation partitions for all datasets:

```bash
python generate_partitions.py
```

This script:
- Reads the predefined partition indices from `data_base/`
- Processes all downloaded datasets
- Generates only tasks listed in each benchmark `splits.csv` (185 Soil NIR and
  704 Soil MIR tasks); discarded candidate index directories are ignored
- Generates the final partitions in the `data/` directory
- Creates for each task the following files:
  - `X_supp.csv`: Support spectra
  - `y_supp.csv`: Support labels
  - `X_query.csv`: Query spectra
  - `y_query.csv`: Query labels

Optionally, you can process only a specific dataset:

```bash
python generate_partitions.py --dataset soil_mir --verbose
```

Available datasets: `diesel`, `corn`, `melamine`, `eggs`, `soil_nir`, `soil_mir`, `mango`, `cgl`, `shootout`, `wheat`.

---

## Training Models

This benchmark includes implementations of several few-shot learning methods. All models support the following datasets:

| Dataset Name | Path | Description |
|--------------|------|-------------|
| `TRIP` | data/TRIP | Mixed spectroscopy benchmark (default) |
| `Mango_y` | data/MangoDataset_by_year | Mango dataset split by year |
| `Mango_yr` | data/MangoDataset_by_year-region | Mango dataset split by year-region |
| `Soil_MIR` | data/SoilDataset_MIR | Soil MIR spectroscopy |
| `Soil_NIR` | data/SoilDataset_NIR | Soil NIR spectroscopy |

### MAML (Model-Agnostic Meta-Learning)

```bash
python train_maml.py \
    --dataset TRIP \
    --episodes 50000 \
    --update_lr 0.1 \
    --meta_lr 0.001 \
    --update_step 5 \
    --update_step_test 10 \
    --k_spt 25 \
    --k_qry 25 \
    --second_order \
    --output results/MAML
```

**Key parameters:**
- `--update_lr`: Inner loop learning rate (task adaptation)
- `--meta_lr`: Outer loop learning rate (meta-optimization)
- `--update_step`: Inner loop steps during training
- `--update_step_test`: Inner loop steps at test time
- `--second_order`: Enable second-order gradients (full MAML)
- `--savgol`: Apply Savitzky-Golay preprocessing

### ProtoNet (Prototypical Networks)

```bash
python train_protonet.py \
    --dataset TRIP \
    --episodes 5000 \
    --lr 0.005 \
    --k_spt 25 \
    --k_qry 25 \
    --embedding after \
    --output results/ProtoNet
```

**Key parameters:**
- `--lr`: Learning rate for encoder training
- `--embedding`: Embedding layer for prototype computation (`after` or `before`)
- `--dist_temp`: Temperature for distance-based weighting

### Fine-Tuning (FT)

```bash
python train_tf.py \
    --dataset TRIP \
    --epochs 500 \
    --lr 0.09 \
    --lr_adapt 0.09 \
    --epochs_adapt 10 \
    --k_spt 25 \
    --k_qry 25 \
    --output results/FT
```

**Key parameters:**
- `--epochs`: Number of pretraining epochs
- `--lr`: Learning rate for pretraining
- `--lr_adapt`: Learning rate for test-time adaptation
- `--epochs_adapt`: Number of adaptation epochs at test time

### SNAIL (Simple Neural Attentive Meta-Learner)

```bash
python train_snail.py \
    --dataset TRIP \
    --epochs 10000 \
    --lr 0.0001 \
    --shots 25 \
    --shots_test 25 \
    --cuda \
    --exp results/SNAIL
```

**Key parameters:**
- `--epochs`: Number of training epochs
- `--shots`: Support samples during training
- `--shots_test`: Support samples at test time
- `--cuda`: Enable GPU training

### Base (Individual Training)

Trains a model from scratch on each test task (no meta-learning):

```bash
python train_individual.py \
    --dataset TRIP \
    --epochs 10 \
    --lr 0.00001 \
    --k_spt 25 \
    --k_qry 25 \
    --output results/Base
```

### Common Parameters

All training scripts share these parameters:

| Parameter | Description |
|-----------|-------------|
| `--dataset` | Dataset name or path |
| `--k_spt` | Number of support samples per task |
| `--k_qry` | Number of query samples per task |
| `--repeats` | Number of experiment repetitions |
| `--output` | Output directory for results |
| `--load_weights` | Path to pretrained weights (optional) |

### Running All Models

A convenience script is provided to run all models with benchmark hyperparameters:

```bash
sbatch run_all_models.sh
```

Or run directly:

```bash
bash run_all_models.sh
```

---

## Offline checks

With `pandas`, `numpy`, `scipy`, `requests`, `tqdm` and `openpyxl` installed:

```bash
python -m unittest discover -s tests -v
```

Tests mock network responses and use temporary synthetic inputs; no full dataset
download is required.
