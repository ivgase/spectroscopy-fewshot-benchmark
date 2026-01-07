#!/usr/bin/env python3
"""
generate_partitions.py

Script to generate partition files (X_supp.csv, X_query.csv, y_supp.csv, y_query.csv)
for all datasets from existing index files (split.csv).

This script should be run after:
1. data/fetch_data.py - to download original data
2. data/transform_ossl_data.py - to generate clean soil datasets

Dependencies:
- pandas
- numpy
- scipy
- pickle

Usage:
    python generate_partitions.py [--dataset DATASET] [--verbose]

Optional arguments:
    --dataset DATASET    Process only a specific dataset (diesel, corn, melamine, eggs, 
                        soil_nir, soil_mir, mango, cgl, shootout, wheat, raman)
    --verbose           Show detailed processing information
"""

import pandas as pd
import numpy as np
import pickle as pkl
import scipy.io
import os
import sys
import argparse
from pathlib import Path


# ============================================================================
# PATH CONFIGURATION
# ============================================================================

BASE_DIR = Path(__file__).parent
DATA_ORIG = BASE_DIR / "data_tmp"
DATA_BASE = BASE_DIR / "data_base"
OUTPUT_DIR = BASE_DIR / "data"

# Index directories (input)
MIXED_DATASET_INDICES = DATA_BASE / "MixedDataset"
SOIL_NIR_INDICES = DATA_BASE / "SoilDataset_NIR"
SOIL_MIR_INDICES = DATA_BASE / "SoilDataset_MIR"
MANGO_BY_YEAR_INDICES = DATA_BASE / "MangoDataset_by_year"
MANGO_BY_YEAR_REGION_INDICES = DATA_BASE / "MangoDataset_by_year-region"

# Output directories
OUTPUT_MIXED = OUTPUT_DIR / "MixedDataset"
OUTPUT_SOIL_NIR = OUTPUT_DIR / "SoilDataset_NIR"
OUTPUT_SOIL_MIR = OUTPUT_DIR / "SoilDataset_MIR"
OUTPUT_MANGO_BY_YEAR = OUTPUT_DIR / "MangoDataset_by_year"
OUTPUT_MANGO_BY_YEAR_REGION = OUTPUT_DIR / "MangoDataset_by_year-region"


# ============================================================================
# DATASET PROCESSING FUNCTIONS
# ============================================================================

def process_diesel(verbose=False):
    """Processes the Diesel dataset."""
    print("\n" + "="*80)
    print("PROCESSING DATASET: DIESEL")
    print("="*80)
    
    # Load original data
    diesel_x = pd.read_csv(DATA_ORIG / "diesel_spec.csv", skiprows=9)
    diesel_x = diesel_x.dropna(axis=1, how='all')
    diesel_x = diesel_x.rename(columns={'Unnamed: 1': 'index'})
    diesel_x.set_index('index', inplace=True)
    
    diesel_y = pd.read_csv(DATA_ORIG / "diesel_prop.csv", skiprows=8)
    diesel_y = diesel_y.dropna(axis=1, how='all')
    diesel_y = diesel_y.rename(columns={'Unnamed: 1': 'index'})
    diesel_y.set_index('index', inplace=True)
    
    if verbose:
        print(f"Data loaded: X={diesel_x.shape}, y={diesel_y.shape}")
    
    # Process each task
    tasks = [d for d in os.listdir(MIXED_DATASET_INDICES) if d.startswith('Diesel_')]
    print(f"Found {len(tasks)} Diesel tasks")
    
    for task in tasks:
        if verbose:
            print(f"\n  Processing: {task}")
        
        # Read indices
        df_indices = pd.read_csv(MIXED_DATASET_INDICES / task / "split.csv", index_col=0)
        
        # Get support/query indices
        supp_idx = df_indices.query("set == 'support'").index.tolist()
        query_idx = df_indices.query("set == 'query'").index.tolist()
        
        # Extract data
        X_supp = diesel_x.loc[supp_idx]
        X_query = diesel_x.loc[query_idx]
        
        # Extract target (property name after "Diesel_")
        property_name = task.split('_')[1]
        y_supp = pd.DataFrame(diesel_y.loc[supp_idx][property_name])
        y_query = pd.DataFrame(diesel_y.loc[query_idx][property_name])
        
        # Save
        output_path = OUTPUT_MIXED / task
        output_path.mkdir(parents=True, exist_ok=True)
        
        X_supp.to_csv(output_path / 'X_supp.csv')
        X_query.to_csv(output_path / 'X_query.csv')
        y_supp.to_csv(output_path / 'y_supp.csv')
        y_query.to_csv(output_path / 'y_query.csv')
        
        if verbose:
            print(f"    X_supp: {X_supp.shape}, X_query: {X_query.shape}")
            print(f"    y_supp: {y_supp.shape}, y_query: {y_query.shape}")
    
    print(f"✓ Diesel: {len(tasks)} tasks processed")


def process_corn(verbose=False):
    """Processes the Corn dataset."""
    print("\n" + "="*80)
    print("PROCESSING DATASET: CORN")
    print("="*80)
    
    # Load original data
    data = scipy.io.loadmat(DATA_ORIG / 'corn.mat')
    propvals = data['propvals']
    data_matrix = propvals['data'][0, 0]
    labels = propvals['label'][0, 0][1][0]
    Y = pd.DataFrame(data_matrix, columns=labels)
    
    if verbose:
        print(f"Available properties: {labels}")
    
    # Process each task
    tasks = [d for d in os.listdir(MIXED_DATASET_INDICES) if d.startswith('Corn_')]
    print(f"Found {len(tasks)} Corn tasks")
    
    for task in tasks:
        if verbose:
            print(f"\n  Processing: {task}")
        
        # Read indices
        df_indices = pd.read_csv(MIXED_DATASET_INDICES / task / "split.csv", index_col=0)
        
        # Get property name
        property_name = task.split('_')[1]#.strip()
        y = pd.DataFrame(Y[property_name])
        
        # Process spectra from 3 instruments
        dfs_query = []
        dfs_support = []
        ys_query = []
        ys_support = []
        
        for instrument in ['m5spec', 'mp5spec', 'mp6spec']:
            x = data[instrument]
            spectra = x['data'][0, 0]
            frequencies = list(x['axisscale'][0, 0][1][0])[0]
            df = pd.DataFrame(spectra, columns=frequencies)
            
            dfs_support.append(df.loc[df_indices.query("set == 'support'").index.tolist()])
            dfs_query.append(df.loc[df_indices.query("set == 'query'").index.tolist()])
            ys_support.append(y.loc[df_indices.query("set == 'support'").index.tolist()])
            ys_query.append(y.loc[df_indices.query("set == 'query'").index.tolist()])
        
        # Concatenate data from 3 instruments
        X_support = pd.concat(dfs_support, axis=0).reset_index(drop=True)
        X_query = pd.concat(dfs_query, axis=0).reset_index(drop=True)
        y_support = pd.concat(ys_support, axis=0).reset_index(drop=True)
        y_query = pd.concat(ys_query, axis=0).reset_index(drop=True)
        
        # Save
        output_path = OUTPUT_MIXED / task
        output_path.mkdir(parents=True, exist_ok=True)
        
        X_support.to_csv(output_path / 'X_supp.csv')
        X_query.to_csv(output_path / 'X_query.csv')
        y_support.to_csv(output_path / 'y_supp.csv')
        y_query.to_csv(output_path / 'y_query.csv')
        
        if verbose:
            print(f"    X_supp: {X_support.shape}, X_query: {X_query.shape}")
            print(f"    y_supp: {y_support.shape}, y_query: {y_query.shape}")
    
    print(f"✓ Corn: {len(tasks)} tasks processed")


def process_melamine(verbose=False):
    """Processes the Melamine dataset."""
    print("\n" + "="*80)
    print("PROCESSING DATASET: MELAMINE")
    print("="*80)
    
    # Load original data
    data = pkl.load(open(DATA_ORIG / 'melamine_data.pkl', 'rb'))
    wl1 = [1/wn*1e7 for wn in data['wn1']]
    wl2 = [1/wn*1e7 for wn in data['wn2']]
    
    # Process each task
    tasks = [d for d in os.listdir(MIXED_DATASET_INDICES) if d.startswith('Melamine_')]
    print(f"Found {len(tasks)} Melamine tasks")
    
    for task in tasks:
        if verbose:
            print(f"\n  Processing: {task}")
        
        # Read indices
        df_indices = pd.read_csv(MIXED_DATASET_INDICES / task / "split.csv", index_col=0)
        
        # Get recipe
        recipe = task.split('_')[1]
        X1 = data[recipe]['X1']
        X2 = data[recipe]['X2']
        y = data[recipe]['Y']
        
        # Convert to DataFrame
        X1 = pd.DataFrame(X1, columns=wl1).iloc[:, ::-1]
        X2 = pd.DataFrame(X2, columns=wl2).iloc[:, ::-1]
        X = pd.concat([X2, X1], axis=1)
        y = pd.DataFrame(y, columns=['y'])
        
        # Extract support/query
        supp_idx = df_indices.query("set == 'support'").index.tolist()
        query_idx = df_indices.query("set == 'query'").index.tolist()
        
        X_supp = X.loc[supp_idx]
        X_query = X.loc[query_idx]
        y_supp = y.loc[supp_idx]
        y_query = y.loc[query_idx]
        
        # Save
        output_path = OUTPUT_MIXED / task
        output_path.mkdir(parents=True, exist_ok=True)
        
        X_supp.to_csv(output_path / 'X_supp.csv')
        X_query.to_csv(output_path / 'X_query.csv')
        y_supp.to_csv(output_path / 'y_supp.csv')
        y_query.to_csv(output_path / 'y_query.csv')
        
        if verbose:
            print(f"    X_supp: {X_supp.shape}, X_query: {X_query.shape}")
            print(f"    y_supp: {y_supp.shape}, y_query: {y_query.shape}")
    
    print(f"✓ Melamine: {len(tasks)} tasks processed")


def process_eggs(verbose=False):
    """Processes the Eggs dataset."""
    print("\n" + "="*80)
    print("PROCESSING DATASET: EGGS")
    print("="*80)
    
    # Load original data
    data = pd.read_csv(DATA_ORIG / 'eggs.csv')
    x = data.drop(columns=["storage_days", "sample"])
    y = pd.DataFrame(data["storage_days"].rename("y"))
    
    # Rename columns (remove "Spectra_" prefix)
    x.columns = x.columns.str.replace(r'^Spectra_(\d+)$', r'\1', regex=True)
    
    if verbose:
        print(f"Data loaded: X={x.shape}, y={y.shape}")
    
    # There is only one Eggs task
    task = 'Eggs'
    if verbose:
        print(f"\n  Processing: {task}")
    
    # Read indices
    df_indices = pd.read_csv(MIXED_DATASET_INDICES / task / "split.csv", index_col=0)
    
    # Extract support/query
    supp_idx = df_indices.query("set == 'support'").index.tolist()
    query_idx = df_indices.query("set == 'query'").index.tolist()
    
    x_supp = x.loc[supp_idx]
    x_query = x.loc[query_idx]
    y_supp = y.loc[supp_idx]
    y_query = y.loc[query_idx]
    
    # Save
    output_path = OUTPUT_MIXED / task
    output_path.mkdir(parents=True, exist_ok=True)
    
    x_supp.to_csv(output_path / 'X_supp.csv')
    x_query.to_csv(output_path / 'X_query.csv')
    y_supp.to_csv(output_path / 'y_supp.csv')
    y_query.to_csv(output_path / 'y_query.csv')
    
    if verbose:
        print(f"    X_supp: {x_supp.shape}, X_query: {x_query.shape}")
        print(f"    y_supp: {y_supp.shape}, y_query: {y_query.shape}")
    
    print(f"✓ Eggs: 1 task processed")


def process_soil_dataset(dataset_type, indices_dir, output_dir, soil_data_path, verbose=False):
    """
    Processes a Soil dataset (NIR or MIR).
    
    Parameters:
    -----------
    dataset_type : str
        Type of dataset ('NIR' or 'MIR')
    indices_dir : Path
        Directory with index files
    output_dir : Path
        Output directory
    soil_data_path : Path
        Path to soil data file
    verbose : bool
        Show detailed information
    """
    print(f"\n  Dataset: {indices_dir.name}")
    
    # Load soil data
    soil_data = pd.read_csv(soil_data_path, index_col=0)
    
    if verbose:
        print(f"  Data loaded: {soil_data.shape}")
    
    # Identify spectral and property columns
    spectral_cols = [col for col in soil_data.columns if col.startswith('scan_')]
    property_cols = [col for col in soil_data.columns if col not in spectral_cols 
                     and col not in ['index', 'Country', 'Continent', 'State', 
                                     'longitude.point_wgs84_dd', 'latitude.point_wgs84_dd']]
    
    # Process each task
    tasks = [d for d in os.listdir(indices_dir) if (indices_dir / d).is_dir()]
    
    for task in tasks:
        split_file = indices_dir / task / "split.csv"
        if not split_file.exists():
            continue
        
        if verbose:
            print(f"    Processing: {task}")
        
        # Read indices
        df_indices = pd.read_csv(split_file, index_col=0)
        
        # Extract support/query indices
        supp_idx = df_indices.query("set == 'support'").index.tolist()
        query_idx = df_indices.query("set == 'query'").index.tolist()
        
        # Extract data
        X_supp = soil_data.loc[supp_idx, spectral_cols]
        X_query = soil_data.loc[query_idx, spectral_cols]
        
        # For "mixed" tasks, the property is in the task name
        # For "by_Prop" tasks, all properties are available
        parts = task.split('-')
        if len(parts) == 2:
            property_name = parts[1]
            # Search for exact column match
            matching_cols = [col for col in property_cols if col.lower() == property_name.lower()]
            if matching_cols:
                y_supp = soil_data.loc[supp_idx, matching_cols]
                y_query = soil_data.loc[query_idx, matching_cols]
            else:
                # If not found, use all properties
                y_supp = soil_data.loc[supp_idx, property_cols]
                y_query = soil_data.loc[query_idx, property_cols]
        else:
            y_supp = soil_data.loc[supp_idx, property_cols]
            y_query = soil_data.loc[query_idx, property_cols]
        
        # Save
        output_path = output_dir / task
        output_path.mkdir(parents=True, exist_ok=True)
        
        X_supp.to_csv(output_path / 'X_supp.csv')
        X_query.to_csv(output_path / 'X_query.csv')
        y_supp.to_csv(output_path / 'y_supp.csv')
        y_query.to_csv(output_path / 'y_query.csv')
        
        # Copy region files if they exist
        region_supp = indices_dir / task / 'region_supp.csv'
        region_query = indices_dir / task / 'region_query.csv'
        
        if region_supp.exists():
            import shutil
            shutil.copy(region_supp, output_path / 'region_supp.csv')
            shutil.copy(region_query, output_path / 'region_query.csv')
        
        if verbose:
            print(f"      X_supp: {X_supp.shape}, X_query: {X_query.shape}")
            print(f"      y_supp: {y_supp.shape}, y_query: {y_query.shape}")
    
    # Copy splits.csv file if it exists
    splits_file = indices_dir / 'splits.csv'
    if splits_file.exists():
        import shutil
        shutil.copy(splits_file, output_dir / 'splits.csv')
    
    print(f"  ✓ {len(tasks)} tasks processed")


def process_soil_nir(verbose=False):
    """Processes the Soil NIR datasets."""
    print("\n" + "="*80)
    print("PROCESSING DATASET: SOIL NIR")
    print("="*80)
    
    soil_nir_path = BASE_DIR / "data_tmp" / "soil_data_transformed_nir.csv"
    
    if not soil_nir_path.exists():
        print(f"⚠ Warning: {soil_nir_path} not found")
        print("  Run first: python transform_soil_data.py")
        return
    
    # Process NIR mixed
    process_soil_dataset('NIR', SOIL_NIR_INDICES, OUTPUT_SOIL_NIR, 
                        soil_nir_path, verbose)


def process_soil_mir(verbose=False):
    """Processes the Soil MIR datasets."""
    print("\n" + "="*80)
    print("PROCESSING DATASET: SOIL MIR")
    print("="*80)
    
    soil_mir_path = BASE_DIR / "data_tmp" / "soil_data_transformed_mir.csv"
    
    if not soil_mir_path.exists():
        print(f"⚠ Warning: {soil_mir_path} not found")
        print("  Run first: python transform_soil_data.py")
        return
    
    # Process MIR mixed
    process_soil_dataset('MIR', SOIL_MIR_INDICES, OUTPUT_SOIL_MIR, 
                        soil_mir_path, verbose)


def process_mango(verbose=False):
    """Processes the Mango datasets."""
    print("\n" + "="*80)
    print("PROCESSING DATASET: MANGO")
    print("="*80)
    
    # Load original data
    data_mango = pd.read_csv(DATA_ORIG / 'mango_data.csv')
    data = data_mango.query('origin == "published"')
    
    nir_columns = [col for col in data.columns if col.isdigit()]
    zero_cols = data.columns[(data == 0).all()].tolist() + ['291']
    
    if verbose:
        print(f"Data loaded: {data.shape}")
        print(f"NIR columns: {len(nir_columns)}")
        print(f"Zero columns: {len(zero_cols)}")
    
    # Process Mango by year-region
    print("\n  Dataset: MangoDataset_by_year-region")
    tasks_year_region = [d for d in os.listdir(MANGO_BY_YEAR_REGION_INDICES) 
                         if (MANGO_BY_YEAR_REGION_INDICES / d).is_dir()]
    
    for task in tasks_year_region:
        split_file = MANGO_BY_YEAR_REGION_INDICES / task / "split.csv"
        if not split_file.exists():
            continue
        
        if verbose:
            print(f"    Processing: {task}")
        
        # Read indices
        df_indices = pd.read_csv(split_file, index_col=0)
        
        # Extract data using indices
        supp_idx = df_indices.query("set == 'support'").index.tolist()
        query_idx = df_indices.query("set == 'query'").index.tolist()
        
        supp_x = data.loc[supp_idx, nir_columns].drop(columns=zero_cols)
        supp_y = data.loc[supp_idx, ['dry_matter']]
        query_x = data.loc[query_idx, nir_columns].drop(columns=zero_cols)
        query_y = data.loc[query_idx, ['dry_matter']]
        
        # Save
        output_path = OUTPUT_MANGO_BY_YEAR_REGION / task
        output_path.mkdir(parents=True, exist_ok=True)
        
        supp_x.to_csv(output_path / 'X_supp.csv')
        supp_y.to_csv(output_path / 'y_supp.csv')
        query_x.to_csv(output_path / 'X_query.csv')
        query_y.to_csv(output_path / 'y_query.csv')
        
        if verbose:
            print(f"      X_supp: {supp_x.shape}, X_query: {query_x.shape}")
    
    # Copy splits.csv
    import shutil
    if (MANGO_BY_YEAR_REGION_INDICES / 'splits.csv').exists():
        shutil.copy(MANGO_BY_YEAR_REGION_INDICES / 'splits.csv', 
                   OUTPUT_MANGO_BY_YEAR_REGION / 'splits.csv')
    
    print(f"  ✓ {len(tasks_year_region)} tasks processed")
    
    # Process Mango by year
    print("\n  Dataset: MangoDataset_by_year")
    tasks_year = [d for d in os.listdir(MANGO_BY_YEAR_INDICES) 
                  if (MANGO_BY_YEAR_INDICES / d).is_dir()]
    
    for task in tasks_year:
        split_file = MANGO_BY_YEAR_INDICES / task / "split.csv"
        if not split_file.exists():
            continue
        
        if verbose:
            print(f"    Processing: {task}")
        
        # Read indices
        df_indices = pd.read_csv(split_file, index_col=0)
        
        # Extract data using indices
        supp_idx = df_indices.query("set == 'support'").index.tolist()
        query_idx = df_indices.query("set == 'query'").index.tolist()
        
        supp_x = data.loc[supp_idx, nir_columns].drop(columns=zero_cols)
        supp_y = data.loc[supp_idx, ['dry_matter']]
        query_x = data.loc[query_idx, nir_columns].drop(columns=zero_cols)
        query_y = data.loc[query_idx, ['dry_matter']]
        
        # Save
        output_path = OUTPUT_MANGO_BY_YEAR / task
        output_path.mkdir(parents=True, exist_ok=True)
        
        supp_x.to_csv(output_path / 'X_supp.csv')
        supp_y.to_csv(output_path / 'y_supp.csv')
        query_x.to_csv(output_path / 'X_query.csv')
        query_y.to_csv(output_path / 'y_query.csv')
        
        if verbose:
            print(f"      X_supp: {supp_x.shape}, X_query: {query_x.shape}")
    
    # Copy splits.csv
    if (MANGO_BY_YEAR_INDICES / 'splits.csv').exists():
        shutil.copy(MANGO_BY_YEAR_INDICES / 'splits.csv', 
                   OUTPUT_MANGO_BY_YEAR / 'splits.csv')
    
    print(f"  ✓ {len(tasks_year)} tasks processed")


def process_cgl(verbose=False):
    """Processes the CGL dataset."""
    print("\n" + "="*80)
    print("PROCESSING DATASET: CGL")
    print("="*80)
    
    # Load original data
    data = scipy.io.loadmat(DATA_ORIG / 'CGL_nir.mat')
    
    def load_nir_dataframe(data_dict, variable='Xcal'):
        """Loads NIR data from .mat file."""
        arr = data_dict[variable]
        data_matrix = arr['data'][0, 0]
        wavelengths = arr['axisscale'][0, 0][1][0][0]
        return pd.DataFrame(data_matrix, columns=wavelengths)
    
    def load_label_dataframe(data_dict, variable='ycal'):
        """Loads labels from .mat file."""
        arr = data_dict[variable]
        data_matrix = arr['data'][0, 0]
        labels = arr['label'][0, 0][1][0]
        return pd.DataFrame(data_matrix, columns=labels)
    
    # Load data
    X_cal = load_nir_dataframe(data, variable='Xcal')
    X_test = load_nir_dataframe(data, variable='Xtest')
    y_cal = load_label_dataframe(data, variable='Ycal')
    y_test = load_label_dataframe(data, variable='Ytest')
    
    # Replace 0 with NaN
    y_cal[y_cal == 0] = np.nan
    y_test[y_test == 0] = np.nan
    
    if verbose:
        print(f"Data loaded: X_cal={X_cal.shape}, X_test={X_test.shape}")
        print(f"Properties: {y_cal.columns.tolist()}")
    
    # Process each property
    tasks_created = 0
    for col in y_cal.columns:
        col_aux = col.split(" ")[0]
        task = f'CGL_{col_aux}'
        
        if verbose:
            print(f"\n  Processing: {task}")
        
        # Support (calibration)
        y_aux = y_cal[col].dropna()
        not_na_index = y_aux.index
        y_supp = pd.DataFrame(y_aux)
        y_supp.columns = [col_aux]
        X_supp = X_cal.loc[not_na_index]
        
        # Query (test)
        y_aux = y_test[col].dropna()
        not_na_index = y_aux.index
        y_query = pd.DataFrame(y_aux)
        y_query.columns = [col_aux]
        X_query = X_test.loc[not_na_index]
        
        # Save
        output_path = OUTPUT_MIXED / task
        output_path.mkdir(parents=True, exist_ok=True)
        
        X_supp.to_csv(output_path / 'X_supp.csv')
        y_supp.to_csv(output_path / 'y_supp.csv')
        X_query.to_csv(output_path / 'X_query.csv')
        y_query.to_csv(output_path / 'y_query.csv')
        
        if verbose:
            print(f"    X_supp: {X_supp.shape}, X_query: {X_query.shape}")
            print(f"    y_supp: {y_supp.shape}, y_query: {y_query.shape}")
        
        tasks_created += 1
    
    print(f"✓ CGL: {tasks_created} tasks processed")


def process_shootout(verbose=False):
    """Processes the Shootout dataset."""
    print("\n" + "="*80)
    print("PROCESSING DATASET: SHOOTOUT")
    print("="*80)
    
    # Load original data
    data = scipy.io.loadmat(DATA_ORIG / 'nir_shootout_2002.mat')
    
    dfs = []
    ys = []
    
    for key in ['calibrate_1', 'calibrate_2', 'test_1', 'test_2', 'validate_1', 'validate_2']:
        nir = data[key]['data'][0, 0].byteswap().newbyteorder()
        wavelengths = data[key]['axisscale'][0, 0][1][0][0].byteswap().newbyteorder()
        nir = pd.DataFrame(nir, columns=wavelengths)
        
        # Determine y name
        if 'calibrate' in key:
            y_name = 'calibrate_Y'
        elif 'test' in key:
            y_name = 'test_Y'
        else:
            y_name = 'validate_Y'
        
        y = data[y_name]['data'][0, 0].byteswap().newbyteorder()
        column_names = data[y_name]['label'][0, 0][1][0].byteswap().newbyteorder()
        y = pd.DataFrame(y, columns=column_names)
        ys.append(y)
        
        nir['particion'] = key
        dfs.append(nir)
    
    df = pd.concat(dfs, axis=0).reset_index(drop=False)
    y = pd.concat(ys, axis=0).reset_index(drop=False)
    
    if verbose:
        print(f"Data loaded: X={df.shape}, y={y.shape}")
        print(f"Properties: {[c for c in y.columns if c != 'index']}")
    
    # Define partitions
    support_part = ['calibrate_1', 'calibrate_2', 'validate_1', 'validate_2']
    query_part = ['test_1', 'test_2']
    
    # Process each property
    tasks_created = 0
    for col in y.columns:
        if col == 'index':
            continue
        
        task = f'Shootout_{col}'
        
        if verbose:
            print(f"\n  Processing: {task}")
        
        # Support
        supp = df.query('particion in @support_part')
        supp = supp.rename(columns={'index': 'sample'})
        y_supp = pd.DataFrame(y.loc[supp.index][col])
        X_supp = supp.drop(columns=['sample', 'particion'])
        
        # Query
        query = df.query('particion in @query_part')
        query = query.rename(columns={'index': 'sample'})
        y_query = pd.DataFrame(y.loc[query.index][col])
        X_query = query.drop(columns=['sample', 'particion'])
        
        # Save
        output_path = OUTPUT_MIXED / task
        output_path.mkdir(parents=True, exist_ok=True)
        
        X_supp.to_csv(output_path / 'X_supp.csv')
        y_supp.to_csv(output_path / 'y_supp.csv')
        X_query.to_csv(output_path / 'X_query.csv')
        y_query.to_csv(output_path / 'y_query.csv')
        
        # Save real indices
        supp_renamed = df.query('particion in @support_part').rename(columns={'index': 'sample'})
        query_renamed = df.query('particion in @query_part').rename(columns={'index': 'sample'})
        
        supp_renamed[["sample", "particion"]].to_csv(output_path / 'real_index_supp.csv')
        query_renamed[["sample", "particion"]].to_csv(output_path / 'real_index_query.csv')
        
        if verbose:
            print(f"    X_supp: {X_supp.shape}, X_query: {X_query.shape}")
            print(f"    y_supp: {y_supp.shape}, y_query: {y_query.shape}")
        
        tasks_created += 1
    
    print(f"✓ Shootout: {tasks_created} tasks processed")


def process_wheat(verbose=False):
    """Processes the Wheat dataset."""
    print("\n" + "="*80)
    print("PROCESSING DATASET: WHEAT")
    print("="*80)
    
    # Load original data
    filename = DATA_ORIG / 'wheat_kernel.xlsx'
    
    calX = pd.read_excel(filename, sheet_name='calibration_X', header=None)
    calY = pd.read_excel(filename, sheet_name='calibration_Y', header=None)
    calY.rename(columns={0: 'y'}, inplace=True)
    testX = pd.read_excel(filename, sheet_name='test_X', header=None)
    testY = pd.read_excel(filename, sheet_name='test_Y', header=None)
    testY.rename(columns={0: 'y'}, inplace=True)
    
    if verbose:
        print(f"Data loaded:")
        print(f"  Calibration: X={calX.shape}, y={calY.shape}")
        print(f"  Test: X={testX.shape}, y={testY.shape}")
    
    # Save
    output_path = OUTPUT_MIXED / 'Wheat'
    output_path.mkdir(parents=True, exist_ok=True)
    
    calX.to_csv(output_path / 'X_supp.csv')
    calY.to_csv(output_path / 'y_supp.csv')
    testX.to_csv(output_path / 'X_query.csv')
    testY.to_csv(output_path / 'y_query.csv')
    
    if verbose:
        print(f"\n  Processing: Wheat")
        print(f"    X_supp: {calX.shape}, X_query: {testX.shape}")
        print(f"    y_supp: {calY.shape}, y_query: {testY.shape}")
    
    print(f"✓ Wheat: 1 task processed")


def process_raman(verbose=False):
    """Processes the Raman dataset."""
    print("\n" + "="*80)
    print("PROCESSING DATASET: RAMAN")
    print("="*80)
    
    def load_and_preprocess_data(filepath, is_train=True):
        """Loads and preprocesses Raman data."""
        if is_train:
            df = pd.read_csv(filepath)
            target_cols = ['Glucose (g/L)', 'Sodium Acetate (g/L)', 'Magnesium Acetate (g/L)']
            y = df[target_cols].dropna().values
            X = df.iloc[:, :-4]
        else:
            df = pd.read_csv(filepath, header=None)
            X = df
            y = None
        
        X.columns = ["sample_id"] + [str(i) for i in range(X.shape[1]-1)]
        X['sample_id'] = X['sample_id'].ffill()
        
        if is_train:
            X['sample_id'] = X['sample_id'].str.strip()
        else:
            X['sample_id'] = X['sample_id'].str.strip().str.replace('sample', '').astype(int)
        
        spectral_cols = X.columns[1:]
        for col in spectral_cols:
            X[col] = X[col].astype(str).str.replace('[', '', regex=False).str.replace(']', '', regex=False)
            X[col] = pd.to_numeric(X[col], errors='coerce')
        
        return X, y
    
    # Load data
    path = 'data/Raman/transfer_plate.csv'
    X_train, y_train = load_and_preprocess_data(path)
    
    target_cols = ['Glucose (g/L)', 'Sodium Acetate (g/L)', 'Magnesium Acetate (g/L)']
    y_train = pd.DataFrame(y_train, columns=target_cols)
    y_train['sample_id'] = [f'sample{i+1}' for i in range(len(y_train))]
    y_train = y_train.loc[y_train.index.repeat(2)].reset_index(drop=True)
    
    if verbose:
        print(f"Data loaded: X={X_train.shape}, y={y_train.shape}")
    
    # Split train/test
    from sklearn.model_selection import train_test_split
    index_train, index_test = train_test_split(X_train.sample_id.unique(), test_size=0.15, random_state=42)
    
    # Process each analyte
    tasks_created = 0
    for col in target_cols:
        task = f'Bio_{col.replace(" (g/L)", "").split(" ")[0]}'
        
        if verbose:
            print(f"\n  Processing: {task}")
        
        # Extract support/query
        y_supp = y_train.query("sample_id in @index_train")[col].rename('y')
        X_supp = X_train.query("sample_id in @index_train").drop(columns=['sample_id'])
        y_query = y_train.query("sample_id in @index_test")[col].rename('y')
        X_query = X_train.query("sample_id in @index_test").drop(columns=['sample_id'])
        
        # Save
        output_path = OUTPUT_MIXED / task
        output_path.mkdir(parents=True, exist_ok=True)
        
        X_supp.to_csv(output_path / 'X_supp.csv', index=True)
        y_supp.to_csv(output_path / 'y_supp.csv', index=True)
        X_query.to_csv(output_path / 'X_query.csv', index=True)
        y_query.to_csv(output_path / 'y_query.csv', index=True)
        
        if verbose:
            print(f"    X_supp: {X_supp.shape}, X_query: {X_query.shape}")
            print(f"    y_supp: {y_supp.shape}, y_query: {y_query.shape}")
        
        tasks_created += 1
    
    print(f"✓ Raman: {tasks_created} tasks processed")


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main function."""
    
    parser = argparse.ArgumentParser(
        description='Generates dataset partitions from existing index files'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['diesel', 'corn', 'melamine', 'eggs', 'soil_nir', 'soil_mir', 
                'mango', 'cgl', 'shootout', 'wheat', 'raman', 'all'],
        default='all',
        help='Dataset to process (default: all)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show detailed processing information'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("DATASET PARTITIONS GENERATION")
    print("="*80)
    print(f"\nBase directory: {BASE_DIR}")
    print(f"Original data: {DATA_ORIG}")
    print(f"Indices: {DATA_BASE}")
    print(f"Output: {OUTPUT_DIR}")
    
    # Verify that the original data directory exists
    if not DATA_ORIG.exists():
        print(f"\n✗ ERROR: Data directory not found: {DATA_ORIG}")
        print("\nRun first: python data/fetch_data.py")
        return 1
    
    # Verify that the index directory exists
    if not DATA_BASE.exists():
        print(f"\n✗ ERROR: Index directory not found: {DATA_BASE}")
        print("\nIndex files must exist in data_base/")
        return 1
    
    # Process datasets according to selected option
    try:
        if args.dataset == 'all' or args.dataset == 'diesel':
            process_diesel(args.verbose)
        
        if args.dataset == 'all' or args.dataset == 'corn':
            process_corn(args.verbose)
        
        if args.dataset == 'all' or args.dataset == 'melamine':
            process_melamine(args.verbose)
        
        if args.dataset == 'all' or args.dataset == 'eggs':
            process_eggs(args.verbose)
        
        if args.dataset == 'all' or args.dataset == 'soil_nir':
            process_soil_nir(args.verbose)
        
        if args.dataset == 'all' or args.dataset == 'soil_mir':
            process_soil_mir(args.verbose)
        
        if args.dataset == 'all' or args.dataset == 'mango':
            process_mango(args.verbose)
        
        if args.dataset == 'all' or args.dataset == 'cgl':
            process_cgl(args.verbose)
        
        if args.dataset == 'all' or args.dataset == 'shootout':
            process_shootout(args.verbose)
        
        if args.dataset == 'all' or args.dataset == 'wheat':
            process_wheat(args.verbose)
        
        if args.dataset == 'all' or args.dataset == 'raman':
            process_raman(args.verbose)
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("\n" + "="*80)
    print("GENERATION COMPLETED")
    print("="*80)
    print("\n✓ All datasets processed successfully")
    print(f"\nGenerated files in: {OUTPUT_DIR}")
    
    return 0


if __name__ == '__main__':
    main()
