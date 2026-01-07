#!/usr/bin/env python3
"""
Script to transform the original OSSL dataset into clean NIR and MIR versions
using previously generated mapping files.

Required input files:
- data_prueba_new/ossl_all_L0_v1.2.csv (original dataset)
- data_base/index_country_mapping.csv (indices and countries for NIR)
- data_base/index_country_soil_mir.csv (indices and countries for MIR)
- data_base/mapping_soil_mir.json (column mapping for MIR)

Output files:
- data/soil_data_transformed_nir.csv (transformed NIR dataset)
- data/soil_data_transformed_mir.csv (transformed MIR dataset)
"""

import pandas as pd
import json
import os


# ============================================================================
# CONFIGURATION
# ============================================================================

DATASET_ORIG = "data_tmp/ossl_all_L0_v1.2.csv"
INDEX_COUNTRY_NIR = "data_base/index_country_mapping.csv"
INDEX_COUNTRY_MIR = "data_base/index_country_soil_mir.csv"
MAPPING_MIR_JSON = "data_base/mapping_soil_mir.json"

OUTPUT_NIR = "data_tmp/soil_data_transformed_nir.csv"
OUTPUT_MIR = "data_tmp/soil_data_transformed_mir.csv"


# ============================================================================
# COLUMN MAPPINGS
# ============================================================================

# NIR mapping (100% verified)
COLUMN_MAPPING_NIR = {
    'k.ext_usda.a725_cmolc.kg': 'K',
    'n.tot_iso.11261_w.pct': 'N',
    'p.ext_iso.11263_mg.kg': 'P',
    'ph.cacl2_iso.10390_index': 'ph_CaCl2',
    'ph.h2o_iso.10390_index': 'pH_h2o',
    'caco3_iso.10693_w.pct': 'CaCO3',
    'oc_iso.10694_w.pct': 'OC',
    'clay.tot_iso.11277_w.pct': 'Clay',
    'silt.tot_iso.11277_w.pct': 'Silt',
    'sand.tot_iso.11277_w.pct': 'Sand',
    'cf_iso.11464_w.pct': 'CF',
    'ec_iso.11265_ds.m': 'EC',
    'cec_iso.11260_cmolc.kg': 'CEC',
}


# ============================================================================
# AUXILIARY FUNCTIONS
# ============================================================================

def load_mapping_mir():
    """Loads the MIR mapping from the JSON file."""
    with open(MAPPING_MIR_JSON, 'r', encoding='utf-8') as f:
        mapping = json.load(f)
    print(f"✓ MIR mapping loaded: {len(mapping)} columns")
    return mapping


def get_spectral_columns(df, spectral_type='visnir'):
    """
    Gets the spectral columns of the specified type.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with the original data
    spectral_type : str
        Type of spectral columns: 'visnir' or 'mir'
    
    Returns:
    --------
    list : List of spectral column names
    """
    pattern = f'scan_{spectral_type}'
    spectral_cols = [col for col in df.columns if pattern in col.lower()]
    
    # Filter columns by spectral range for VIS-NIR (400-2500 nm)
    if spectral_type == 'visnir':
        filtered_cols = []
        for col in spectral_cols:
            try:
                # Extract wavelength from column name
                # Format: scan_visnir.XXX_ref
                wavelength = int(col.split('.')[1].split('_')[0])
                if 400 <= wavelength <= 2500:
                    filtered_cols.append(col)
            except (IndexError, ValueError):
                # If wavelength cannot be extracted, include the column
                filtered_cols.append(col)
        spectral_cols = filtered_cols
    
    return sorted(spectral_cols)


def apply_column_mapping(df_orig, indices, column_mapping):
    """
    Applies column mapping to an original OSSL DataFrame.
    
    Parameters:
    -----------
    df_orig : pd.DataFrame
        Original OSSL DataFrame
    indices : array-like
        Row indices to select from df_orig
    column_mapping : dict
        Dictionary with mapping {original_column: new_column}
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with columns renamed according to the mapping
    """
    # Verify that all mapping columns exist
    missing_cols = [col for col in column_mapping.keys() if col not in df_orig.columns]
    if missing_cols:
        raise ValueError(f"Columns not found in df_orig: {missing_cols}")
    
    # Select rows
    df_selected = df_orig.iloc[indices].copy()
    
    # Select and rename columns
    df_result = df_selected[list(column_mapping.keys())].copy()
    df_result.columns = [column_mapping[col] for col in df_result.columns]
    
    # Reset index
    df_result = df_result.reset_index(drop=True)
    
    return df_result


def transform_dataset(df_orig, index_country_file, column_mapping, spectral_type='visnir'):
    """
    Transforms the original dataset according to the provided indices and mapping.
    
    Parameters:
    -----------
    df_orig : pd.DataFrame
        Original OSSL DataFrame
    index_country_file : str
        Path to CSV file with 'index' and 'Country' columns
    column_mapping : dict
        Dictionary with property column mapping
    spectral_type : str
        Type of spectral data: 'visnir' or 'mir'
    
    Returns:
    --------
    pd.DataFrame
        Transformed DataFrame with all columns
    """
    print(f"\n{'='*80}")
    print(f"Transforming {spectral_type.upper()} dataset")
    print('='*80)
    
    # 1. Load indices and countries
    print(f"\n1. Loading indices and countries from {index_country_file}...")
    index_country = pd.read_csv(index_country_file)
    print(f"   ✓ {len(index_country)} rows loaded")
    
    # Determine available columns
    has_state = 'State' in index_country.columns
    has_continent = 'Continent' in index_country.columns
    
    # Build list of geographic columns
    geo_cols = ['index', 'Country']
    if has_continent:
        geo_cols.append('Continent')
    if has_state:
        geo_cols.append('State')
    
    print(f"   ✓ Geographic columns: {geo_cols}")
    
    # 2. Get spectral columns
    print(f"\n2. Getting {spectral_type} spectral columns...")
    spectral_cols = get_spectral_columns(df_orig, spectral_type)
    print(f"   ✓ {len(spectral_cols)} spectral columns found")
    if spectral_cols:
        print(f"   ✓ Range: {spectral_cols[0]} to {spectral_cols[-1]}")
    
    # 3. Extract spectral data
    print(f"\n3. Extracting spectral data...")
    indices = index_country['index'].values
    df_spectral = df_orig.iloc[indices][spectral_cols].copy()
    df_spectral = df_spectral.reset_index(drop=True)
    print(f"   ✓ Shape: {df_spectral.shape}")
    
    # 4. Apply property mapping
    print(f"\n4. Applying property mapping...")
    df_properties = apply_column_mapping(df_orig, indices, column_mapping)
    print(f"   ✓ {len(column_mapping)} properties mapped")
    print(f"   ✓ Shape: {df_properties.shape}")
    
    # 5. Add coordinates if MIR
    if spectral_type == 'mir':
        print(f"\n5. Extracting geographic coordinates...")
        coord_cols = ['longitude.point_wgs84_dd', 'latitude.point_wgs84_dd']
        df_coords = df_orig.iloc[indices][coord_cols].copy()
        df_coords = df_coords.reset_index(drop=True)
        print(f"   ✓ Coordinates extracted: {df_coords.shape}")
    else:
        df_coords = None
    
    # 6. Combine all
    step = 6 if spectral_type == 'mir' else 5
    print(f"\n{step}. Combining datasets...")
    
    # Create DataFrame with geographic information
    df_geo = index_country[geo_cols].copy()
    df_geo = df_geo.reset_index(drop=True)
    
    # Concatenate according to type
    if spectral_type == 'mir':
        # MIR: index, Country, State, coordinates, properties, spectral data
        df_final = pd.concat([
            df_geo,
            df_coords,
            df_properties,
            df_spectral
        ], axis=1)
    else:
        # NIR: index, Country, [Continent], properties, spectral data
        df_final = pd.concat([
            df_geo,
            df_properties,
            df_spectral
        ], axis=1)
    
    print(f"   ✓ Final dataset: {df_final.shape}")
    print(f"   ✓ Columns: {list(df_final.columns[:10])}... (showing first 10)")
    
    return df_final


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main function of the script."""
    
    print("\n" + "="*80)
    print("SOIL DATASETS TRANSFORMATION")
    print("="*80)
    
    # Verify that input files exist
    print("\nVerifying input files...")
    required_files = [
        DATASET_ORIG,
        INDEX_COUNTRY_NIR,
        INDEX_COUNTRY_MIR,
        MAPPING_MIR_JSON,
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print("✗ ERROR: Missing files:")
        for f in missing_files:
            print(f"  - {f}")
        return 1
    
    print("✓ All input files found")
    
    # Load original dataset
    print(f"\nLoading original dataset from {DATASET_ORIG}...")
    df_orig = pd.read_csv(DATASET_ORIG)
    print(f"✓ Dataset loaded: {df_orig.shape}")
    
    # Load MIR mapping
    mapping_mir = load_mapping_mir()
    
    # ========================================================================
    # TRANSFORM NIR
    # ========================================================================
    
    print("\n" + "="*80)
    print("PROCESSING NIR DATASET")
    print("="*80)
    
    df_nir_transformed = transform_dataset(
        df_orig=df_orig,
        index_country_file=INDEX_COUNTRY_NIR,
        column_mapping=COLUMN_MAPPING_NIR,
        spectral_type='visnir'
    )
    
    # Save transformed NIR
    print(f"\nSaving transformed NIR dataset to {OUTPUT_NIR}...")
    os.makedirs(os.path.dirname(OUTPUT_NIR), exist_ok=True)
    df_nir_transformed.to_csv(OUTPUT_NIR, index=False)
    print(f"✓ File saved: {OUTPUT_NIR}")
    
    # ========================================================================
    # TRANSFORM MIR
    # ========================================================================
    
    print("\n" + "="*80)
    print("PROCESSING MIR DATASET")
    print("="*80)
    
    df_mir_transformed = transform_dataset(
        df_orig=df_orig,
        index_country_file=INDEX_COUNTRY_MIR,
        column_mapping=mapping_mir,
        spectral_type='mir'
    )
    
    # Save transformed MIR
    print(f"\nSaving transformed MIR dataset to {OUTPUT_MIR}...")
    os.makedirs(os.path.dirname(OUTPUT_MIR), exist_ok=True)
    df_mir_transformed.to_csv(OUTPUT_MIR, index=False)
    print(f"✓ File saved: {OUTPUT_MIR}")
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    
    print("\nGenerated files:")
    print(f"  ✓ {OUTPUT_NIR} (shape: {df_nir_transformed.shape})")
    print(f"  ✓ {OUTPUT_MIR} (shape: {df_mir_transformed.shape})")
    
    print("\n✓ Transformation completed successfully")
    
    return 0


if __name__ == "__main__":
    exit(main())
