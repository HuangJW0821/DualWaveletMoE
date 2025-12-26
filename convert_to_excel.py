#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
import pandas as pd
import argparse
import os

def convert_benchmark_to_excel(json_file_path, output_excel_path=None):
    """
    Convert benchmark JSON result to Excel format
    
    Args:
        json_file_path: Path to the JSON benchmark result file
        output_excel_path: Path to output Excel file (optional, defaults to same name as JSON with .xlsx extension)
    """
    # Read JSON file
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Prepare data for DataFrame
    rows = []
    
    # Add per-dataset results (only time_seq_loss, matching the image format)
    if 'per_dataset_results' in data:
        for dataset_name, dataset_data in data['per_dataset_results'].items():
            if 'time_seq_loss' in dataset_data:
                mse = dataset_data['time_seq_loss'].get('mse', None)
                mae = dataset_data['time_seq_loss'].get('mae', None)
                
                # Add MSE row (with dataset name)
                if mse is not None:
                    rows.append({
                        'Dataset': dataset_name,
                        'Metric': 'MSE',
                        'Value': mse
                    })
                # Add MAE row (without dataset name, leave empty)
                if mae is not None:
                    rows.append({
                        'Dataset': '',  # Empty for MAE row
                        'Metric': 'MAE',
                        'Value': mae
                    })
    
    # Add average rows at the bottom (from benchmark_result)
    if 'benchmark_result' in data and 'time_seq_loss' in data['benchmark_result']:
        benchmark_mse = data['benchmark_result']['time_seq_loss'].get('mse', None)
        benchmark_mae = data['benchmark_result']['time_seq_loss'].get('mae', None)
        
        if benchmark_mse is not None:
            rows.append({
                'Dataset': 'avg',
                'Metric': 'MSE',
                'Value': benchmark_mse
            })
        if benchmark_mae is not None:
            rows.append({
                'Dataset': '',  # Empty for MAE row
                'Metric': 'MAE',
                'Value': benchmark_mae
            })
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    
    # Determine output path
    if output_excel_path is None:
        base_name = os.path.splitext(json_file_path)[0]
        output_excel_path = f"{base_name}.xlsx"
    
    # Save to Excel
    df.to_excel(output_excel_path, index=False, engine='openpyxl')
    print(f"Successfully converted to Excel: {output_excel_path}")
    print(f"Total rows: {len(df)}")
    
    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert benchmark JSON result to Excel')
    parser.add_argument('json_file', type=str, help='Path to JSON benchmark result file')
    parser.add_argument('-o', '--output', type=str, default=None, help='Output Excel file path (optional)')
    
    args = parser.parse_args()
    
    convert_benchmark_to_excel(args.json_file, args.output)

