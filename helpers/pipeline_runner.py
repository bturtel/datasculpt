from typing import List, Dict, Any
import os
import json
import csv
from sculptor.sculptor_pipeline import SculptorPipeline
from helpers.data_sources import BaseDataSource

def run_from_config(
    config_path: str,
    n_workers: int = 1,
    show_progress: bool = True
) -> List[Dict[str, Any]]:
    """Process data using config-specified input source(s) and write to either JSON or CSV."""
    
    # Create pipeline from config
    pipeline = SculptorPipeline.from_config(config_path)
    
    if not pipeline.input_config:
        raise ValueError("No input configuration specified")
    
    # Handle single input or list of inputs
    input_configs = pipeline.input_config if isinstance(pipeline.input_config, list) else [pipeline.input_config]
    
    # Collect data from all sources
    all_data = []
    for input_cfg in input_configs:
        if 'type' not in input_cfg:
            raise ValueError("Each input config must specify a 'type'")
            
        DataSourceClass = BaseDataSource.get_source_class(input_cfg['type'])
        source_config = input_cfg.get('config', {})
        data_source = DataSourceClass(**source_config)
        
        df = data_source.get_data()
        all_data.extend(df.to_dict('records'))
    
    # Process combined data
    results = pipeline.process(all_data, n_workers=n_workers, show_progress=show_progress)
    
    # Save results if output configured
    if pipeline.output_config:
        output_path = pipeline.output_config['path']
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        _, file_extension = os.path.splitext(output_path)
        file_extension = file_extension.lower()

        if file_extension == '.json':
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, default=str)
        elif file_extension == '.csv':
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                if results:
                    fieldnames = list(results[0].keys())
                    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
                    writer.writeheader()
                    writer.writerows(results)
        else:
            raise ValueError(f"Unsupported file extension: {file_extension}")
    
    return results