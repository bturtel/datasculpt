from typing import List, Dict, Any
from sculptor import SculptorPipeline
import argparse
from dotenv import load_dotenv
import os

def run_from_config(
    config_path: str,
    n_workers: int = 1,
    show_progress: bool = True
) -> List[Dict[str, Any]]:
    """Process data using config-specified input source(s) and write to either JSON or CSV."""

    # Create pipeline from config
    pipeline = SculptorPipeline.from_config(config_path)

    # Use the built-in process_from_config method
    results = pipeline.process_from_config(n_workers=n_workers, show_progress=show_progress)

    return results

def main():
    """Command line interface for the Sculptor pipeline."""
    # Load environment variables from .env file
    load_dotenv()

    parser = argparse.ArgumentParser(description="Run Sculptor pipeline from config file")
    parser.add_argument(
        "config_path",
        type=str,
        help="Path to the configuration file"
    )
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=1,
        help="Number of workers (default: 1)"
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bar"
    )

    args = parser.parse_args()
    
    results = run_from_config(
        config_path=args.config_path,
        n_workers=args.workers,
        show_progress=not args.no_progress
    )
    
    print(f"Processing complete. Generated {len(results)} results.")

if __name__ == "__main__":
    main()