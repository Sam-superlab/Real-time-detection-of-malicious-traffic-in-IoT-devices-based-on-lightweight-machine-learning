import os
import requests
import hashlib
import yaml
import logging
from tqdm import tqdm
import pandas as pd
import zipfile
from typing import Optional, List, Dict
import argparse
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatasetDownloader:
    """Download and prepare the dataset."""

    def __init__(self, config_path: str = "configs/config.yaml", dataset: str = "bot-iot"):
        """Initialize the downloader with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.raw_data_path = self.config['data']['raw_data_path']
        os.makedirs(self.raw_data_path, exist_ok=True)

        # Get dataset configuration
        self.datasets = self.config['data']['datasets']
        self.selected_dataset = self.datasets.get(dataset)
        if not self.selected_dataset:
            available_datasets = list(self.datasets.keys())
            raise ValueError(
                f"Unknown dataset: {dataset}. Available datasets: {available_datasets}")

    def download_file(self, url: str, filename: str, use_mirror: bool = False) -> bool:
        """Download a file with progress bar."""
        try:
            # Try primary URL first
            response = requests.get(url, stream=True)
            if response.status_code == 404 and use_mirror and self.selected_dataset.get('mirrors'):
                logger.info(f"Primary URL failed, trying mirror...")
                for mirror in self.selected_dataset['mirrors']:
                    mirror_url = mirror + filename
                    response = requests.get(mirror_url, stream=True)
                    if response.status_code == 200:
                        break

            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))
            block_size = 8192

            with open(filename, 'wb') as f, tqdm(
                total=total_size,
                unit='iB',
                unit_scale=True,
                desc=f"Downloading {os.path.basename(filename)}"
            ) as pbar:
                for data in response.iter_content(block_size):
                    size = f.write(data)
                    pbar.update(size)

            return True

        except Exception as e:
            logger.error(f"Error downloading {filename}: {str(e)}")
            if use_mirror:
                logger.error(
                    "All download attempts failed (primary URL and mirrors)")
            return False

    def extract_dataset(self, zip_file: str, output_dir: str) -> bool:
        """Extract dataset files from zip archive."""
        try:
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                # Extract all files
                zip_ref.extractall(output_dir)
                logger.info(f"Extracted {zip_file} to {output_dir}")
            return True

        except Exception as e:
            logger.error(f"Error extracting {zip_file}: {str(e)}")
            return False

    def verify_dataset(self, dataset_path: str) -> bool:
        """Verify the downloaded dataset structure and contents."""
        try:
            # Check if essential files exist
            required_files = [
                'README.md',
                'data',
                'labels'
            ]

            for file in required_files:
                file_path = os.path.join(dataset_path, file)
                if not os.path.exists(file_path):
                    logger.error(f"Missing required file/directory: {file}")
                    return False

            # Check data directory contents
            data_dir = os.path.join(dataset_path, 'data')
            if len(os.listdir(data_dir)) == 0:
                logger.error("Data directory is empty")
                return False

            return True

        except Exception as e:
            logger.error(f"Error verifying dataset: {str(e)}")
            return False

    def download_dataset(self, use_mirror: bool = False) -> bool:
        """Download the selected dataset."""
        success = True

        # Create dataset-specific directory
        dataset_dir = os.path.join(
            self.raw_data_path, self.selected_dataset['description'].lower().replace(' ', '_'))
        os.makedirs(dataset_dir, exist_ok=True)

        # Download all URLs for the dataset
        for url in self.selected_dataset['urls']:
            filename = os.path.join(dataset_dir, os.path.basename(url))

            if os.path.exists(filename):
                logger.info(f"File already exists: {filename}")
                continue

            logger.info(f"Downloading from {url}...")
            if not self.download_file(url, filename, use_mirror):
                success = False
                continue

            # If it's a zip file, extract it
            if filename.endswith('.zip'):
                if not self.extract_dataset(filename, dataset_dir):
                    success = False
                    continue

                # Optionally remove zip file after extraction
                os.remove(filename)

        # Verify dataset
        if success and not self.verify_dataset(dataset_dir):
            success = False

        return success

    def prepare_dataset(self) -> bool:
        """Prepare the dataset for training."""
        try:
            dataset_dir = os.path.join(
                self.raw_data_path, self.selected_dataset['description'].lower().replace(' ', '_'))

            # Dataset-specific preprocessing steps can be added here
            logger.info("Dataset preparation completed successfully")
            return True

        except Exception as e:
            logger.error(f"Error preparing dataset: {str(e)}")
            return False


def main():
    """Main function to download and prepare the dataset."""
    parser = argparse.ArgumentParser(
        description="Download and prepare dataset")
    parser.add_argument(
        "--dataset",
        choices=['bot-iot', 'n-baiot', 'iot-23', 'ton-iot'],
        default='bot-iot',
        help="Dataset to download (default: bot-iot)"
    )
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--use-mirror",
        action="store_true",
        help="Use mirror URLs if primary download fails"
    )
    args = parser.parse_args()

    try:
        downloader = DatasetDownloader(
            config_path=args.config, dataset=args.dataset)

        # Download dataset
        logger.info(f"Starting {args.dataset} dataset download...")
        if not downloader.download_dataset(use_mirror=args.use_mirror):
            logger.error("Dataset download failed")
            sys.exit(1)

        # Prepare dataset
        logger.info("Preparing dataset...")
        if not downloader.prepare_dataset():
            logger.error("Dataset preparation failed")
            sys.exit(1)

        logger.info("Dataset download and preparation completed successfully!")

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
