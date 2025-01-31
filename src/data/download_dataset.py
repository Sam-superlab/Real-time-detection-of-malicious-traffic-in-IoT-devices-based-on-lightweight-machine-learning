import os
import requests
import hashlib
import yaml
import logging
from tqdm import tqdm
import pandas as pd
import zipfile
from typing import Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatasetDownloader:
    """Download and prepare the CIC-IDS2017 dataset."""

    def __init__(self, config_path: str = "configs/config.yaml"):
        """Initialize the downloader with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.raw_data_path = self.config['data']['raw_data_path']
        os.makedirs(self.raw_data_path, exist_ok=True)

        # Dataset URLs and checksums
        self.datasets = {
            'monday': {
                'url': 'https://example.com/cicids2017/monday.pcap.zip',  # Replace with actual URL
                'md5': 'abc123',  # Replace with actual MD5
                'filename': 'monday.pcap.zip'
            },
            'tuesday': {
                'url': 'https://example.com/cicids2017/tuesday.pcap.zip',
                'md5': 'def456',
                'filename': 'tuesday.pcap.zip'
            },
            # Add more days as needed
        }

    def download_file(self, url: str, filename: str) -> bool:
        """Download a file with progress bar."""
        try:
            response = requests.get(url, stream=True)
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
            return False

    def verify_checksum(self, filename: str, expected_md5: str) -> bool:
        """Verify file integrity using MD5 checksum."""
        md5_hash = hashlib.md5()

        try:
            with open(filename, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    md5_hash.update(chunk)

            calculated_md5 = md5_hash.hexdigest()
            return calculated_md5 == expected_md5

        except Exception as e:
            logger.error(f"Error verifying checksum for {filename}: {str(e)}")
            return False

    def extract_pcap(self, zip_file: str, output_dir: str) -> bool:
        """Extract PCAP files from zip archive."""
        try:
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                # Extract only .pcap files
                for file in zip_ref.namelist():
                    if file.endswith('.pcap'):
                        zip_ref.extract(file, output_dir)
                        logger.info(f"Extracted {file}")
            return True

        except Exception as e:
            logger.error(f"Error extracting {zip_file}: {str(e)}")
            return False

    def download_datasets(self):
        """Download all dataset files."""
        for day, info in self.datasets.items():
            output_file = os.path.join(self.raw_data_path, info['filename'])

            # Skip if file exists and checksum matches
            if os.path.exists(output_file) and self.verify_checksum(output_file, info['md5']):
                logger.info(
                    f"Skipping {day}, file already exists and checksum matches")
                continue

            # Download file
            logger.info(f"Downloading {day} dataset...")
            if self.download_file(info['url'], output_file):
                # Verify checksum
                if self.verify_checksum(output_file, info['md5']):
                    logger.info(
                        f"Successfully downloaded and verified {day} dataset")

                    # Extract PCAP files
                    if self.extract_pcap(output_file, self.raw_data_path):
                        logger.info(f"Successfully extracted {day} dataset")
                        # Optionally remove zip file after extraction
                        os.remove(output_file)
                    else:
                        logger.error(f"Failed to extract {day} dataset")
                else:
                    logger.error(
                        f"Checksum verification failed for {day} dataset")
                    os.remove(output_file)
            else:
                logger.error(f"Failed to download {day} dataset")

    def prepare_labels(self):
        """Prepare labels for the dataset using provided CSV files."""
        # This would typically involve:
        # 1. Reading the label CSV files
        # 2. Matching timestamps with PCAP files
        # 3. Creating a mapping of packet/flow -> label
        # 4. Saving the labels in a format that can be used during training
        pass


def main():
    """Main function to download and prepare the dataset."""
    downloader = DatasetDownloader()

    # Download datasets
    logger.info("Starting dataset download...")
    downloader.download_datasets()

    # Prepare labels
    logger.info("Preparing dataset labels...")
    downloader.prepare_labels()

    logger.info("Dataset preparation completed!")


if __name__ == "__main__":
    main()
