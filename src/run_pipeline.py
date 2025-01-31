import os
import logging
import argparse
import yaml
import time
from datetime import datetime
import subprocess
from typing import Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Pipeline:
    """Run the complete malicious traffic detection pipeline."""

    def __init__(self, config_path: str = "configs/config.yaml"):
        """Initialize the pipeline with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.start_time = datetime.now()
        self.pipeline_steps = {
            'download_data': self.download_data,
            'preprocess_data': self.preprocess_data,
            'train_model': self.train_model,
            'run_tests': self.run_tests,
            'start_detection': self.start_detection
        }

    def run_command(self, command: str) -> bool:
        """Run a shell command and return success status."""
        try:
            logger.info(f"Running command: {command}")
            result = subprocess.run(
                command,
                shell=True,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            logger.info(result.stdout)
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Command failed: {str(e)}")
            logger.error(f"Error output: {e.stderr}")
            return False

    def download_data(self) -> bool:
        """Download and prepare the dataset."""
        logger.info("Starting data download and preparation...")
        return self.run_command("python src/data/download_dataset.py")

    def preprocess_data(self) -> bool:
        """Preprocess the raw data."""
        logger.info("Starting data preprocessing...")
        return self.run_command("python src/data/preprocess.py")

    def train_model(self) -> bool:
        """Train and compress the model."""
        logger.info("Starting model training...")
        return self.run_command("python src/models/train.py")

    def run_tests(self) -> bool:
        """Run the test suite."""
        logger.info("Running tests...")
        return self.run_command("pytest tests/")

    def start_detection(self) -> bool:
        """Start the detection system."""
        logger.info("Starting detection system...")

        # Start API server
        api_process = subprocess.Popen(
            ["python", "src/detection/api.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        # Wait a bit for the server to start
        time.sleep(5)

        if api_process.poll() is not None:
            logger.error("API server failed to start")
            return False

        logger.info("Detection system started successfully")
        return True

    def run_pipeline(self, steps: Optional[list] = None) -> bool:
        """Run the complete pipeline or specified steps."""
        if steps is None:
            steps = list(self.pipeline_steps.keys())

        success = True
        for step in steps:
            if step not in self.pipeline_steps:
                logger.error(f"Unknown pipeline step: {step}")
                continue

            logger.info(f"Running pipeline step: {step}")
            start_time = time.time()

            if not self.pipeline_steps[step]():
                logger.error(f"Pipeline step failed: {step}")
                success = False
                break

            duration = time.time() - start_time
            logger.info(f"Completed {step} in {duration:.2f} seconds")

        # Log pipeline completion
        total_duration = (datetime.now() - self.start_time).total_seconds()
        if success:
            logger.info(f"Pipeline completed successfully in {
                        total_duration:.2f} seconds")
        else:
            logger.error(f"Pipeline failed after {total_duration:.2f} seconds")

        return success


def main():
    """Main function to run the pipeline."""
    parser = argparse.ArgumentParser(
        description="Run the malicious traffic detection pipeline")
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=['download_data', 'preprocess_data',
                 'train_model', 'run_tests', 'start_detection'],
        help="Specify which pipeline steps to run"
    )
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to configuration file"
    )

    args = parser.parse_args()

    # Initialize and run pipeline
    pipeline = Pipeline(config_path=args.config)
    success = pipeline.run_pipeline(steps=args.steps)

    # Exit with appropriate status
    exit(0 if success else 1)


if __name__ == "__main__":
    main()
