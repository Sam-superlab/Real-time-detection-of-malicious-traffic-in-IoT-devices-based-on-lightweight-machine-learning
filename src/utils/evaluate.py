import os
import sys
import time
import logging
import argparse
import yaml
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import onnxruntime as rt
from typing import Dict, Tuple, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluate model performance on test data."""

    def __init__(self, config_path: str = "configs/config.yaml"):
        """Initialize the evaluator with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

    def load_test_data(self, test_data_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load test data and labels."""
        try:
            df = pd.read_parquet(test_data_path)
            X = df.drop('label', axis=1).values.astype(np.float32)
            y = df['label'].values
            return X, y
        except Exception as e:
            logger.error(f"Error loading test data: {str(e)}")
            sys.exit(1)

    def load_model(self, model_path: str) -> rt.InferenceSession:
        """Load ONNX model."""
        try:
            session = rt.InferenceSession(model_path)
            return session
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            sys.exit(1)

    def evaluate_performance(self, session: rt.InferenceSession, X: np.ndarray, y: np.ndarray,
                             batch_size: int = 32) -> Dict:
        """Evaluate model performance metrics."""
        try:
            input_name = session.get_inputs()[0].name
            predictions = []
            inference_times = []

            # Process batches
            for i in range(0, len(X), batch_size):
                batch = X[i:i + batch_size]

                # Measure inference time
                start_time = time.time()
                pred = session.run(None, {input_name: batch})[0]
                inference_times.append(time.time() - start_time)

                predictions.extend(pred)

            # Convert predictions to binary
            y_pred = (np.array(predictions) > 0.5).astype(int)

            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y, y_pred),
                'precision': precision_score(y, y_pred),
                'recall': recall_score(y, y_pred),
                'f1': f1_score(y, y_pred),
                'auc_roc': roc_auc_score(y, predictions),
                'avg_inference_time_ms': np.mean(inference_times) * 1000,
                'max_inference_time_ms': np.max(inference_times) * 1000,
                'min_inference_time_ms': np.min(inference_times) * 1000,
                'std_inference_time_ms': np.std(inference_times) * 1000
            }

            return metrics

        except Exception as e:
            logger.error(f"Error evaluating performance: {str(e)}")
            sys.exit(1)

    def evaluate_resource_usage(self, session: rt.InferenceSession, X: np.ndarray,
                                batch_size: int = 32) -> Dict:
        """Evaluate resource usage during inference."""
        try:
            import psutil
            import os

            process = psutil.Process(os.getpid())
            input_name = session.get_inputs()[0].name

            # Measure baseline memory
            baseline_memory = process.memory_info().rss / (1024 * 1024)  # MB

            # Warm-up run
            _ = session.run(None, {input_name: X[:batch_size]})

            # Measure peak memory and CPU usage during inference
            peak_memory = baseline_memory
            cpu_percentages = []

            for i in range(0, len(X), batch_size):
                batch = X[i:i + batch_size]

                # Measure CPU usage
                cpu_start = process.cpu_percent()
                _ = session.run(None, {input_name: batch})
                cpu_percentages.append(process.cpu_percent() - cpu_start)

                # Update peak memory
                current_memory = process.memory_info().rss / (1024 * 1024)
                peak_memory = max(peak_memory, current_memory)

            resources = {
                'baseline_memory_mb': baseline_memory,
                'peak_memory_mb': peak_memory,
                'memory_increase_mb': peak_memory - baseline_memory,
                'avg_cpu_percent': np.mean(cpu_percentages),
                'max_cpu_percent': np.max(cpu_percentages)
            }

            return resources

        except Exception as e:
            logger.error(f"Error evaluating resource usage: {str(e)}")
            return {}

    def print_results(self, metrics: Dict, resources: Optional[Dict] = None):
        """Print evaluation results in a formatted way."""
        print("\n" + "="*50)
        print("Performance Metrics:")
        print("="*50)
        print(f"Accuracy:     {metrics['accuracy']*100:.1f}%")
        print(f"Precision:    {metrics['precision']*100:.1f}%")
        print(f"Recall:       {metrics['recall']*100:.1f}%")
        print(f"F1-Score:     {metrics['f1']*100:.1f}%")
        print(f"AUC-ROC:      {metrics['auc_roc']:.3f}")
        print("\nInference Time:")
        print(f"Average:      {metrics['avg_inference_time_ms']:.2f}ms")
        print(f"Maximum:      {metrics['max_inference_time_ms']:.2f}ms")
        print(f"Minimum:      {metrics['min_inference_time_ms']:.2f}ms")
        print(f"Std Dev:      {metrics['std_inference_time_ms']:.2f}ms")

        if resources:
            print("\n" + "="*50)
            print("Resource Usage:")
            print("="*50)
            print(f"Baseline Memory:  {resources['baseline_memory_mb']:.1f}MB")
            print(f"Peak Memory:      {resources['peak_memory_mb']:.1f}MB")
            print(f"Memory Increase:  {resources['memory_increase_mb']:.1f}MB")
            print(f"Avg CPU Usage:    {resources['avg_cpu_percent']:.1f}%")
            print(f"Max CPU Usage:    {resources['max_cpu_percent']:.1f}%")


def main():
    """Main function to evaluate model performance."""
    parser = argparse.ArgumentParser(
        description="Evaluate model performance on test data")
    parser.add_argument(
        "--test_data",
        required=True,
        help="Path to test data file"
    )
    parser.add_argument(
        "--model_path",
        required=True,
        help="Path to ONNX model file"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for inference"
    )
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to configuration file"
    )
    args = parser.parse_args()

    try:
        evaluator = ModelEvaluator(config_path=args.config)

        # Load test data and model
        logger.info("Loading test data and model...")
        X, y = evaluator.load_test_data(args.test_data)
        session = evaluator.load_model(args.model_path)

        # Evaluate performance
        logger.info("Evaluating model performance...")
        metrics = evaluator.evaluate_performance(
            session, X, y, args.batch_size)

        # Evaluate resource usage
        logger.info("Evaluating resource usage...")
        resources = evaluator.evaluate_resource_usage(
            session, X, args.batch_size)

        # Print results
        evaluator.print_results(metrics, resources)

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
