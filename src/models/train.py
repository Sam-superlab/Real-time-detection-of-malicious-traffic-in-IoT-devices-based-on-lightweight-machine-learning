import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import yaml
import joblib
import onnx
import onnxmltools
from onnxmltools.convert.lightgbm.operator_converters.LightGbm import convert_lightgbm
import onnxruntime as rt
from typing import Dict, Tuple
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Train and optimize LightGBM model for malicious traffic detection."""

    def __init__(self, config_path: str = "configs/config.yaml"):
        """Initialize the model trainer with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.model_params = self.config['model']['params']
        self.training_params = self.config['training']
        self.compression_params = self.config['model']['compression']

    def prepare_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load and prepare data for training."""
        processed_data_path = self.config['data']['processed_data_path']

        # Load all processed feature files
        dfs = []
        for filename in os.listdir(processed_data_path):
            if filename.endswith('_features.parquet'):
                file_path = os.path.join(processed_data_path, filename)
                df = pd.read_parquet(file_path)
                dfs.append(df)

        # Combine all data
        data = pd.concat(dfs, ignore_index=True)

        # Prepare features and labels
        X = data.drop('label', axis=1)  # Assuming 'label' column exists
        y = data['label']

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config['data']['test_data_ratio'],
            random_state=self.training_params['random_seed']
        )

        return X_train, X_test, y_train, y_test

    def train_model(self, X_train: np.ndarray, X_test: np.ndarray,
                    y_train: np.ndarray, y_test: np.ndarray) -> Tuple[lgb.Booster, Dict]:
        """Train LightGBM model and return model and metrics."""
        logger.info("Starting model training...")

        # Create dataset
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

        # Train model
        model = lgb.train(
            params=self.model_params,
            train_set=train_data,
            valid_sets=[valid_data],
            num_boost_round=self.training_params['num_epochs'],
            early_stopping_rounds=self.training_params['early_stopping_rounds']
        )

        # Evaluate model
        y_pred = model.predict(X_test)
        y_pred_binary = (y_pred > 0.5).astype(int)

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred_binary),
            'precision': precision_score(y_test, y_pred_binary),
            'recall': recall_score(y_test, y_pred_binary),
            'f1': f1_score(y_test, y_pred_binary),
            'auc': roc_auc_score(y_test, y_pred)
        }

        logger.info(f"Training completed. Metrics: {metrics}")

        return model, metrics

    def compress_model(self, model: lgb.Booster) -> bytes:
        """Compress model using ONNX quantization."""
        logger.info("Starting model compression...")

        # Convert to ONNX
        initial_types = [('input', onnxmltools.convert.common.data_types.FloatTensorType(
            [None, model.num_feature()]))]
        onnx_model = convert_lightgbm(
            model, initial_types=initial_types, target_opset=12)

        # Quantize model if enabled
        if self.compression_params['enable_quantization']:
            from onnxmltools.utils.float16_converter import convert_float_to_float16
            onnx_model = convert_float_to_float16(onnx_model)

        # Get compressed model size
        compressed_size = len(onnx_model.SerializeToString()
                              ) / (1024 * 1024)  # Size in MB
        logger.info(f"Compressed model size: {compressed_size:.2f} MB")

        return onnx_model.SerializeToString()

    def save_model(self, model: lgb.Booster, compressed_model: bytes):
        """Save both original and compressed models."""
        models_path = os.path.join(self.config['data']['models'])
        os.makedirs(models_path, exist_ok=True)

        # Save original model
        original_path = os.path.join(models_path, 'model.txt')
        model.save_model(original_path)

        # Save compressed model
        compressed_path = os.path.join(models_path, 'model.onnx')
        with open(compressed_path, 'wb') as f:
            f.write(compressed_model)

        logger.info(f"Models saved to {models_path}")

    def verify_inference(self, compressed_model: bytes, X_test: np.ndarray):
        """Verify inference speed and accuracy of compressed model."""
        logger.info("Verifying inference performance...")

        # Create ONNX inference session
        session = rt.InferenceSession(compressed_model)
        input_name = session.get_inputs()[0].name

        # Test inference time
        import time
        times = []
        for _ in range(100):
            start = time.time()
            _ = session.run(None, {input_name: X_test[0:1].astype(np.float32)})
            times.append(time.time() - start)

        avg_inference_time = np.mean(times) * 1000  # Convert to ms
        logger.info(f"Average inference time: {avg_inference_time:.2f} ms")

        return avg_inference_time


def main():
    """Main function to train and compress model."""
    trainer = ModelTrainer()

    # Prepare data
    X_train, X_test, y_train, y_test = trainer.prepare_data()

    # Train model
    model, metrics = trainer.train_model(X_train, X_test, y_train, y_test)

    # Compress model
    compressed_model = trainer.compress_model(model)

    # Save models
    trainer.save_model(model, compressed_model)

    # Verify inference
    avg_inference_time = trainer.verify_inference(compressed_model, X_test)

    # Log final results
    logger.info("Training completed successfully!")
    logger.info(f"Final metrics: {metrics}")
    logger.info(f"Average inference time: {avg_inference_time:.2f} ms")


if __name__ == "__main__":
    main()
