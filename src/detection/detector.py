import os
import time
import queue
import threading
import logging
from typing import List, Dict, Optional
import yaml
import numpy as np
import onnxruntime as rt
from scapy.all import sniff
from scapy.layers.inet import IP, TCP, UDP
from collections import deque

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MaliciousTrafficDetector:
    """Real-time malicious traffic detector using ONNX model."""

    def __init__(self, config_path: str = "configs/config.yaml", test_mode: bool = False):
        """Initialize the detector with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Initialize packet buffer
        self.window_size = self.config['data']['window_size']
        self.packet_buffer = deque(maxlen=self.window_size)

        # Initialize detection parameters
        self.confidence_threshold = self.config['detection']['confidence_threshold']
        self.interface = self.config['detection']['interface']

        # Setup packet queue and processing thread
        self.packet_queue = queue.Queue()
        self.stop_flag = threading.Event()

        # Statistics
        self.stats = {
            'packets_processed': 0,
            'windows_analyzed': 0,
            'malicious_detected': 0,
            'processing_time': []
        }

        # Load model if not in test mode
        self.test_mode = test_mode
        if not test_mode:
            model_path = os.path.join(
                self.config['data']['models'], 'model.onnx')
            if os.path.exists(model_path):
                self.session = rt.InferenceSession(model_path)
                self.input_name = self.session.get_inputs()[0].name
            else:
                logger.warning(
                    "No model found at %s. Running in feature extraction mode only.", model_path)
                self.session = None
                self.input_name = None

    def extract_features(self, packet) -> Dict:
        """Extract features from a single packet."""
        features = {}

        # Packet length
        features['packet_length'] = len(packet)

        # IP layer features
        if IP in packet:
            features['protocol'] = packet[IP].proto
            features['ttl'] = packet[IP].ttl
            features['src_ip'] = packet[IP].src
            features['dst_ip'] = packet[IP].dst
        else:
            features['protocol'] = 0
            features['ttl'] = 0
            features['src_ip'] = '0.0.0.0'
            features['dst_ip'] = '0.0.0.0'

        # Transport layer features
        if TCP in packet:
            features['src_port'] = packet[TCP].sport
            features['dst_port'] = packet[TCP].dport
            features['tcp_flags'] = packet[TCP].flags
            features['tcp_window'] = packet[TCP].window
        elif UDP in packet:
            features['src_port'] = packet[UDP].sport
            features['dst_port'] = packet[UDP].dport
            features['tcp_flags'] = 0
            features['tcp_window'] = 0
        else:
            features['src_port'] = 0
            features['dst_port'] = 0
            features['tcp_flags'] = 0
            features['tcp_window'] = 0

        return features

    def process_window(self, packets: List) -> Optional[np.ndarray]:
        """Process a window of packets and compute features for model input."""
        if not packets:
            return None

        # Extract features from all packets
        packet_features = [self.extract_features(p) for p in packets]

        # Compute statistical features
        window_features = {}

        # Packet length statistics
        lengths = [f['packet_length'] for f in packet_features]
        window_features['packet_length_mean'] = np.mean(lengths)
        window_features['packet_length_std'] = np.std(lengths)
        window_features['packet_length_min'] = np.min(lengths)
        window_features['packet_length_max'] = np.max(lengths)

        # Inter-arrival time statistics
        if len(packets) > 1:
            arrival_times = [float(p.time) for p in packets]
            inter_arrival_times = np.diff(arrival_times)
            window_features['inter_arrival_time_mean'] = np.mean(
                inter_arrival_times)
            window_features['inter_arrival_time_std'] = np.std(
                inter_arrival_times)
            window_duration = arrival_times[-1] - arrival_times[0]
        else:
            window_features['inter_arrival_time_mean'] = 0.0
            window_features['inter_arrival_time_std'] = 0.0
            window_duration = 1.0  # Default duration for single packet

        # Protocol distribution
        protocols = [f['protocol'] for f in packet_features]
        unique_protocols, counts = np.unique(protocols, return_counts=True)
        window_features['protocol_distribution'] = float(
            len(unique_protocols)) / len(protocols)

        # Port distribution
        ports = [f['src_port'] for f in packet_features] + \
            [f['dst_port'] for f in packet_features]
        unique_ports, counts = np.unique(ports, return_counts=True)
        window_features['port_distribution'] = float(
            len(unique_ports)) / len(ports)

        # TCP flags distribution
        tcp_flags = [f['tcp_flags'] for f in packet_features]
        unique_flags, counts = np.unique(tcp_flags, return_counts=True)
        window_features['tcp_flags_distribution'] = float(
            len(unique_flags)) / len(tcp_flags)

        # Flow-based features
        window_features['flow_duration'] = window_duration
        window_features['flow_bytes_per_second'] = sum(
            lengths) / (window_duration + 1e-6)
        window_features['flow_packets_per_second'] = len(
            packets) / (window_duration + 1e-6)
        window_features['packet_count_per_second'] = len(
            packets) / (window_duration + 1e-6)

        # Convert to numpy array in the correct order
        feature_vector = np.array([float(window_features[f])
                                  for f in self.config['data']['feature_list']], dtype=np.float32)
        return feature_vector.reshape(1, -1)

    def predict(self, features: np.ndarray) -> float:
        """Make prediction using ONNX model."""
        if self.test_mode or self.session is None:
            # Return dummy prediction in test mode
            return 0.5

        start_time = time.time()
        prediction = self.session.run(
            None, {self.input_name: features.astype(np.float32)})[0]
        self.stats['processing_time'].append(time.time() - start_time)
        return prediction[0][0]

    def packet_callback(self, packet):
        """Callback function for packet capture."""
        self.packet_queue.put(packet)
        self.stats['packets_processed'] += 1

    def process_packets(self):
        """Process packets from queue and detect malicious traffic."""
        while not self.stop_flag.is_set():
            try:
                # Get packet from queue
                packet = self.packet_queue.get(timeout=1)
                self.packet_buffer.append(packet)

                # Process window when buffer is full
                if len(self.packet_buffer) == self.window_size:
                    features = self.process_window(list(self.packet_buffer))
                    if features is not None:
                        probability = self.predict(features)
                        self.stats['windows_analyzed'] += 1

                        if probability > self.confidence_threshold:
                            self.stats['malicious_detected'] += 1
                            self.handle_malicious_traffic(
                                list(self.packet_buffer), probability)

                self.packet_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing packet: {str(e)}")

    def handle_malicious_traffic(self, packets: List, probability: float):
        """Handle detected malicious traffic."""
        # Get source and destination information
        src_ips = set()
        dst_ips = set()
        for packet in packets:
            if IP in packet:
                src_ips.add(packet[IP].src)
                dst_ips.add(packet[IP].dst)

        # Log the detection
        logger.warning(
            f"Malicious traffic detected!\n"
            f"Confidence: {probability:.2f}\n"
            f"Source IPs: {src_ips}\n"
            f"Destination IPs: {dst_ips}"
        )

        # Here you could add additional actions like:
        # - Blocking IPs using iptables
        # - Sending alerts
        # - Storing packet data for further analysis

    def start(self):
        """Start the detection system."""
        if not self.test_mode and self.session is None:
            logger.error("Cannot start detection without a trained model")
            return False

        logger.info(f"Starting detection on interface {self.interface}")

        # Start processing thread
        self.processing_thread = threading.Thread(target=self.process_packets)
        self.processing_thread.start()

        # Start packet capture
        try:
            sniff(
                iface=self.interface,
                prn=self.packet_callback,
                store=0,
                stop_filter=lambda _: self.stop_flag.is_set()
            )
        except Exception as e:
            logger.error(f"Error capturing packets: {str(e)}")
        finally:
            self.stop()

    def stop(self):
        """Stop the detection system."""
        logger.info("Stopping detection system...")
        self.stop_flag.set()
        self.processing_thread.join()

        # Log statistics
        avg_processing_time = np.mean(
            self.stats['processing_time']) * 1000 if self.stats['processing_time'] else 0
        logger.info(
            f"Detection system stopped.\n"
            f"Packets processed: {self.stats['packets_processed']}\n"
            f"Windows analyzed: {self.stats['windows_analyzed']}\n"
            f"Malicious traffic detected: {self.stats['malicious_detected']}\n"
            f"Average processing time: {avg_processing_time:.2f} ms"
        )


def main():
    """Main function to run the detector."""
    detector = MaliciousTrafficDetector()

    try:
        detector.start()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
        detector.stop()


if __name__ == "__main__":
    main()
