import os
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from scapy.all import rdpcap
from scapy.layers.inet import IP, TCP, UDP
import yaml
from tqdm import tqdm


class PacketProcessor:
    """Process network packets and extract features for malicious traffic detection."""

    def __init__(self, config_path: str = "configs/config.yaml"):
        """Initialize the packet processor with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.window_size = self.config['data']['window_size']
        self.feature_list = self.config['data']['feature_list']

    def extract_basic_features(self, packet) -> Dict:
        """Extract basic features from a single packet."""
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

    def process_packet_window(self, packets: List) -> Dict:
        """Process a window of packets and compute statistical features."""
        if not packets:
            return None

        # Extract basic features for all packets
        packet_features = [self.extract_basic_features(p) for p in packets]

        # Compute statistical features
        window_features = {}

        # Packet length statistics
        lengths = [f['packet_length'] for f in packet_features]
        window_features['packet_length_mean'] = np.mean(lengths)
        window_features['packet_length_std'] = np.std(lengths)
        window_features['packet_length_min'] = np.min(lengths)
        window_features['packet_length_max'] = np.max(lengths)

        # Protocol distribution
        protocols = [f['protocol'] for f in packet_features]
        protocol_dist = pd.Series(protocols).value_counts(normalize=True)
        window_features['protocol_distribution'] = protocol_dist.to_dict()

        # Port distribution
        src_ports = [f['src_port'] for f in packet_features]
        dst_ports = [f['dst_port'] for f in packet_features]
        port_dist = pd.Series(
            src_ports + dst_ports).value_counts(normalize=True)
        window_features['port_distribution'] = port_dist.to_dict()

        # TCP flags distribution
        tcp_flags = [f['tcp_flags'] for f in packet_features]
        tcp_flags_dist = pd.Series(tcp_flags).value_counts(normalize=True)
        window_features['tcp_flags_distribution'] = tcp_flags_dist.to_dict()

        # Flow-based features
        window_features['flow_duration'] = len(packets)
        window_features['flow_bytes_per_second'] = sum(
            lengths) / (len(packets) + 1e-6)
        window_features['flow_packets_per_second'] = len(packets)

        return window_features

    def process_pcap_file(self, pcap_path: str) -> pd.DataFrame:
        """Process a PCAP file and return a DataFrame with features."""
        print(f"Processing PCAP file: {pcap_path}")

        # Read packets from PCAP file
        packets = rdpcap(pcap_path)

        # Process packets in windows
        windows = []
        for i in tqdm(range(0, len(packets), self.window_size)):
            window = packets[i:i + self.window_size]
            features = self.process_packet_window(window)
            if features:
                windows.append(features)

        # Convert to DataFrame
        df = pd.DataFrame(windows)

        return df

    def save_features(self, df: pd.DataFrame, output_path: str):
        """Save processed features to a file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_parquet(output_path)
        print(f"Saved processed features to: {output_path}")


def main():
    """Main function to process PCAP files."""
    processor = PacketProcessor()

    # Process raw PCAP files
    raw_data_path = processor.config['data']['raw_data_path']
    processed_data_path = processor.config['data']['processed_data_path']

    for filename in os.listdir(raw_data_path):
        if filename.endswith('.pcap'):
            pcap_path = os.path.join(raw_data_path, filename)
            output_path = os.path.join(
                processed_data_path,
                filename.replace('.pcap', '_features.parquet')
            )

            # Process PCAP file
            df = processor.process_pcap_file(pcap_path)

            # Save features
            processor.save_features(df, output_path)


if __name__ == "__main__":
    main()
