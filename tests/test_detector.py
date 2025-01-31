import os
import pytest
import numpy as np
from scapy.all import IP, TCP, UDP, Ether
from src.detection.detector import MaliciousTrafficDetector


@pytest.fixture
def detector():
    """Create a detector instance for testing."""
    return MaliciousTrafficDetector(test_mode=True)


@pytest.fixture
def sample_packet():
    """Create a sample packet for testing."""
    return Ether()/IP(src="192.168.1.1", dst="192.168.1.2")/TCP(sport=12345, dport=80)


def test_extract_features(detector, sample_packet):
    """Test feature extraction from a single packet."""
    features = detector.extract_features(sample_packet)

    assert isinstance(features, dict)
    assert 'packet_length' in features
    assert 'protocol' in features
    assert 'src_ip' in features
    assert features['src_ip'] == "192.168.1.1"
    assert features['dst_ip'] == "192.168.1.2"
    assert features['src_port'] == 12345
    assert features['dst_port'] == 80


def test_process_window(detector):
    """Test processing a window of packets."""
    # Create a list of sample packets
    packets = [
        Ether()/IP(src="192.168.1.1", dst="192.168.1.2")/TCP(sport=12345, dport=80),
        Ether()/IP(src="192.168.1.2", dst="192.168.1.3")/UDP(sport=53, dport=53),
        Ether()/IP(src="192.168.1.1", dst="192.168.1.3")/TCP(sport=12346, dport=443)
    ]

    features = detector.process_window(packets)

    assert isinstance(features, np.ndarray)
    assert features.shape[0] == 1  # One window
    assert not np.any(np.isnan(features))  # No NaN values


def test_predict(detector):
    """Test model prediction."""
    # Create dummy features matching the expected shape
    dummy_features = np.random.rand(
        1, len(detector.config['data']['feature_list']))

    # Make prediction
    prediction = detector.predict(dummy_features)

    assert isinstance(prediction, float)
    assert 0 <= prediction <= 1  # Probability should be between 0 and 1


def test_empty_window(detector):
    """Test handling of empty packet windows."""
    features = detector.process_window([])
    assert features is None


def test_malformed_packet(detector):
    """Test handling of malformed packets."""
    # Create a malformed packet (only Ethernet layer)
    malformed_packet = Ether()

    features = detector.extract_features(malformed_packet)

    assert isinstance(features, dict)
    assert features['protocol'] == 0
    assert features['src_ip'] == '0.0.0.0'
    assert features['dst_ip'] == '0.0.0.0'


def test_statistics_tracking(detector):
    """Test statistics tracking functionality."""
    initial_processed = detector.stats['packets_processed']
    initial_windows = detector.stats['windows_analyzed']

    # Process some packets
    packets = [
        Ether()/IP(src="192.168.1.1", dst="192.168.1.2")/TCP(sport=12345, dport=80)
        for _ in range(detector.window_size)
    ]

    for packet in packets:
        detector.packet_callback(packet)

    assert detector.stats['packets_processed'] == initial_processed + \
        len(packets)


def test_config_loading(detector):
    """Test configuration loading."""
    assert isinstance(detector.window_size, int)
    assert isinstance(detector.confidence_threshold, float)
    assert isinstance(detector.interface, str)
    assert isinstance(detector.config['data']['feature_list'], list)


def test_handle_malicious_traffic(detector, caplog):
    """Test malicious traffic handling and logging."""
    packets = [
        Ether()/IP(src="192.168.1.1", dst="192.168.1.2")/TCP(sport=12345, dport=80)
        for _ in range(3)
    ]

    detector.handle_malicious_traffic(packets, 0.95)

    # Check if the warning was logged
    assert any(
        "Malicious traffic detected" in record.message for record in caplog.records)
    assert any("192.168.1.1" in record.message for record in caplog.records)
    assert any("192.168.1.2" in record.message for record in caplog.records)


if __name__ == '__main__':
    pytest.main([__file__])
