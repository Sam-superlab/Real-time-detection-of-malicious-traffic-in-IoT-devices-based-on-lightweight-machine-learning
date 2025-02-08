// Chart.js Configuration for Traffic Visualization
const createTrafficChart = () => {
    const ctx = document.getElementById('trafficChart').getContext('2d');
    window.trafficChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Normal Traffic',
                data: [],
                borderColor: '#00C851',
                backgroundColor: 'rgba(0, 200, 81, 0.1)',
                fill: true
            }, {
                label: 'Suspicious Traffic',
                data: [],
                borderColor: '#ff4444',
                backgroundColor: 'rgba(255, 68, 68, 0.1)',
                fill: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: {
                duration: 750,
                easing: 'easeInOutQuart'
            },
            scales: {
                x: {
                    display: true,
                    title: {
                        display: true,
                        text: 'Time (s)'
                    }
                },
                y: {
                    display: true,
                    title: {
                        display: true,
                        text: 'Packets/s'
                    },
                    beginAtZero: true
                }
            },
            plugins: {
                legend: {
                    position: 'top',
                },
                tooltip: {
                    mode: 'index',
                    intersect: false,
                }
            }
        }
    });
};

// Performance Chart Configuration
const createPerformanceChart = () => {
    const ctx = document.getElementById('performanceChart').getContext('2d');
    return new Chart(ctx, {
        type: 'line',
        data: {
            labels: Array.from({ length: 10 }, (_, i) => `Day ${i + 1}`),
            datasets: [{
                label: 'Detection Accuracy',
                data: [95, 96, 94, 97, 95, 98, 96, 97, 98, 99],
                borderColor: '#2563eb',
                tension: 0.4,
                fill: false
            }, {
                label: 'False Positive Rate',
                data: [5, 4, 6, 3, 5, 2, 4, 3, 2, 1],
                borderColor: '#ef4444',
                tension: 0.4,
                fill: false
            }]
        },
        options: {
            responsive: true,
            interaction: {
                intersect: false,
                mode: 'index'
            },
            plugins: {
                legend: {
                    position: 'bottom'
                },
                tooltip: {
                    backgroundColor: '#1e293b',
                    titleColor: '#fff',
                    bodyColor: '#fff',
                    displayColors: false,
                    callbacks: {
                        label: function (context) {
                            let label = context.dataset.label || '';
                            if (label) {
                                label += ': ';
                            }
                            label += context.parsed.y + '%';
                            return label;
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    ticks: {
                        callback: function (value) {
                            return value + '%';
                        }
                    }
                }
            }
        }
    });
};

// Demo State Management
let demoRunning = false;
let startTime = null;
let dataPoints = 0;
let totalPackets = 0;
let totalThreats = 0;
let updateInterval;

// Simulated network traffic patterns
const generateTrafficData = () => {
    const time = (Date.now() - startTime) / 1000;
    const normalTraffic = 50 + 20 * Math.sin(time / 5) + Math.random() * 10;
    const suspiciousSpike = Math.random() > 0.9 ? 40 + Math.random() * 30 : Math.random() * 5;

    return {
        normal: normalTraffic,
        suspicious: suspiciousSpike,
        time: Math.floor(time)
    };
};

// Update chart with new data
const updateTrafficChart = (data) => {
    const chart = window.trafficChart;

    if (chart.data.labels.length > 20) {
        chart.data.labels.shift();
        chart.data.datasets[0].data.shift();
        chart.data.datasets[1].data.shift();
    }

    chart.data.labels.push(data.time + 's');
    chart.data.datasets[0].data.push(data.normal);
    chart.data.datasets[1].data.push(data.suspicious);
    chart.update('none');
};

// Update metrics display
const updateMetrics = async () => {
    try {
        let metrics;
        if (api.isConnected) {
            // Get real metrics from API
            metrics = await api.getMetrics();
        } else {
            // Simulate metrics
            const normalTraffic = Math.floor(Math.random() * 50) + 30;
            const suspiciousTraffic = Math.floor(Math.random() * 10);

            totalPackets += normalTraffic + suspiciousTraffic;
            if (Math.random() < 0.3) { // 30% chance of threat
                totalThreats++;
                showThreatAlert({
                    confidence: Math.random() * 0.5 + 0.5,
                    sourceIP: `192.168.1.${Math.floor(Math.random() * 255)}`,
                    timestamp: new Date().toISOString()
                });
            }

            metrics = {
                normalTraffic,
                suspiciousTraffic,
                totalPackets,
                totalThreats,
                avgLatency: Math.random() * 10 + 5
            };
        }

        // Update metrics display
        document.getElementById('totalPackets').textContent = metrics.totalPackets.toLocaleString();
        document.getElementById('totalThreats').textContent = metrics.totalThreats.toLocaleString();
        document.getElementById('avgLatency').textContent = metrics.avgLatency.toFixed(2);

        // Update chart
        updateChart(metrics.normalTraffic, metrics.suspiciousTraffic);

    } catch (error) {
        console.error('Error updating metrics:', error);
        if (api.isConnected) {
            // If API connection fails, switch to simulation mode
            api.disconnect();
            const statusDot = document.getElementById('statusIndicator');
            const statusText = document.getElementById('statusText');
            statusDot.style.backgroundColor = '#FFA500';
            statusText.textContent = 'Simulation Mode';
        }
    }
};

// API Configuration
const API_CONFIG = {
    baseUrl: 'http://localhost:8000',
    pollInterval: 1000, // 1 second
};

// API Connection Management
class DetectionAPI {
    constructor() {
        this.isConnected = false;
        this.pollInterval = null;
    }

    async start() {
        try {
            const response = await fetch(`${API_CONFIG.baseUrl}/start`, {
                method: 'POST'
            });
            if (response.ok) {
                this.isConnected = true;
                this.startPolling();
                return true;
            }
            return false;
        } catch (error) {
            console.error('Failed to start detection:', error);
            return false;
        }
    }

    async stop() {
        try {
            const response = await fetch(`${API_CONFIG.baseUrl}/stop`, {
                method: 'POST'
            });
            if (response.ok) {
                this.isConnected = false;
                this.stopPolling();
                return true;
            }
            return false;
        } catch (error) {
            console.error('Failed to stop detection:', error);
            return false;
        }
    }

    startPolling() {
        this.pollInterval = setInterval(async () => {
            await this.updateStats();
            await this.checkAlerts();
        }, API_CONFIG.pollInterval);
    }

    stopPolling() {
        if (this.pollInterval) {
            clearInterval(this.pollInterval);
            this.pollInterval = null;
        }
    }

    async updateStats() {
        try {
            const response = await fetch(`${API_CONFIG.baseUrl}/stats`);
            if (response.ok) {
                const stats = await response.json();
                this.updateUI(stats);
            }
        } catch (error) {
            console.error('Failed to fetch stats:', error);
        }
    }

    async checkAlerts() {
        try {
            const response = await fetch(`${API_CONFIG.baseUrl}/alerts?limit=1`);
            if (response.ok) {
                const alerts = await response.json();
                if (alerts && alerts.length > 0) {
                    showThreatAlert(alerts[0]);
                }
            }
        } catch (error) {
            console.error('Failed to fetch alerts:', error);
        }
    }

    updateUI(stats) {
        // Update packet count
        document.getElementById('packetCount').textContent =
            stats.packets_processed.toLocaleString();

        // Update threats detected
        document.getElementById('threatCount').textContent =
            stats.malicious_detected.toLocaleString();

        // Update average latency
        document.getElementById('avgLatency').textContent =
            `${stats.average_processing_time_ms.toFixed(2)}ms`;

        // Update traffic chart
        if (window.trafficChart) {
            updateTrafficChart({
                normal: stats.packets_processed - stats.malicious_detected,
                suspicious: stats.malicious_detected,
                time: Math.floor(stats.uptime_seconds)
            });
        }
    }
}

// Initialize API
const api = new DetectionAPI();

// Enhanced showThreatAlert function
const showThreatAlert = (alert) => {
    const alertDiv = document.createElement('div');
    alertDiv.className = 'threat-alert';
    alertDiv.innerHTML = `
        <i class="fas fa-exclamation-triangle"></i>
        Suspicious Activity Detected!
        <div class="alert-details">
            <span>Confidence: ${(alert.confidence * 100).toFixed(1)}%</span>
            <span>Source IPs: ${alert.source_ips.join(', ')}</span>
        </div>
        <span class="timestamp">${new Date(alert.timestamp).toLocaleTimeString()}</span>
    `;

    document.querySelector('.demo-controls').appendChild(alertDiv);

    // Remove alert after 3 seconds
    setTimeout(() => {
        alertDiv.remove();
    }, 3000);
};

// Modified startDemo function
const startDemo = async () => {
    const button = document.getElementById('startDemo');
    const statusDot = document.getElementById('statusIndicator');
    const statusText = document.getElementById('statusText');

    if (!demoRunning) {
        // Initialize or reset state
        totalPackets = 0;
        totalThreats = 0;
        dataPoints = 0;
        startTime = Date.now();

        // Initialize chart if needed
        if (!window.trafficChart) {
            createTrafficChart();
        }

        // Try to start real detection
        try {
            const started = await api.start();

            if (started) {
                // Real detection mode
                console.log('Connected to detection API');
                statusDot.style.backgroundColor = '#00C851';
                statusText.textContent = 'Connected';
                button.textContent = 'Stop Detection';
            } else {
                // Fallback to simulation mode
                console.log('Falling back to simulation mode');
                statusDot.style.backgroundColor = '#FFA500';
                statusText.textContent = 'Simulation Mode';
                button.textContent = 'Stop Demo';
                updateInterval = setInterval(updateMetrics, 1000);
            }
        } catch (error) {
            // Error fallback to simulation mode
            console.error('API connection failed:', error);
            statusDot.style.backgroundColor = '#FFA500';
            statusText.textContent = 'Simulation Mode';
            button.textContent = 'Stop Demo';
            updateInterval = setInterval(updateMetrics, 1000);
        }

        demoRunning = true;
        button.classList.add('running');

    } else {
        // Stop detection/simulation
        if (api.isConnected) {
            await api.stop();
        }

        clearInterval(updateInterval);
        demoRunning = false;
        button.textContent = 'Start Detection';
        button.classList.remove('running');
        statusDot.style.backgroundColor = '#ff4444';
        statusText.textContent = 'Disconnected';
    }
};

// Smooth scrolling for navigation links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});

// Initialize charts and demo when the page loads
document.addEventListener('DOMContentLoaded', () => {
    createPerformanceChart();

    // Initialize demo button
    const demoButton = document.getElementById('startDemo');
    if (demoButton) {
        demoButton.addEventListener('click', startDemo);
    }

    // Add scroll animations
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('visible');
            }
        });
    }, {
        threshold: 0.1
    });

    document.querySelectorAll('.feature-card, .doc-card').forEach((el) => {
        observer.observe(el);
    });
});

// Theme switcher
const toggleTheme = () => {
    const root = document.documentElement;
    const isDark = root.classList.toggle('dark-theme');

    if (isDark) {
        root.style.setProperty('--background-color', '#1a1a1a');
        root.style.setProperty('--text-color', '#ffffff');
        root.style.setProperty('--card-background', '#2d2d2d');
    } else {
        root.style.setProperty('--background-color', '#f8fafc');
        root.style.setProperty('--text-color', '#1e293b');
        root.style.setProperty('--card-background', '#ffffff');
    }

    // Update chart colors if it exists
    const chart = Chart.getChart('performanceChart');
    if (chart) {
        chart.options.scales.y.grid.color = isDark ? '#404040' : '#e5e7eb';
        chart.options.scales.x.grid.color = isDark ? '#404040' : '#e5e7eb';
        chart.update();
    }
};

// Mobile menu toggle
const toggleMobileMenu = () => {
    const navLinks = document.querySelector('.nav-links');
    navLinks.classList.toggle('show');
};

// Add updateChart function
const updateChart = (normalTraffic, suspiciousTraffic) => {
    if (!window.trafficChart) return;

    const chart = window.trafficChart;
    const timestamp = Math.floor((Date.now() - startTime) / 1000);

    // Add new data point
    chart.data.labels.push(timestamp);
    chart.data.datasets[0].data.push(normalTraffic);
    chart.data.datasets[1].data.push(suspiciousTraffic);

    // Keep last 30 seconds of data
    if (chart.data.labels.length > 30) {
        chart.data.labels.shift();
        chart.data.datasets[0].data.shift();
        chart.data.datasets[1].data.shift();
    }

    chart.update('none'); // Update without animation for smoother real-time updates
}; 