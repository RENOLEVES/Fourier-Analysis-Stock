import numpy as np
import pandas as pd
from scipy import signal
from scipy.fft import fft, fftfreq
from scipy.signal.windows import hann
import matplotlib.pyplot as plt


class FourierFeatureExtractor:
    
    def __init__(self, window_size=256, min_period=2, max_period=None):
        self.window_size = window_size
        self.min_period = min_period
        self.max_period = max_period if max_period else window_size // 2
        
    def compute_fft(self, price_series):
        
        if len(price_series) < self.window_size:
            return {'dominant_period': np.nan, 'dominant_power': 0.0, 
                   'frequency': np.nan, 'period_strength': 0.0}
        
        # Use only the last window_size points
        window = price_series[-self.window_size:]
        
        # Detrend and normalize
        detrended = signal.detrend(window)
        normalized = (detrended - np.mean(detrended)) / (np.std(detrended) + 1e-8)
        
        # Apply Hann window to reduce spectral leakage
        windowed = normalized * signal.hann(len(normalized))
        
        # Compute FFT
        fft_vals = fft(windowed)
        freqs = fftfreq(len(windowed))
        
        # Power spectrum (one-sided)
        power = np.abs(fft_vals) ** 2
        positive_freqs = freqs[:len(freqs)//2]
        positive_power = power[:len(power)//2]
        
        # Convert frequency to period
        periods = 1 / (positive_freqs + 1e-8)
        
        # Filter by min/max period constraints
        valid_mask = (periods >= self.min_period) & (periods <= self.max_period)
        valid_periods = periods[valid_mask]
        valid_power = positive_power[valid_mask]
        
        if len(valid_power) == 0:
            return {'dominant_period': np.nan, 'dominant_power': 0.0,
                   'frequency': np.nan, 'period_strength': 0.0}
        
        # Find dominant period
        dominant_idx = np.argmax(valid_power)
        dominant_period = valid_periods[dominant_idx]
        dominant_power = valid_power[dominant_idx]
        
        # Normalize power by total power (period strength: 0-1)
        period_strength = dominant_power / (np.sum(valid_power) + 1e-8)
        
        return {
            'dominant_period': dominant_period,
            'dominant_power': dominant_power,
            'frequency': 1 / dominant_period if dominant_period > 0 else np.nan,
            'period_strength': period_strength
        }
    
    def extract_features(self, price_data, output_all=False):

        if isinstance(price_data, pd.Series):
            prices = price_data.values
        else:
            prices = np.array(price_data)
        
        n = len(prices)
        dominant_periods = np.full(n, np.nan)
        period_strengths = np.full(n, np.nan)
        
        # Need at least window_size points to compute FFT
        for i in range(self.window_size - 1, n):
            fft_result = self.compute_fft(prices[:i+1])
            dominant_periods[i] = fft_result['dominant_period']
            period_strengths[i] = fft_result['period_strength']
        
        features = pd.DataFrame({
            'fourier_period': dominant_periods,
            'fourier_period_strength': period_strengths
        }, index=price_data.index if isinstance(price_data, pd.Series) else None)
        
        if output_all:
            # Add additional metrics
            features['fourier_period_bins'] = pd.cut(
                features['fourier_period'], 
                bins=[0, 5, 10, 20, 50, 100, np.inf],
                labels=['very_short', 'short', 'medium', 'long', 'very_long', 'extreme']
            )
        
        return features
    
    def plot_fft(self, price_series, save_path=None):

        window = np.array(price_series[-self.window_size:])
        detrended = signal.detrend(window)
        normalized = (detrended - np.mean(detrended)) / (np.std(detrended) + 1e-8)
        windowed = normalized * signal.hann(len(normalized))
        
        fft_vals = fft(windowed)
        freqs = fftfreq(len(windowed))
        power = np.abs(fft_vals) ** 2
        positive_freqs = freqs[:len(freqs)//2]
        positive_power = power[:len(power)//2]
        periods = 1 / (positive_freqs + 1e-8)
        
        valid_mask = (periods >= self.min_period) & (periods <= self.max_period)
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Time domain
        axes[0].plot(window)
        axes[0].set_title('Price Series (Last {} Points)'.format(self.window_size))
        axes[0].set_ylabel('Price')
        axes[0].grid()
        
        # Frequency domain
        axes[1].semilogy(periods[valid_mask], positive_power[valid_mask], linewidth=1.5)
        axes[1].set_xlabel('Period (bars)')
        axes[1].set_ylabel('Power')
        axes[1].set_title('Power Spectrum (Valid Period Range: {}-{})'.format(
            self.min_period, self.max_period))
        axes[1].grid()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100)
        plt.show()


# === Usage Example ===
if __name__ == "__main__":
    # Load your BTC data
    btc_data = pd.read_csv(r"E:/IntelliJ IDEA 2020.2/IdeaProjects/Fourier-Analysis-Stock/data/source/cleaned_btc_2022-06_2025-06.csv")
    
    # Initialize extractor
    extractor = FourierFeatureExtractor(
        window_size=128,      # 128 bars for analysis
        min_period=2,         # Minimum 2-bar period
        max_period=64         # Maximum 64-bar period
    )
    
    # Extract features
    fourier_features = extractor.extract_features(
        btc_data['Close'], 
        output_all=True
    )
    
    # Add to original data
    btc_data_enhanced = pd.concat([btc_data, fourier_features], axis=1)
    
    # Visualize
    extractor.plot_fft(btc_data['Close'].values, save_path='fourier_analysis.png')
    
    # Use in calibration.py
    print(btc_data_enhanced.head())