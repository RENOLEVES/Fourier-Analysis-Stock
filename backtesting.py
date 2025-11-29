# -*- coding: utf-8 -*-
"""
Created on Sat Nov 29 14:17:38 2025

@author: Administrator
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path


# Add Fourier extractor path
sys.path.insert(0, r"E:\IntelliJ IDEA 2020.2\IdeaProjects\Fourier-Analysis-Stock")


from fourier_extractor import FourierFeatureExtractor


# === Configuration ===
SOURCE_DATA_PATH = r"E:/IntelliJ IDEA 2020.2/IdeaProjects/Fourier-Analysis-Stock/data/source/cleaned_btc_2022-06_2025-06.csv"
MODEL_PATH = r"E:/IntelliJ IDEA 2020.2/IdeaProjects/Fourier-Analysis-Stock/data/model/final_model_v1.pkl"
MODEL_DATA_PATH = r"E:/IntelliJ IDEA 2020.2/IdeaProjects/Fourier-Analysis-Stock/data/model"
MODEL_VERSION = 'v1'


def load_and_prepare_data_with_features(source_data_path):
    print("\nLoading original data...")
    data = pd.read_csv(source_data_path)
    data.reset_index(drop=True, inplace=True)
    print(f"Data loaded: {data.shape[0]} rows × {data.shape[1]} columns")
    print(f"Original columns: {list(data.columns)}")
    
    # === Generate Fourier Features ===
    print("\n[1/2] Generating Fourier features...")
    try:
        extractor = FourierFeatureExtractor(window_size=128, min_period=2, max_period=64)
        fourier_features = extractor.extract_features(data['Close'], output_all=True)
        data = pd.concat([data, fourier_features], axis=1)
        print(f"Fourier features added:")
        print(f"   - fourier_period")
        print(f"   - fourier_period_strength")
        print(f"   - fourier_period_bins")
    except Exception as e:
        print(f"Error generating Fourier features: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    print(f"\nFinal data shape: {data.shape}")
    print(f"All columns available:")
    for col in data.columns:
        print(f"   - {col}")
    
    return data

def load_model(model_path):
    """Load the trained RandomForest model."""
    try:
        model = joblib.load(model_path)
        print(f"Model loaded from: {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def load_best_features(model_data_path, version):
    """Load the best features list from training."""
    try:
        features_path = os.path.join(model_data_path, f"model_best_features_{version}.csv")
        best_features_df = pd.read_csv(features_path, index_col=0)
        best_cols = best_features_df['Name'].tolist()
        print(f"Loaded best features ({len(best_cols)} features):")
        for col in best_cols:
            print(f"   - {col}")
        return best_cols
    except Exception as e:
        print(f"Error loading best features: {e}")
        return None

def generate_backtest_from_model(model, data, best_cols):
    """Generate backtest results using the loaded model."""
    print("\nGenerating predictions...")
    
    # Check which columns are missing
    missing_cols = [col for col in best_cols if col not in data.columns]
    available_cols = [col for col in best_cols if col in data.columns]
    
    if missing_cols:
        print(f"Missing columns: {missing_cols}")
        print(f"Cannot proceed with missing features")
        return None
    
    print(f"All {len(best_cols)} required columns available")
    
    # Prepare input data
    X = data[available_cols].fillna(data[available_cols].mean())
    print(f"Prepared input data: {X.shape}")
    
    # Handle NaN values
    X = X.fillna(0)
    
    # Make predictions
    predictions = model.predict(X)
    print(f"Generated {len(predictions)} predictions")
    
    # Use score column as actual
    if 'score' in data.columns:
        actual_values = data['score'].values
        print(f"Using 'score' column as actual values")
    else:
        print("'score' column not found")
        actual_values = np.zeros(len(predictions))
    
    # Create backtest dataframe
    backtest_results = pd.DataFrame({
        'index': np.arange(len(data)),
        'Perdiction': predictions,
        'score': actual_values,
    })
    
    # Add price data
    if 'Close' in data.columns:
        backtest_results['Close'] = data['Close'].values
        print(f"Added price data")
    
    # Calculate performance metrics
    backtest_results['daily_return'] = backtest_results['score']
    backtest_results['portfolio_value'] = (1 + backtest_results['daily_return']).cumprod()
    backtest_results['error'] = backtest_results['Perdiction'] - backtest_results['score']
    backtest_results['abs_error'] = np.abs(backtest_results['error'])
    
    print(f"Generated backtest results")
    
    return backtest_results

def save_backtest_results(backtest_results, filepath):
    """Save backtest results."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(backtest_results, filepath)
    print(f"Backtest results saved to: {filepath}")

def plot_backtest_comprehensive(backtest_results, save_dir=MODEL_DATA_PATH, version=MODEL_VERSION):
    """Generate comprehensive backtest visualizations."""
    os.makedirs(save_dir, exist_ok=True)
    
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    colors = {
        'portfolio': '#2E86AB',
        'scatter': '#A23B72',
        'histogram': '#F18F01',
        'error': '#06A77D',
        'return': '#D62246'
    }
    
    # === Plot 1: Portfolio Value ---
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(backtest_results.index, backtest_results['portfolio_value'], 
            linewidth=2.5, color=colors['portfolio'], label='Portfolio Value')
    ax1.fill_between(backtest_results.index, backtest_results['portfolio_value'], 
                     alpha=0.3, color=colors['portfolio'])
    ax1.set_title('Portfolio Value Over Time', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Portfolio Value', fontsize=11, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # === Plot 2: Predictions vs Actual ---
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.scatter(backtest_results['score'], backtest_results['Perdiction'], 
               alpha=0.5, s=30, color=colors['scatter'])
    min_val = min(backtest_results['score'].min(), backtest_results['Perdiction'].min())
    max_val = max(backtest_results['score'].max(), backtest_results['Perdiction'].max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    ax2.set_xlabel('Actual Values', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Predictions', fontsize=11, fontweight='bold')
    ax2.set_title('Predictions vs Actual', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # === Plot 3: Residuals Distribution ---
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.hist(backtest_results['error'], bins=50, color=colors['histogram'], alpha=0.7, edgecolor='black')
    ax3.axvline(x=backtest_results['error'].mean(), color='red', linestyle='--', 
               linewidth=2, label=f'Mean: {backtest_results["error"].mean():.4f}')
    ax3.set_xlabel('Prediction Error', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax3.set_title('Distribution of Prediction Errors', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # === Plot 4: Errors Over Time ---
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.scatter(backtest_results.index, backtest_results['error'], 
               alpha=0.5, s=20, color=colors['error'])
    ax4.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax4.set_xlabel('Backtest Index', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Error (Actual - Predicted)', fontsize=11, fontweight='bold')
    ax4.set_title('Prediction Errors Over Time', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # === Plot 5: Cumulative Returns ---
    ax5 = fig.add_subplot(gs[2, 1])
    cumulative_returns = (backtest_results['portfolio_value'] - 1) * 100
    ax5.plot(backtest_results.index, cumulative_returns, linewidth=2.5, color=colors['return'])
    ax5.fill_between(backtest_results.index, cumulative_returns, alpha=0.3, color=colors['return'])
    ax5.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax5.set_xlabel('Backtest Index', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Cumulative Return (%)', fontsize=11, fontweight='bold')
    ax5.set_title('Cumulative Strategy Returns', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    plt.suptitle(f'Backtesting Results - Model {version}', 
                fontsize=16, fontweight='bold', y=0.995)
    
    plot_path = os.path.join(save_dir, f'backtest_analysis_{version}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {plot_path}")
    plt.show()

def print_backtest_metrics(backtest_results):
    """Print summary statistics."""
    print("\n" + "="*60)
    print("BACKTESTING METRICS")
    print("="*60)
    
    final_value = backtest_results['portfolio_value'].iloc[-1]
    total_return = (final_value - 1) * 100
    
    print(f"Initial Portfolio Value: $1.00")
    print(f"Final Portfolio Value: ${final_value:.4f}")
    print(f"Total Return: {total_return:.2f}%")
    print(f"\nPrediction Metrics:")
    print(f"  Mean Error: {backtest_results['error'].mean():.6f}")
    print(f"  Std Dev Error: {backtest_results['error'].std():.6f}")
    print(f"  Mean Absolute Error: {backtest_results['abs_error'].mean():.6f}")
    print(f"  Max Error: {backtest_results['error'].max():.6f}")
    print(f"  Min Error: {backtest_results['error'].min():.6f}")
    
    if 'Close' in backtest_results.columns:
        print(f"\nPrice Statistics:")
        print(f"  Mean Price: ${backtest_results['Close'].mean():.2f}")
        print(f"  Min Price: ${backtest_results['Close'].min():.2f}")
        print(f"  Max Price: ${backtest_results['Close'].max():.2f}")
    
    print("="*60 + "\n")

# === Main execution ===
if __name__ == "__main__":
    print("\n" + "="*70)
    print("REGENERATE ALL FEATURES & BACKTEST")
    print("="*70)
    
    try:
        # Step 1: Load and prepare data with all features
        print("\n[STEP 1/4] Loading Data & Generating Features...")
        data = load_and_prepare_data_with_features(SOURCE_DATA_PATH)
        
        if data is None:
            raise Exception("Failed to prepare data")
        
        # Step 2: Load model
        print("\n[STEP 2/4] Loading Model...")
        model = load_model(MODEL_PATH)
        
        if model is None:
            raise Exception("Failed to load model")
        
        # Step 3: Load best features
        print("\n[STEP 3/4] Loading Best Features Configuration...")
        best_cols = load_best_features(MODEL_DATA_PATH, MODEL_VERSION)
        
        if best_cols is None:
            raise Exception("Failed to load best features")
        
        # Step 4: Generate backtest
        print("\n[STEP 4a/5] Generating Backtest Results...")
        backtest_results = generate_backtest_from_model(model, data, best_cols)
        
        if backtest_results is None:
            raise Exception("Failed to generate backtest results")
        
        # Step 5a: Save results
        print("\n[STEP 4b/5] Saving Results...")
        backtest_path = os.path.join(MODEL_DATA_PATH, f"backtest_result_{MODEL_VERSION}.pkl")
        save_backtest_results(backtest_results, backtest_path)
        
        # Step 5b: Plot
        print("\n[STEP 5/5] Creating Visualizations...")
        plot_backtest_comprehensive(backtest_results, save_dir=MODEL_DATA_PATH, version=MODEL_VERSION)
        
        # Print metrics
        print_backtest_metrics(backtest_results)
        
        print("ALL COMPLETE!")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()