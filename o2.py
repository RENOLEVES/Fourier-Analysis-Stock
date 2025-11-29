# -*- coding: utf-8 -*-
"""
Created on Sat Nov 29 14:10:30 2025

@author: Administrator
"""

import joblib
from backtest_plot_visualizer import BacktestPlotter

# Load your backtesting results
backtest_results = joblib.load("E:/IntelliJ IDEA 2020.2/IdeaProjects/Fourier-Analysis-Stock/data/model/final_model_v1.pkl")

# Create plotter
plotter = BacktestPlotter()

# Generate visualizations
plotter.plot_backtest_results(
    backtest_results,
    prediction_col='Perdiction',
    actual_col='score',
    price_col='Close',
    save_path='backtest_results.png'
)