"""
Script to generate figures for the README.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def create_feature_importance_plot():
    """Create feature importance plot."""
    # Data from analysis
    features = ['time_normalized', 'ntg', 'porosity', 'kh_kv_ratio', 
                'kh', 'kv', 'owc', 'initial_pressure']
    importance = [0.00070, 0.000066, 0.000052, -0.0000003, 0, 0, 0, 0]
    std = [0.000006, 0.0000015, 0.0000013, 0.00000002, 0, 0, 0, 0]
    
    # Create figure
    plt.figure(figsize=(12, 6))
    bars = plt.barh(features, importance, xerr=std, capsize=5)
    
    # Customize
    plt.title('Feature Importance Analysis', fontsize=14, pad=20)
    plt.xlabel('Importance Score', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    
    # Color bars based on importance
    for bar in bars:
        if bar.get_width() > 0:
            bar.set_color('royalblue')
        else:
            bar.set_color('lightgray')
    
    # Add value labels
    for i, v in enumerate(importance):
        if v != 0:
            plt.text(v + 0.00001, i, f'{v:.6f}', va='center')
    
    plt.tight_layout()
    plt.savefig('figures/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_production_map():
    """Create production potential map."""
    # Create sample data
    porosity = np.linspace(0.1, 0.3, 100)
    ntg = np.linspace(0.5, 0.9, 100)
    X, Y = np.meshgrid(porosity, ntg)
    
    # Calculate production potential (example function)
    Z = 1000 * X * Y  # Simplified production potential
    
    # Create figure
    plt.figure(figsize=(10, 8))
    contour = plt.contourf(X, Y, Z, levels=20, cmap='viridis')
    plt.colorbar(contour, label='Production Potential')
    
    # Add labels and title
    plt.title('Production Potential Map', fontsize=14, pad=20)
    plt.xlabel('Porosity', fontsize=12)
    plt.ylabel('Net-to-Gross Ratio', fontsize=12)
    
    # Add annotations
    plt.annotate('Optimal Zone', xy=(0.25, 0.8), xytext=(0.15, 0.6),
                arrowprops=dict(facecolor='black', shrink=0.05))
    
    plt.tight_layout()
    plt.savefig('figures/production_map.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_business_impact_plot():
    """Create business impact analysis plot."""
    # Create sample data
    categories = ['Time Savings', 'Cost Reduction', 'Decision Support', 'Risk Management']
    proxy = [100, 80, 75, 70]  # Proxy model impact
    traditional = [0, 20, 25, 30]  # Traditional simulation impact
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Impact Comparison
    x = np.arange(len(categories))
    width = 0.35
    
    ax1.bar(x - width/2, proxy, width, label='Proxy Model', color='royalblue')
    ax1.bar(x + width/2, traditional, width, label='Traditional', color='lightgray')
    
    ax1.set_title('Impact Comparison', fontsize=14, pad=20)
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, rotation=45)
    ax1.set_ylabel('Impact Score', fontsize=12)
    ax1.legend()
    
    # Plot 2: Cost Savings
    time_savings = [1, 10, 100]  # Hours
    cost_savings = [1000, 10000, 100000]  # USD
    
    ax2.plot(time_savings, cost_savings, 'o-', color='royalblue', linewidth=2)
    ax2.set_title('Cost Savings vs Time', fontsize=14, pad=20)
    ax2.set_xlabel('Time Saved (hours)', fontsize=12)
    ax2.set_ylabel('Cost Savings (USD)', fontsize=12)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    
    # Add annotations
    for i, (x, y) in enumerate(zip(time_savings, cost_savings)):
        ax2.annotate(f'${y:,.0f}', (x, y), xytext=(5, 5),
                    textcoords='offset points')
    
    plt.tight_layout()
    plt.savefig('figures/business_impact.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Generate all figures for the README."""
    # Create figures directory
    Path('figures').mkdir(exist_ok=True)
    
    # Generate figures
    print("Generating feature importance plot...")
    create_feature_importance_plot()
    
    print("Generating production map...")
    create_production_map()
    
    print("Generating business impact plot...")
    create_business_impact_plot()
    
    print("All figures generated successfully!")

if __name__ == "__main__":
    main() 