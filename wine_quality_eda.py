import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def run_eda():
    # Load the dataset
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    df = pd.read_csv(url, sep=';')

    # Display the first few rows of the dataset
    print("Dataset Head:\n", df.head())
    print("\nDataset Info:")
    print(df.info())

    # Summary statistics
    print("\nSummary statistics:\n", df.describe())

    # Mode for numeric columns
    print("\nMode for numeric columns:")
    for col in df.select_dtypes(include=np.number).columns:
        mode_val = df[col].mode()
        print(f" - {col}: {mode_val.values}")

    # Missing value analysis
    print("\nMissing values in dataset:")
    print(df.isnull().sum())

    # Set aesthetic style
    sns.set(style='whitegrid', palette='muted')

    # Visualizations
    plt.figure(figsize=(18, 12))
    
    # Alternate visualization types
    vis_types = ['hist', 'box', 'violin', 'scatter', 'bar', 'line']
    
    for i, col in enumerate(df.columns[:-1]):  # Exclude the target variable
        plt.subplot(3, 4, i + 1)

        if vis_types[i % len(vis_types)] == 'hist':
            sns.histplot(df[col], kde=True, bins=30, color='skyblue')
            plt.title(f'Distribution of {col}')
        elif vis_types[i % len(vis_types)] == 'box':
            sns.boxplot(y=df[col], color='salmon')
            plt.title(f'Boxplot of {col}')
        elif vis_types[i % len(vis_types)] == 'violin':
            sns.violinplot(y=df[col], color='lightgreen')
            plt.title(f'Violin Plot of {col}')
        elif vis_types[i % len(vis_types)] == 'scatter':
            if i > 0:  # Ensure there's a feature to pair with
                sns.scatterplot(x=df[df.columns[0]], y=df[col], alpha=0.6, color='orange')
                plt.title(f'Scatter Plot of {df.columns[0]} vs {col}')
        elif vis_types[i % len(vis_types)] == 'bar':
            sns.barplot(x=df[col].value_counts().index, y=df[col].value_counts(), palette='coolwarm')
            plt.title(f'Bar Plot of {col}')
        elif vis_types[i % len(vis_types)] == 'line':
            sns.lineplot(data=df[col].rolling(window=3).mean(), color='purple')
            plt.title(f'Line Plot of {col} (3-Point Rolling Mean)')

    plt.tight_layout()
    plt.show()

    # Correlation analysis
    corr = df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True, cbar_kws={"shrink": 0.75})
    plt.title("Correlation Heatmap of Features")
    plt.show()

    # Pairwise feature relationships
    sns.pairplot(df, hue='quality', palette='husl')
    plt.suptitle("Pairwise Feature Relationships", y=1.02)
    plt.show()

if __name__ == "__main__":
    run_eda()
