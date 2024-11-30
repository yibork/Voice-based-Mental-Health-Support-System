import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ----------------------------- Configuration -----------------------------

# Paths
metrics_json_path = os.path.join("outputs", "tts_metrics.json")
plots_output_dir = os.path.join("plots")
os.makedirs(plots_output_dir, exist_ok=True)

# Metrics to plot
metrics_to_plot = ['inference_time_sec', 'real_time_factor', 'stoi_score', 'pesq_score', 
                  'mcd_score', 'mmsd_score']

# ----------------------------- Functions -----------------------------

def load_metrics(json_path):
    """Load metrics from JSON file into a Pandas DataFrame."""
    with open(json_path, "r") as f:
        metrics = json.load(f)
    df = pd.DataFrame(metrics)
    return df

def preprocess_data(df):
    """Handle missing values and compute average metrics."""
    # Print DataFrame columns for debugging
    print("Columns in DataFrame:", df.columns.tolist())
    
    # Drop rows with missing essential metrics
    missing_metrics = [metric for metric in metrics_to_plot if metric not in df.columns]
    if missing_metrics:
        print(f"Missing metrics in DataFrame: {missing_metrics}")
        raise KeyError(f"Missing metrics: {missing_metrics}")
    
    df_clean = df.dropna(subset=metrics_to_plot)
    
    # Compute average metrics per model
    avg_metrics = df_clean.groupby('model_name').mean(numeric_only=True).reset_index()
    
    return df_clean, avg_metrics

def plot_average_metrics(avg_metrics, metrics, output_dir):
    """Plot average metrics per TTS model as bar plots."""
    for metric in metrics:
        plt.figure(figsize=(12,6))
        sns.barplot(x='model_name', y=metric, data=avg_metrics, palette="viridis")
        plt.title(f'Average {metric.replace("_", " ").title()} per TTS Model')
        plt.xlabel('TTS Model')
        plt.ylabel(metric.replace('_', ' ').title())
        plt.xticks(rotation=45)
        plt.legend().remove()
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f"average_{metric}.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"Saved average {metric} plot at {plot_path}")

def plot_correlation_heatmap(avg_metrics, metrics, output_dir):
    """Plot heatmap of average metrics per TTS model."""
    corr = avg_metrics[metrics].corr()
    plt.figure(figsize=(10,8))
    sns.heatmap(corr, annot=True, cmap='Blues', fmt=".2f")
    plt.title('Correlation Matrix of TTS Metrics')
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "correlation_matrix.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved correlation matrix heatmap at {plot_path}")

def plot_box_plots(df, metrics, output_dir):
    """Plot distribution of metrics per TTS model as box plots."""
    for metric in metrics:
        plt.figure(figsize=(12,6))
        sns.boxplot(x='model_name', y=metric, data=df, palette="Set2")
        plt.title(f'Distribution of {metric.upper()} across TTS Models')
        plt.xlabel('TTS Model')
        plt.ylabel(metric.upper())
        plt.xticks(rotation=45)
        plt.legend().remove()
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f'boxplot_{metric}.png')
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"Saved box plot for {metric} at {plot_path}")

# ----------------------------- Main Plotting Script -----------------------------

def generate_plots(metrics_json_path, plots_output_dir, metrics_to_plot):
    # Load metrics
    df = load_metrics(metrics_json_path)
    print(f"Loaded {len(df)} records from {metrics_json_path}")
    
    # Preprocess data
    try:
        df_clean, avg_metrics = preprocess_data(df)
        print("Preprocessed data: handled missing values and computed average metrics.")
    except KeyError as e:
        print(f"Preprocessing failed: {e}")
        return
    
    # Plot average metrics
    plot_average_metrics(avg_metrics, metrics_to_plot, plots_output_dir)
    
    # Plot correlation heatmap
    plot_correlation_heatmap(avg_metrics, metrics_to_plot, plots_output_dir)
    
    # Plot box plots for distribution analysis
    plot_box_plots(df_clean, metrics_to_plot, plots_output_dir)
    
    print("All plots have been generated and saved.")

# ----------------------------- Execute Plotting -----------------------------

if __name__ == "__main__":
    generate_plots(metrics_json_path, plots_output_dir, metrics_to_plot)
