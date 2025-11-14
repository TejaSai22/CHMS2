"""
CSCE5380 Data Mining - Group 15
PHASE 3: PATTERN DISCOVERY & ANOMALY DETECTION (Weeks 5-6)
Crop Health Monitoring from Remote Sensing

Owner: Teja Sai Srinivas Kunisetty
Goal: DTW-based clustering and anomaly detection

This phase uses DTW (Dynamic Time Warping) for clustering time-series with
temporal misalignment and identifies anomalous growth patterns.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)


class PatternDiscoveryEngine:
    """
    Discover crop growth patterns using DTW clustering and detect anomalies
    """
    
    def __init__(self, features_dir="./outputs/phase2/features",
                 output_dir="./outputs/phase3"):
        """
        Initialize the pattern discovery engine
        
        Args:
            features_dir: Directory with Phase 2 features
            output_dir: Directory for Phase 3 outputs
        """
        self.features_dir = Path(features_dir)
        self.output_dir = Path(output_dir)
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'clusters').mkdir(exist_ok=True)
        (self.output_dir / 'anomalies').mkdir(exist_ok=True)
        (self.output_dir / 'visualizations').mkdir(exist_ok=True)
        (self.output_dir / 'patterns').mkdir(exist_ok=True)
        (self.output_dir / 'reports').mkdir(exist_ok=True)
        
        # Data containers
        self.df_temporal = None
        self.df_aggregated = None
        self.df_spatial = None
        self.timeseries_data = None
        self.cluster_model = None
        self.cluster_labels = None
        self.anomaly_scores = None
        
        print("="*80)
        print("PHASE 3: PATTERN DISCOVERY & ANOMALY DETECTION")
        print("="*80)
        print(f"\n✅ Engine initialized")
        print(f"   Features from Phase 2: {self.features_dir}")
        print(f"   Output directory: {self.output_dir}\n")
    
    def load_phase2_data(self):
        """Load feature data from Phase 2"""
        print("\n" + "="*80)
        print("STEP 1: LOADING PHASE 2 DATA")
        print("="*80 + "\n")
        
        print("📥 Loading Phase 2 feature files...\n")
        
        # Load temporal features
        temporal_file = self.features_dir / 'temporal_features.csv'
        if not temporal_file.exists():
            raise FileNotFoundError(f"Temporal features not found at {temporal_file}")
        self.df_temporal = pd.read_csv(temporal_file)
        print(f"   ✅ Temporal features: {len(self.df_temporal):,} rows, {self.df_temporal['Parcel_ID'].nunique()} parcels")
        
        # Load aggregated features
        aggregated_file = self.features_dir / 'aggregated_features.csv'
        if aggregated_file.exists():
            self.df_aggregated = pd.read_csv(aggregated_file)
            print(f"   ✅ Aggregated features: {len(self.df_aggregated):,} parcels")
        
        # Load spatial features
        spatial_file = self.features_dir / 'spatial_features.csv'
        if spatial_file.exists():
            self.df_spatial = pd.read_csv(spatial_file)
            print(f"   ✅ Spatial features: {len(self.df_spatial):,} parcels")
        
        print(f"\n✅ Phase 2 data loaded successfully\n")
    
    def prepare_timeseries_data(self, feature='Mean_NDVI', max_parcels=None):
        """
        Prepare time-series data for DTW clustering
        
        Args:
            feature: Which feature to use for time-series ('Mean_NDVI' or 'Mean_EVI')
            max_parcels: Maximum number of parcels to process (None = all)
        """
        print("\n" + "="*80)
        print("STEP 2: PREPARING TIME-SERIES DATA FOR DTW")
        print("="*80 + "\n")
        
        print(f"📊 Preparing {feature} time-series for clustering...\n")
        print("   DTW (Dynamic Time Warping) allows clustering of time-series")
        print("   that have similar patterns but are temporally misaligned.")
        print("   This is crucial for crops with different planting dates!\n")
        
        # Pivot temporal features to create time-series format
        # From: Parcel_ID | Timestep | Feature_Value
        # To: Parcel_ID as rows, Timesteps as columns
        
        print(f"   Pivoting data: {feature} × Timestep...\n")
        
        pivot_data = self.df_temporal.pivot(
            index='Parcel_ID',
            columns='Timestep',
            values=feature
        )
        
        # Handle any missing values
        pivot_data = pivot_data.fillna(pivot_data.mean())
        
        if max_parcels:
            pivot_data = pivot_data.head(max_parcels)
        
        # Convert to numpy array: shape (n_parcels, n_timesteps)
        timeseries_array = pivot_data.values
        
        # Reshape for tslearn: shape (n_parcels, n_timesteps, 1)
        timeseries_array = timeseries_array.reshape((timeseries_array.shape[0], timeseries_array.shape[1], 1))
        
        # Normalize time-series
        print("   Normalizing time-series (mean=0, std=1)...")
        scaler = TimeSeriesScalerMeanVariance()
        timeseries_normalized = scaler.fit_transform(timeseries_array)
        
        self.timeseries_data = {
            'data': timeseries_normalized,
            'parcel_ids': pivot_data.index.tolist(),
            'feature': feature,
            'n_parcels': timeseries_normalized.shape[0],
            'n_timesteps': timeseries_normalized.shape[1]
        }
        
        print(f"\n✅ Time-series data prepared")
        print(f"   Shape: {timeseries_normalized.shape}")
        print(f"   Parcels: {self.timeseries_data['n_parcels']:,}")
        print(f"   Timesteps: {self.timeseries_data['n_timesteps']}")
        print(f"   Feature: {feature}\n")
        
        return self.timeseries_data
    
    def perform_dtw_clustering(self, n_clusters=5, random_state=42):
        """
        Perform K-Means clustering with DTW metric
        
        This is THE key innovation from your prompt: using DTW to cluster
        time-series that may have similar patterns but different timing.
        
        Args:
            n_clusters: Number of clusters (growth patterns) to identify
            random_state: Random seed for reproducibility
        """
        print("\n" + "="*80)
        print("STEP 3: DTW-BASED K-MEANS CLUSTERING")
        print("="*80 + "\n")
        
        print(f"🎯 Clustering {self.timeseries_data['n_parcels']:,} parcels into {n_clusters} growth patterns...\n")
        print("   Using TimeSeriesKMeans with DTW (Dynamic Time Warping) metric")
        print("   This handles temporal misalignment in crop growth patterns\n")
        
        # Initialize DTW-based K-Means
        print("   Initializing model...")
        self.cluster_model = TimeSeriesKMeans(
            n_clusters=n_clusters,
            metric="dtw",  # KEY: Use DTW distance instead of Euclidean
            max_iter=10,   # Increase if needed (DTW is computationally expensive)
            random_state=random_state,
            n_jobs=-1,     # Use all CPU cores
            verbose=1
        )
        
        # Fit and predict
        print(f"\n   Fitting model (this may take a few minutes for {self.timeseries_data['n_parcels']:,} parcels)...")
        self.cluster_labels = self.cluster_model.fit_predict(self.timeseries_data['data'])
        
        # Get cluster statistics
        unique, counts = np.unique(self.cluster_labels, return_counts=True)
        
        print(f"\n✅ DTW clustering complete")
        print(f"\n   Cluster Distribution:")
        for cluster_id, count in zip(unique, counts):
            percentage = (count / len(self.cluster_labels)) * 100
            print(f"   Cluster {cluster_id}: {count:5,} parcels ({percentage:5.2f}%)")
        
        # Create cluster assignments DataFrame
        cluster_df = pd.DataFrame({
            'Parcel_ID': self.timeseries_data['parcel_ids'],
            'Cluster': self.cluster_labels
        })
        
        # Merge with aggregated features
        if self.df_aggregated is not None:
            cluster_df = cluster_df.merge(self.df_aggregated, on='Parcel_ID', how='left')
        
        # Save cluster assignments
        output_file = self.output_dir / 'clusters' / 'cluster_assignments.csv'
        cluster_df.to_csv(output_file, index=False)
        print(f"\n   💾 Saved cluster assignments to: {output_file}\n")
        
        return cluster_df
    
    def detect_anomalies(self, contamination=0.05):
        """
        Detect anomalous growth patterns using Isolation Forest
        
        Anomalies could indicate:
        - Crop stress (disease, pests, drought)
        - Measurement errors
        - Unusual environmental conditions
        
        Args:
            contamination: Expected proportion of anomalies (default: 5%)
        """
        print("\n" + "="*80)
        print("STEP 4: ANOMALY DETECTION")
        print("="*80 + "\n")
        
        print(f"🔍 Detecting anomalous growth patterns (contamination={contamination})...\n")
        print("   Using Isolation Forest algorithm")
        print("   Anomalies may indicate crop stress or unusual conditions\n")
        
        # Use aggregated features for anomaly detection
        if self.df_aggregated is None:
            print("   ⚠️  Aggregated features not available, skipping anomaly detection")
            return None
        
        # ========================================
        # FEATURE ENGINEERING: First Derivative
        # ========================================
        print("   📊 Computing first derivative (rate of change) of NDVI time-series...")
        print("      Sharp negative values indicate rapid crop stress\n")
        
        # Calculate first derivative for each parcel
        derivative_features = []
        
        for parcel_id in self.df_temporal['Parcel_ID'].unique():
            parcel_data = self.df_temporal[self.df_temporal['Parcel_ID'] == parcel_id].sort_values('Timestep')
            
            if len(parcel_data) > 1:
                # Calculate first derivative (rate of change)
                ndvi_values = parcel_data['Mean_NDVI'].values
                ndvi_derivative = np.diff(ndvi_values)  # First derivative
                
                # Compute statistics on the derivative
                derivative_features.append({
                    'Parcel_ID': parcel_id,
                    'NDVI_Derivative_Mean': np.mean(ndvi_derivative),
                    'NDVI_Derivative_Std': np.std(ndvi_derivative),
                    'NDVI_Derivative_Min': np.min(ndvi_derivative),  # Most negative drop
                    'NDVI_Derivative_Max': np.max(ndvi_derivative),  # Largest increase
                    'NDVI_Negative_Slope_Count': np.sum(ndvi_derivative < 0),  # # of declining periods
                    'NDVI_Sharp_Drop_Count': np.sum(ndvi_derivative < -0.1)  # Severe drops
                })
        
        df_derivatives = pd.DataFrame(derivative_features)
        
        # Merge derivative features with aggregated features
        self.df_aggregated = self.df_aggregated.merge(df_derivatives, on='Parcel_ID', how='left')
        
        print(f"   ✅ Added {len(df_derivatives.columns)-1} derivative-based features\n")
        
        # Select numeric features for anomaly detection (now includes derivatives!)
        feature_cols = [col for col in self.df_aggregated.columns 
                       if col not in ['Parcel_ID', 'Patch_ID', 'Crop_Label']]
        X = self.df_aggregated[feature_cols].fillna(0)
        
        # Standardize features
        print(f"   Standardizing {len(feature_cols)} features...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Fit Isolation Forest
        print(f"   Fitting Isolation Forest...")
        iso_forest = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_jobs=-1
        )
        
        # -1 for anomalies, 1 for normal
        predictions = iso_forest.fit_predict(X_scaled)
        
        # Get anomaly scores (lower = more anomalous)
        self.anomaly_scores = iso_forest.score_samples(X_scaled)
        
        # Create anomaly DataFrame
        anomaly_df = pd.DataFrame({
            'Parcel_ID': self.df_aggregated['Parcel_ID'],
            'Is_Anomaly': predictions == -1,
            'Anomaly_Score': self.anomaly_scores
        })
        
        # Merge with aggregated features
        anomaly_df = anomaly_df.merge(self.df_aggregated, on='Parcel_ID', how='left')
        
        n_anomalies = np.sum(predictions == -1)
        anomaly_percentage = (n_anomalies / len(predictions)) * 100
        
        print(f"\n✅ Anomaly detection complete")
        print(f"   Total parcels analyzed: {len(predictions):,}")
        print(f"   Anomalies detected: {n_anomalies:,} ({anomaly_percentage:.2f}%)")
        print(f"   Normal parcels: {np.sum(predictions == 1):,}")
        
        # Save anomaly results
        output_file = self.output_dir / 'anomalies' / 'anomaly_scores.csv'
        anomaly_df.to_csv(output_file, index=False)
        print(f"\n   💾 Saved anomaly scores to: {output_file}\n")
        
        # Save top anomalies
        top_anomalies = anomaly_df[anomaly_df['Is_Anomaly']].sort_values('Anomaly_Score').head(20)
        top_file = self.output_dir / 'anomalies' / 'top_anomalies.csv'
        top_anomalies.to_csv(top_file, index=False)
        print(f"   💾 Saved top 20 anomalies to: {top_file}\n")
        
        return anomaly_df
    
    def visualize_clusters(self, cluster_df, n_samples_per_cluster=3):
        """
        Visualize cluster patterns and characteristics
        
        Args:
            cluster_df: DataFrame with cluster assignments
            n_samples_per_cluster: Number of sample time-series to plot per cluster
        """
        print("\n" + "="*80)
        print("STEP 5: VISUALIZING CLUSTER PATTERNS")
        print("="*80 + "\n")
        
        print("📊 Creating cluster visualizations...\n")
        
        n_clusters = len(np.unique(self.cluster_labels))
        
        # 1. Plot cluster centroids
        print("   1. Plotting cluster centroids...")
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for cluster_id in range(n_clusters):
            ax = axes[cluster_id]
            
            # Plot centroid
            centroid = self.cluster_model.cluster_centers_[cluster_id].ravel()
            timesteps = np.arange(len(centroid))
            
            ax.plot(timesteps, centroid, linewidth=3, color='red', 
                   label=f'Centroid (Cluster {cluster_id})', zorder=10)
            
            # Plot sample members
            cluster_mask = self.cluster_labels == cluster_id
            cluster_indices = np.where(cluster_mask)[0]
            
            # Sample a few members
            sample_size = min(n_samples_per_cluster, len(cluster_indices))
            sample_indices = np.random.choice(cluster_indices, sample_size, replace=False)
            
            for idx in sample_indices:
                ts = self.timeseries_data['data'][idx].ravel()
                ax.plot(timesteps, ts, alpha=0.3, linewidth=1, color='blue')
            
            ax.set_xlabel('Timestep', fontsize=11)
            ax.set_ylabel(f'{self.timeseries_data["feature"]} (normalized)', fontsize=11)
            ax.set_title(f'Cluster {cluster_id} ({np.sum(cluster_mask)} parcels)', 
                        fontsize=12, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
        
        # Hide extra subplot
        if n_clusters < len(axes):
            axes[n_clusters].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'visualizations' / 'cluster_centroids.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Cluster size distribution
        print("   2. Creating cluster distribution plot...")
        fig, ax = plt.subplots(figsize=(12, 7))
        
        unique, counts = np.unique(self.cluster_labels, return_counts=True)
        colors = plt.cm.Set3(np.linspace(0, 1, n_clusters))
        
        bars = ax.bar(unique, counts, color=colors, edgecolor='black', linewidth=2)
        ax.set_xlabel('Cluster ID', fontsize=13, fontweight='bold')
        ax.set_ylabel('Number of Parcels', fontsize=13, fontweight='bold')
        ax.set_title('Distribution of Parcels Across Clusters', fontsize=15, fontweight='bold')
        ax.set_xticks(unique)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add count labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height):,}',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'visualizations' / 'cluster_distribution.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Cluster characteristics (if aggregated features available)
        if 'NDVI_Mean' in cluster_df.columns:
            print("   3. Analyzing cluster characteristics...")
            
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # NDVI Mean
            cluster_df.boxplot(column='NDVI_Mean', by='Cluster', ax=axes[0, 0])
            axes[0, 0].set_title('NDVI Mean by Cluster', fontsize=13, fontweight='bold')
            axes[0, 0].set_xlabel('Cluster', fontsize=11)
            axes[0, 0].set_ylabel('NDVI Mean', fontsize=11)
            
            # NDVI Peak
            cluster_df.boxplot(column='NDVI_Peak_Value', by='Cluster', ax=axes[0, 1])
            axes[0, 1].set_title('NDVI Peak by Cluster', fontsize=13, fontweight='bold')
            axes[0, 1].set_xlabel('Cluster', fontsize=11)
            axes[0, 1].set_ylabel('NDVI Peak', fontsize=11)
            
            # NDVI Slope
            cluster_df.boxplot(column='NDVI_Slope', by='Cluster', ax=axes[1, 0])
            axes[1, 0].set_title('NDVI Slope by Cluster', fontsize=13, fontweight='bold')
            axes[1, 0].set_xlabel('Cluster', fontsize=11)
            axes[1, 0].set_ylabel('NDVI Slope', fontsize=11)
            
            # NDVI Range
            cluster_df.boxplot(column='NDVI_Range', by='Cluster', ax=axes[1, 1])
            axes[1, 1].set_title('NDVI Range by Cluster', fontsize=13, fontweight='bold')
            axes[1, 1].set_xlabel('Cluster', fontsize=11)
            axes[1, 1].set_ylabel('NDVI Range', fontsize=11)
            
            plt.suptitle('')  # Remove default title
            plt.tight_layout()
            plt.savefig(self.output_dir / 'visualizations' / 'cluster_characteristics.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"\n✅ Cluster visualizations created")
        print(f"   Saved to: {self.output_dir / 'visualizations'}\n")
    
    def visualize_anomalies(self, anomaly_df):
        """
        Visualize anomaly detection results
        
        Args:
            anomaly_df: DataFrame with anomaly scores
        """
        print("\n" + "="*80)
        print("STEP 6: VISUALIZING ANOMALIES")
        print("="*80 + "\n")
        
        print("📊 Creating anomaly visualizations...\n")
        
        # 1. Anomaly score distribution
        print("   1. Plotting anomaly score distribution...")
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Histogram
        axes[0].hist(anomaly_df['Anomaly_Score'], bins=50, 
                    color='skyblue', edgecolor='black', alpha=0.7)
        axes[0].axvline(anomaly_df[anomaly_df['Is_Anomaly']]['Anomaly_Score'].max(), 
                       color='red', linestyle='--', linewidth=2, 
                       label='Anomaly Threshold')
        axes[0].set_xlabel('Anomaly Score', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title('Distribution of Anomaly Scores', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.3)
        
        # Scatter: NDVI vs Anomaly Score
        if 'NDVI_Mean' in anomaly_df.columns:
            normal = anomaly_df[~anomaly_df['Is_Anomaly']]
            anomalies = anomaly_df[anomaly_df['Is_Anomaly']]
            
            axes[1].scatter(normal['NDVI_Mean'], normal['Anomaly_Score'], 
                          alpha=0.5, s=20, c='blue', label='Normal')
            axes[1].scatter(anomalies['NDVI_Mean'], anomalies['Anomaly_Score'], 
                          alpha=0.8, s=50, c='red', marker='X', label='Anomaly')
            axes[1].set_xlabel('NDVI Mean', fontsize=12)
            axes[1].set_ylabel('Anomaly Score', fontsize=12)
            axes[1].set_title('Anomaly Scores vs NDVI Mean', fontsize=14, fontweight='bold')
            axes[1].legend(fontsize=11)
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'visualizations' / 'anomaly_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Top anomalies time-series
        print("   2. Plotting top anomalous time-series...")
        top_anomalies = anomaly_df[anomaly_df['Is_Anomaly']].sort_values('Anomaly_Score').head(6)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        
        for idx, (_, row) in enumerate(top_anomalies.iterrows()):
            if idx >= 6:
                break
            
            ax = axes[idx]
            parcel_id = row['Parcel_ID']
            
            # Get time-series for this parcel
            parcel_temporal = self.df_temporal[self.df_temporal['Parcel_ID'] == parcel_id].sort_values('Timestep')
            
            ax.plot(parcel_temporal['Timestep'], parcel_temporal['Mean_NDVI'], 
                   marker='o', linewidth=2, color='red', label='NDVI')
            ax.plot(parcel_temporal['Timestep'], parcel_temporal['Mean_EVI'], 
                   marker='s', linewidth=2, color='orange', label='EVI', alpha=0.7)
            
            ax.set_xlabel('Timestep', fontsize=10)
            ax.set_ylabel('Index Value', fontsize=10)
            ax.set_title(f'Anomaly: {parcel_id}\nScore: {row["Anomaly_Score"]:.4f}', 
                        fontsize=11, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'visualizations' / 'top_anomalies_timeseries.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n✅ Anomaly visualizations created")
        print(f"   Saved to: {self.output_dir / 'visualizations'}\n")
    
    def generate_report(self, cluster_df, anomaly_df):
        """
        Generate comprehensive Phase 3 report
        
        Args:
            cluster_df: DataFrame with cluster assignments
            anomaly_df: DataFrame with anomaly scores
        """
        print("\n" + "="*80)
        print("STEP 7: GENERATING PHASE 3 REPORT")
        print("="*80 + "\n")
        
        report_lines = []
        report_lines.append("="*80)
        report_lines.append("PHASE 3: PATTERN DISCOVERY & ANOMALY DETECTION")
        report_lines.append("="*80)
        report_lines.append(f"\nReport Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Clustering summary
        report_lines.append("="*80)
        report_lines.append("DTW-BASED CLUSTERING RESULTS")
        report_lines.append("="*80)
        report_lines.append(f"\nClustering Algorithm: K-Means with DTW (Dynamic Time Warping) metric")
        report_lines.append(f"Number of Clusters: {len(np.unique(self.cluster_labels))}")
        report_lines.append(f"Total Parcels Clustered: {len(self.cluster_labels):,}")
        report_lines.append(f"Feature Used: {self.timeseries_data['feature']}")
        report_lines.append(f"Timesteps: {self.timeseries_data['n_timesteps']}")
        
        report_lines.append("\nCluster Distribution:")
        unique, counts = np.unique(self.cluster_labels, return_counts=True)
        for cluster_id, count in zip(unique, counts):
            percentage = (count / len(self.cluster_labels)) * 100
            report_lines.append(f"  Cluster {cluster_id}: {count:5,} parcels ({percentage:5.2f}%)")
        
        # Cluster characteristics
        if 'NDVI_Mean' in cluster_df.columns:
            report_lines.append("\nCluster Characteristics:")
            for cluster_id in unique:
                cluster_data = cluster_df[cluster_df['Cluster'] == cluster_id]
                report_lines.append(f"\n  Cluster {cluster_id}:")
                report_lines.append(f"    Mean NDVI:  {cluster_data['NDVI_Mean'].mean():.4f} ± {cluster_data['NDVI_Mean'].std():.4f}")
                report_lines.append(f"    Peak NDVI:  {cluster_data['NDVI_Peak_Value'].mean():.4f} ± {cluster_data['NDVI_Peak_Value'].std():.4f}")
                report_lines.append(f"    NDVI Slope: {cluster_data['NDVI_Slope'].mean():.4f} ± {cluster_data['NDVI_Slope'].std():.4f}")
        
        # Anomaly detection summary
        report_lines.append("\n" + "="*80)
        report_lines.append("ANOMALY DETECTION RESULTS")
        report_lines.append("="*80)
        report_lines.append(f"\nAnomaly Detection Algorithm: Isolation Forest")
        report_lines.append(f"Total Parcels Analyzed: {len(anomaly_df):,}")
        report_lines.append(f"Anomalies Detected: {anomaly_df['Is_Anomaly'].sum():,} ({anomaly_df['Is_Anomaly'].sum() / len(anomaly_df) * 100:.2f}%)")
        report_lines.append(f"Normal Parcels: {(~anomaly_df['Is_Anomaly']).sum():,}")
        
        report_lines.append(f"\nAnomaly Score Statistics:")
        report_lines.append(f"  Mean Score: {anomaly_df['Anomaly_Score'].mean():.4f}")
        report_lines.append(f"  Std Score:  {anomaly_df['Anomaly_Score'].std():.4f}")
        report_lines.append(f"  Min Score:  {anomaly_df['Anomaly_Score'].min():.4f} (most anomalous)")
        report_lines.append(f"  Max Score:  {anomaly_df['Anomaly_Score'].max():.4f} (most normal)")
        
        # Output files
        report_lines.append("\n" + "="*80)
        report_lines.append("OUTPUT FILES")
        report_lines.append("="*80)
        report_lines.append(f"\n✅ Cluster Analysis:")
        report_lines.append(f"  • cluster_assignments.csv ({len(cluster_df):,} parcels)")
        
        report_lines.append(f"\n✅ Anomaly Detection:")
        report_lines.append(f"  • anomaly_scores.csv ({len(anomaly_df):,} parcels)")
        report_lines.append(f"  • top_anomalies.csv (top 20 anomalies)")
        
        report_lines.append(f"\n✅ Visualizations:")
        report_lines.append(f"  • cluster_centroids.png")
        report_lines.append(f"  • cluster_distribution.png")
        report_lines.append(f"  • cluster_characteristics.png")
        report_lines.append(f"  • anomaly_analysis.png")
        report_lines.append(f"  • top_anomalies_timeseries.png")
        
        # Insights
        report_lines.append("\n" + "="*80)
        report_lines.append("KEY INSIGHTS")
        report_lines.append("="*80)
        report_lines.append("\n✨ Growth Patterns Identified:")
        report_lines.append("  DTW clustering successfully grouped parcels with similar growth")
        report_lines.append("  trajectories, even when temporally misaligned (different planting dates).")
        
        report_lines.append("\n✨ Anomalous Patterns:")
        report_lines.append(f"  {anomaly_df['Is_Anomaly'].sum()} parcels show unusual growth patterns.")
        report_lines.append("  These may indicate crop stress, disease, or environmental factors.")
        
        # Next steps
        report_lines.append("\n" + "="*80)
        report_lines.append("NEXT STEPS: PHASE 4")
        report_lines.append("="*80)
        report_lines.append("\n✨ Ready for Phase 4: Predictive Modeling & Evaluation")
        report_lines.append("\n  Use cluster_assignments.csv and anomaly_scores.csv for:")
        report_lines.append("  • Training Random Forest and XGBoost models")
        report_lines.append("  • Building LSTM networks for temporal prediction")
        report_lines.append("  • Predicting yield and crop stress classification")
        
        report_lines.append("\n" + "="*80)
        report_lines.append("PHASE 3 COMPLETE ✅")
        report_lines.append("="*80 + "\n")
        
        # Save report
        report_path = self.output_dir / 'reports' / 'phase3_report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        # Print to console
        print('\n'.join(report_lines))
        
        print(f"💾 Report saved to: {report_path}\n")
    
    def run_pipeline(self, feature='Mean_NDVI', n_clusters=5, max_parcels=None):
        """
        Run the complete Phase 3 pipeline
        
        Args:
            feature: Feature to use for time-series ('Mean_NDVI' or 'Mean_EVI')
            n_clusters: Number of clusters for DTW K-Means
            max_parcels: Maximum parcels to process (None = all)
        """
        print("\n" + "🚀"*40)
        print("RUNNING PHASE 3 COMPLETE PIPELINE")
        print("🚀"*40 + "\n")
        
        start_time = datetime.now()
        
        # Step 1: Load Phase 2 data
        self.load_phase2_data()
        
        # Step 2: Prepare time-series data
        self.prepare_timeseries_data(feature=feature, max_parcels=max_parcels)
        
        # Step 3: DTW clustering
        cluster_df = self.perform_dtw_clustering(n_clusters=n_clusters)
        
        # Step 4: Anomaly detection
        anomaly_df = self.detect_anomalies()
        
        # Step 5: Visualize clusters
        self.visualize_clusters(cluster_df)
        
        # Step 6: Visualize anomalies
        if anomaly_df is not None:
            self.visualize_anomalies(anomaly_df)
        
        # Step 7: Generate report
        self.generate_report(cluster_df, anomaly_df)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print("\n" + "="*80)
        print("PHASE 3 PIPELINE COMPLETE ✅")
        print("="*80)
        print(f"\n⏱️  Total execution time: {duration:.2f} seconds ({duration/60:.2f} minutes)")
        print(f"📊 Parcels clustered: {len(cluster_df):,}")
        print(f"🎯 Growth patterns identified: {n_clusters}")
        print(f"💾 Outputs saved to: {self.output_dir}")
        print(f"\n✨ Ready for Phase 4: Predictive Modeling\n")


def main():
    """Main execution function"""
    print("\n" + "🌾"*40)
    print("CSCE5380 - Crop Health Monitoring from Remote Sensing")
    print("PHASE 3: Pattern Discovery & Anomaly Detection")
    print("🌾"*40 + "\n")
    
    # Initialize engine
    engine = PatternDiscoveryEngine(
        features_dir="./outputs/phase2/features",
        output_dir="./outputs/phase3"
    )
    
    # Run complete pipeline
    # Set max_parcels=500 for quick testing, None for all parcels
    engine.run_pipeline(
        feature='Mean_NDVI',  # Can also use 'Mean_EVI'
        n_clusters=5,          # Number of growth patterns to identify
        max_parcels=None       # Process all parcels
    )


if __name__ == "__main__":
    main()
