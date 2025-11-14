"""
CSCE5380 Data Mining - Group 15
PHASE 2: PARCEL SEGMENTATION & TEMPORAL FEATURE EXTRACTION (Weeks 3-4)
Crop Health Monitoring from Remote Sensing

Owner: Snehal Teja Adidam
Goal: Extract parcel-level temporal features and compute GLCM texture features

This phase loads Phase 1 outputs and creates per-parcel temporal feature datasets
for use in Phase 3 (DTW clustering) and Phase 4 (predictive modeling).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from skimage.feature import graycomatrix, graycoprops
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.figsize'] = (14, 10)


class ParcelFeatureExtractor:
    """
    Extract parcel-level temporal and spatial features from Phase 1 outputs
    """
    
    def __init__(self, input_dir="./outputs/phase1/processed_data/sample_patches", 
                 output_dir="./outputs/phase2"):
        """
        Initialize the parcel feature extractor
        
        Args:
            input_dir: Directory with Phase 1 sample patches
            output_dir: Directory for Phase 2 outputs
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'visualizations').mkdir(exist_ok=True)
        (self.output_dir / 'features').mkdir(exist_ok=True)
        (self.output_dir / 'segments').mkdir(exist_ok=True)
        (self.output_dir / 'patterns').mkdir(exist_ok=True)
        
        # Data containers
        self.temporal_features = []
        self.spatial_features = []
        self.parcel_metadata = []
        
        print("="*80)
        print("PHASE 2: PARCEL SEGMENTATION & TEMPORAL FEATURE EXTRACTION")
        print("="*80)
        print(f"\n✅ Processor initialized")
        print(f"   Input from Phase 1: {self.input_dir}")
        print(f"   Output directory: {self.output_dir}\n")
    
    def extract_temporal_features(self, max_patches=None):
        """
        Extract temporal features for each parcel across all timesteps
        
        For each parcel at each timestep, compute:
        - Mean, Std, P25, P50, P75 of NDVI
        - Mean, Std, P25, P50, P75 of EVI
        - Parcel size (number of pixels)
        
        Args:
            max_patches: Maximum number of patches to process (None = all)
        """
        print("\n" + "="*80)
        print("STEP 1: EXTRACTING TEMPORAL FEATURES")
        print("="*80 + "\n")
        
        print("📊 Computing per-parcel temporal statistics...\n")
        print("   Features computed at each timestep for each parcel:")
        print("   • Mean, Std, P25, P50, P75 of NDVI")
        print("   • Mean, Std, P25, P50, P75 of EVI")
        print("   • Parcel size (pixel count)\n")
        
        # Get list of patch files
        ndvi_files = sorted(list(self.input_dir.glob("*_ndvi.npy")))
        
        if max_patches:
            ndvi_files = ndvi_files[:max_patches]
        
        print(f"   Processing {len(ndvi_files)} patches...\n")
        
        for ndvi_file in tqdm(ndvi_files, desc="   Processing patches"):
            patch_id = ndvi_file.stem.replace('_ndvi', '')
            
            # Load data for this patch
            try:
                ndvi = np.load(ndvi_file)  # Shape: (43, 128, 128)
                evi_file = self.input_dir / f"{patch_id}_evi.npy"
                evi = np.load(evi_file)  # Shape: (43, 128, 128)
                parcels_file = self.input_dir / f"{patch_id}_parcels.npy"
                parcels = np.load(parcels_file)  # Shape: (128, 128)
                labels_file = self.input_dir / f"{patch_id}_labels.npy"
                labels = np.load(labels_file) if labels_file.exists() else None  # Shape: (128, 128)
            except Exception as e:
                print(f"\n   ⚠️  Error loading patch {patch_id}: {e}")
                continue
            
            # Get unique parcel IDs (skip 0 which is background)
            unique_parcels = np.unique(parcels)
            unique_parcels = unique_parcels[unique_parcels != 0]
            
            n_timesteps = ndvi.shape[0]
            
            # Extract features for each parcel
            for parcel_id in unique_parcels:
                # Get parcel mask
                mask = (parcels == parcel_id)
                parcel_size = np.sum(mask)
                
                # Skip very small parcels (< 10 pixels)
                if parcel_size < 10:
                    continue
                
                # Get crop label for this parcel (if available)
                crop_label = None
                if labels is not None and labels.ndim == 2:
                    parcel_labels = labels[mask]
                    # Most common label in parcel
                    unique_labels, counts = np.unique(parcel_labels, return_counts=True)
                    crop_label = unique_labels[np.argmax(counts)]
                
                # Extract temporal features for each timestep
                for t in range(n_timesteps):
                    # Get NDVI and EVI values for this parcel at this timestep
                    parcel_ndvi = ndvi[t][mask]
                    parcel_evi = evi[t][mask]
                    
                    # Compute statistics
                    feature_dict = {
                        'Patch_ID': patch_id,
                        'Parcel_ID': f"{patch_id}_{parcel_id}",
                        'Timestep': t,
                        'Parcel_Size': parcel_size,
                        'Crop_Label': crop_label,
                        
                        # NDVI statistics
                        'Mean_NDVI': np.mean(parcel_ndvi),
                        'Std_NDVI': np.std(parcel_ndvi),
                        'Min_NDVI': np.min(parcel_ndvi),
                        'Max_NDVI': np.max(parcel_ndvi),
                        'P25_NDVI': np.percentile(parcel_ndvi, 25),
                        'P50_NDVI': np.percentile(parcel_ndvi, 50),
                        'P75_NDVI': np.percentile(parcel_ndvi, 75),
                        
                        # EVI statistics
                        'Mean_EVI': np.mean(parcel_evi),
                        'Std_EVI': np.std(parcel_evi),
                        'Min_EVI': np.min(parcel_evi),
                        'Max_EVI': np.max(parcel_evi),
                        'P25_EVI': np.percentile(parcel_evi, 25),
                        'P50_EVI': np.percentile(parcel_evi, 50),
                        'P75_EVI': np.percentile(parcel_evi, 75),
                    }
                    
                    self.temporal_features.append(feature_dict)
        
        # Convert to DataFrame
        df_temporal = pd.DataFrame(self.temporal_features)
        
        print(f"\n✅ Temporal feature extraction complete")
        print(f"   Total features extracted: {len(df_temporal):,} rows")
        print(f"   Unique parcels: {df_temporal['Parcel_ID'].nunique():,}")
        print(f"   Timesteps per parcel: {df_temporal['Timestep'].nunique()}")
        print(f"   Features per row: {len(df_temporal.columns)}\n")
        
        # Save temporal features
        output_file = self.output_dir / 'features' / 'temporal_features.csv'
        df_temporal.to_csv(output_file, index=False)
        print(f"   💾 Saved to: {output_file}\n")
        
        return df_temporal
    
    def extract_spatial_features(self, max_patches=None):
        """
        Extract spatial texture features for each parcel using GLCM
        
        Gray Level Co-occurrence Matrix (GLCM) captures spatial patterns
        by analyzing how often pixel pairs with certain values occur at
        specific distances and angles.
        
        Features extracted:
        - Contrast: Intensity difference between pixel and neighbor
        - Dissimilarity: Similar to contrast but with linear weighting
        - Homogeneity: Closeness of GLCM elements to diagonal
        - Energy: Sum of squared GLCM elements (uniformity)
        - Correlation: Linear dependency of gray levels
        - ASM (Angular Second Moment): Similar to energy
        
        Args:
            max_patches: Maximum number of patches to process (None = all)
        """
        print("\n" + "="*80)
        print("STEP 2: EXTRACTING SPATIAL TEXTURE FEATURES (GLCM)")
        print("="*80 + "\n")
        
        print("🔍 Computing GLCM texture features...\n")
        print("   GLCM (Gray Level Co-occurrence Matrix) analyzes spatial patterns")
        print("   Texture features computed for each parcel:")
        print("   • Contrast: Intensity difference between pixels")
        print("   • Dissimilarity: Similar to contrast (linear)")
        print("   • Homogeneity: Closeness to diagonal")
        print("   • Energy: Uniformity measure")
        print("   • Correlation: Linear dependency")
        print("   • ASM: Angular Second Moment\n")
        
        # Get list of patch files
        ndvi_files = sorted(list(self.input_dir.glob("*_ndvi.npy")))
        
        if max_patches:
            ndvi_files = ndvi_files[:max_patches]
        
        print(f"   Processing {len(ndvi_files)} patches...\n")
        
        for ndvi_file in tqdm(ndvi_files, desc="   Computing GLCM"):
            patch_id = ndvi_file.stem.replace('_ndvi', '')
            
            # Load data
            try:
                ndvi = np.load(ndvi_file)  # Shape: (43, 128, 128)
                images_file = self.input_dir / f"{patch_id}_images.npy"
                images = np.load(images_file) if images_file.exists() else None
                parcels_file = self.input_dir / f"{patch_id}_parcels.npy"
                parcels = np.load(parcels_file)
            except Exception as e:
                print(f"\n   ⚠️  Error loading patch {patch_id}: {e}")
                continue
            
            # Use NIR band for texture analysis (band 6 in original, band 7 in processed)
            # We'll use the mean NDVI over time as a texture source
            mean_ndvi = np.mean(ndvi, axis=0)  # Shape: (128, 128)
            
            # Normalize to 0-255 for GLCM
            mean_ndvi_normalized = ((mean_ndvi - mean_ndvi.min()) / 
                                   (mean_ndvi.max() - mean_ndvi.min() + 1e-8) * 255)
            mean_ndvi_normalized = mean_ndvi_normalized.astype(np.uint8)
            
            # Get unique parcels
            unique_parcels = np.unique(parcels)
            unique_parcels = unique_parcels[unique_parcels != 0]
            
            # Extract GLCM features for each parcel
            for parcel_id in unique_parcels:
                mask = (parcels == parcel_id)
                parcel_size = np.sum(mask)
                
                if parcel_size < 10:
                    continue
                
                # Extract parcel region
                rows, cols = np.where(mask)
                min_row, max_row = rows.min(), rows.max()
                min_col, max_col = cols.min(), cols.max()
                
                # Get parcel subimage
                parcel_ndvi = mean_ndvi_normalized[min_row:max_row+1, min_col:max_col+1]
                parcel_mask = mask[min_row:max_row+1, min_col:max_col+1]
                
                # Apply mask
                parcel_ndvi_masked = parcel_ndvi * parcel_mask
                
                try:
                    # Compute GLCM
                    # distances: [1] - check neighbors at distance 1
                    # angles: [0, π/4, π/2, 3π/4] - check all directions
                    glcm = graycomatrix(
                        parcel_ndvi_masked,
                        distances=[1],
                        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                        levels=256,
                        symmetric=True,
                        normed=True
                    )
                    
                    # Extract properties (averaged over all angles)
                    contrast = graycoprops(glcm, 'contrast').mean()
                    dissimilarity = graycoprops(glcm, 'dissimilarity').mean()
                    homogeneity = graycoprops(glcm, 'homogeneity').mean()
                    energy = graycoprops(glcm, 'energy').mean()
                    correlation = graycoprops(glcm, 'correlation').mean()
                    asm = graycoprops(glcm, 'ASM').mean()
                    
                    feature_dict = {
                        'Patch_ID': patch_id,
                        'Parcel_ID': f"{patch_id}_{parcel_id}",
                        'Parcel_Size': parcel_size,
                        'GLCM_Contrast': contrast,
                        'GLCM_Dissimilarity': dissimilarity,
                        'GLCM_Homogeneity': homogeneity,
                        'GLCM_Energy': energy,
                        'GLCM_Correlation': correlation,
                        'GLCM_ASM': asm
                    }
                    
                    self.spatial_features.append(feature_dict)
                    
                except Exception as e:
                    # Skip parcels with errors (too small, etc.)
                    continue
        
        # Convert to DataFrame
        df_spatial = pd.DataFrame(self.spatial_features)
        
        print(f"\n✅ Spatial feature extraction complete")
        print(f"   Total parcels processed: {len(df_spatial):,}")
        print(f"   GLCM features per parcel: 6\n")
        
        # Save spatial features
        output_file = self.output_dir / 'features' / 'spatial_features.csv'
        df_spatial.to_csv(output_file, index=False)
        print(f"   💾 Saved to: {output_file}\n")
        
        return df_spatial
    
    def create_aggregated_features(self, df_temporal):
        """
        Create aggregated features across all timesteps for each parcel
        
        This creates a single feature vector per parcel by aggregating
        temporal statistics (min, max, mean, std, slope) across time.
        
        Args:
            df_temporal: DataFrame with temporal features
        """
        print("\n" + "="*80)
        print("STEP 3: CREATING AGGREGATED FEATURES")
        print("="*80 + "\n")
        
        print("📈 Aggregating temporal features across all timesteps...\n")
        print("   For each parcel, computing:")
        print("   • Min, Max, Mean, Std of NDVI/EVI over time")
        print("   • Peak values and their timesteps")
        print("   • Temporal slope (linear regression)\n")
        
        aggregated_features = []
        
        for parcel_id in tqdm(df_temporal['Parcel_ID'].unique(), desc="   Aggregating"):
            parcel_data = df_temporal[df_temporal['Parcel_ID'] == parcel_id].sort_values('Timestep')
            
            # Extract time-series
            ndvi_series = parcel_data['Mean_NDVI'].values
            evi_series = parcel_data['Mean_EVI'].values
            timesteps = parcel_data['Timestep'].values
            
            # Compute temporal slope (rate of change)
            if len(timesteps) > 1:
                ndvi_slope, _, _, _, _ = stats.linregress(timesteps, ndvi_series)
                evi_slope, _, _, _, _ = stats.linregress(timesteps, evi_series)
            else:
                ndvi_slope = 0
                evi_slope = 0
            
            # Find peak values
            peak_ndvi_idx = np.argmax(ndvi_series)
            peak_evi_idx = np.argmax(evi_series)
            
            feature_dict = {
                'Parcel_ID': parcel_id,
                'Patch_ID': parcel_data['Patch_ID'].iloc[0],
                'Parcel_Size': parcel_data['Parcel_Size'].iloc[0],
                'Crop_Label': parcel_data['Crop_Label'].iloc[0],
                
                # NDVI aggregated statistics
                'NDVI_Min': np.min(ndvi_series),
                'NDVI_Max': np.max(ndvi_series),
                'NDVI_Mean': np.mean(ndvi_series),
                'NDVI_Std': np.std(ndvi_series),
                'NDVI_Peak_Value': ndvi_series[peak_ndvi_idx],
                'NDVI_Peak_Timestep': timesteps[peak_ndvi_idx],
                'NDVI_Slope': ndvi_slope,
                
                # EVI aggregated statistics
                'EVI_Min': np.min(evi_series),
                'EVI_Max': np.max(evi_series),
                'EVI_Mean': np.mean(evi_series),
                'EVI_Std': np.std(evi_series),
                'EVI_Peak_Value': evi_series[peak_evi_idx],
                'EVI_Peak_Timestep': timesteps[peak_evi_idx],
                'EVI_Slope': evi_slope,
                
                # Derived features
                'NDVI_Range': np.max(ndvi_series) - np.min(ndvi_series),
                'EVI_Range': np.max(evi_series) - np.min(evi_series),
                'NDVI_CV': np.std(ndvi_series) / (np.mean(ndvi_series) + 1e-8),  # Coefficient of variation
                'EVI_CV': np.std(evi_series) / (np.mean(evi_series) + 1e-8),
            }
            
            aggregated_features.append(feature_dict)
        
        df_aggregated = pd.DataFrame(aggregated_features)
        
        print(f"\n✅ Aggregated feature creation complete")
        print(f"   Total parcels: {len(df_aggregated):,}")
        print(f"   Features per parcel: {len(df_aggregated.columns)}\n")
        
        # Save aggregated features
        output_file = self.output_dir / 'features' / 'aggregated_features.csv'
        df_aggregated.to_csv(output_file, index=False)
        print(f"   💾 Saved to: {output_file}\n")
        
        return df_aggregated
    
    def visualize_features(self, df_temporal, df_spatial, df_aggregated):
        """
        Create visualizations of extracted features
        
        Args:
            df_temporal: Temporal features DataFrame
            df_spatial: Spatial features DataFrame
            df_aggregated: Aggregated features DataFrame
        """
        print("\n" + "="*80)
        print("STEP 4: GENERATING VISUALIZATIONS")
        print("="*80 + "\n")
        
        print("📊 Creating feature visualizations...\n")
        
        # 1. Sample parcel time-series
        print("   1. Plotting sample parcel time-series...")
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        sample_parcels = df_temporal['Parcel_ID'].unique()[:4]
        
        for idx, parcel_id in enumerate(sample_parcels):
            ax = axes[idx // 2, idx % 2]
            parcel_data = df_temporal[df_temporal['Parcel_ID'] == parcel_id].sort_values('Timestep')
            
            ax.plot(parcel_data['Timestep'], parcel_data['Mean_NDVI'], 
                   label='NDVI', marker='o', linewidth=2)
            ax.plot(parcel_data['Timestep'], parcel_data['Mean_EVI'], 
                   label='EVI', marker='s', linewidth=2)
            ax.fill_between(parcel_data['Timestep'], 
                          parcel_data['Mean_NDVI'] - parcel_data['Std_NDVI'],
                          parcel_data['Mean_NDVI'] + parcel_data['Std_NDVI'],
                          alpha=0.3)
            ax.set_xlabel('Timestep', fontsize=11)
            ax.set_ylabel('Index Value', fontsize=11)
            ax.set_title(f'Parcel {parcel_id}', fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'visualizations' / 'sample_timeseries.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Feature distributions
        print("   2. Plotting feature distributions...")
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # NDVI/EVI aggregated statistics
        axes[0, 0].hist(df_aggregated['NDVI_Mean'], bins=50, alpha=0.7, color='green', edgecolor='black')
        axes[0, 0].set_xlabel('Mean NDVI', fontsize=11)
        axes[0, 0].set_ylabel('Frequency', fontsize=11)
        axes[0, 0].set_title('Distribution of Mean NDVI', fontsize=12, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].hist(df_aggregated['EVI_Mean'], bins=50, alpha=0.7, color='blue', edgecolor='black')
        axes[0, 1].set_xlabel('Mean EVI', fontsize=11)
        axes[0, 1].set_ylabel('Frequency', fontsize=11)
        axes[0, 1].set_title('Distribution of Mean EVI', fontsize=12, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[0, 2].hist(df_aggregated['NDVI_Slope'], bins=50, alpha=0.7, color='orange', edgecolor='black')
        axes[0, 2].set_xlabel('NDVI Slope', fontsize=11)
        axes[0, 2].set_ylabel('Frequency', fontsize=11)
        axes[0, 2].set_title('Distribution of NDVI Slope', fontsize=12, fontweight='bold')
        axes[0, 2].grid(True, alpha=0.3)
        
        # GLCM features
        if len(df_spatial) > 0:
            axes[1, 0].hist(df_spatial['GLCM_Contrast'], bins=50, alpha=0.7, color='red', edgecolor='black')
            axes[1, 0].set_xlabel('GLCM Contrast', fontsize=11)
            axes[1, 0].set_ylabel('Frequency', fontsize=11)
            axes[1, 0].set_title('Distribution of GLCM Contrast', fontsize=12, fontweight='bold')
            axes[1, 0].grid(True, alpha=0.3)
            
            axes[1, 1].hist(df_spatial['GLCM_Homogeneity'], bins=50, alpha=0.7, color='purple', edgecolor='black')
            axes[1, 1].set_xlabel('GLCM Homogeneity', fontsize=11)
            axes[1, 1].set_ylabel('Frequency', fontsize=11)
            axes[1, 1].set_title('Distribution of GLCM Homogeneity', fontsize=12, fontweight='bold')
            axes[1, 1].grid(True, alpha=0.3)
            
            axes[1, 2].hist(df_spatial['GLCM_Energy'], bins=50, alpha=0.7, color='brown', edgecolor='black')
            axes[1, 2].set_xlabel('GLCM Energy', fontsize=11)
            axes[1, 2].set_ylabel('Frequency', fontsize=11)
            axes[1, 2].set_title('Distribution of GLCM Energy', fontsize=12, fontweight='bold')
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'visualizations' / 'feature_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Correlation heatmap
        print("   3. Creating correlation heatmap...")
        numeric_cols = df_aggregated.select_dtypes(include=[np.number]).columns
        corr_matrix = df_aggregated[numeric_cols].corr()
        
        fig, ax = plt.subplots(figsize=(14, 12))
        im = ax.imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
        
        # Set ticks
        ax.set_xticks(np.arange(len(numeric_cols)))
        ax.set_yticks(np.arange(len(numeric_cols)))
        ax.set_xticklabels(numeric_cols, rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(numeric_cols, fontsize=9)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Correlation', fontsize=11)
        
        ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'visualizations' / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n✅ Visualizations created")
        print(f"   Saved to: {self.output_dir / 'visualizations'}\n")
    
    def generate_report(self, df_temporal, df_spatial, df_aggregated):
        """
        Generate comprehensive Phase 2 report
        
        Args:
            df_temporal: Temporal features DataFrame
            df_spatial: Spatial features DataFrame
            df_aggregated: Aggregated features DataFrame
        """
        print("\n" + "="*80)
        print("STEP 5: GENERATING PHASE 2 REPORT")
        print("="*80 + "\n")
        
        report_lines = []
        report_lines.append("="*80)
        report_lines.append("PHASE 2: PARCEL SEGMENTATION & TEMPORAL FEATURE EXTRACTION")
        report_lines.append("="*80)
        report_lines.append(f"\nReport Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Dataset summary
        report_lines.append("="*80)
        report_lines.append("DATASET SUMMARY")
        report_lines.append("="*80)
        report_lines.append(f"\nTemporal Features:")
        report_lines.append(f"  • Total observations: {len(df_temporal):,}")
        report_lines.append(f"  • Unique parcels: {df_temporal['Parcel_ID'].nunique():,}")
        report_lines.append(f"  • Unique patches: {df_temporal['Patch_ID'].nunique()}")
        report_lines.append(f"  • Timesteps: {df_temporal['Timestep'].nunique()}")
        report_lines.append(f"  • Features per observation: {len(df_temporal.columns)}")
        
        report_lines.append(f"\nSpatial Features (GLCM):")
        report_lines.append(f"  • Total parcels: {len(df_spatial):,}")
        report_lines.append(f"  • GLCM features: 6 (contrast, dissimilarity, homogeneity, energy, correlation, ASM)")
        
        report_lines.append(f"\nAggregated Features:")
        report_lines.append(f"  • Total parcels: {len(df_aggregated):,}")
        report_lines.append(f"  • Features per parcel: {len(df_aggregated.columns)}")
        
        # Feature statistics
        report_lines.append("\n" + "="*80)
        report_lines.append("FEATURE STATISTICS")
        report_lines.append("="*80)
        
        report_lines.append("\nNDVI Aggregated Statistics:")
        report_lines.append(f"  Mean NDVI:  {df_aggregated['NDVI_Mean'].mean():.4f} ± {df_aggregated['NDVI_Mean'].std():.4f}")
        report_lines.append(f"  Peak NDVI:  {df_aggregated['NDVI_Peak_Value'].mean():.4f} ± {df_aggregated['NDVI_Peak_Value'].std():.4f}")
        report_lines.append(f"  NDVI Range: {df_aggregated['NDVI_Range'].mean():.4f} ± {df_aggregated['NDVI_Range'].std():.4f}")
        report_lines.append(f"  NDVI Slope: {df_aggregated['NDVI_Slope'].mean():.4f} ± {df_aggregated['NDVI_Slope'].std():.4f}")
        
        report_lines.append("\nEVI Aggregated Statistics:")
        report_lines.append(f"  Mean EVI:  {df_aggregated['EVI_Mean'].mean():.4f} ± {df_aggregated['EVI_Mean'].std():.4f}")
        report_lines.append(f"  Peak EVI:  {df_aggregated['EVI_Peak_Value'].mean():.4f} ± {df_aggregated['EVI_Peak_Value'].std():.4f}")
        report_lines.append(f"  EVI Range: {df_aggregated['EVI_Range'].mean():.4f} ± {df_aggregated['EVI_Range'].std():.4f}")
        report_lines.append(f"  EVI Slope: {df_aggregated['EVI_Slope'].mean():.4f} ± {df_aggregated['EVI_Slope'].std():.4f}")
        
        if len(df_spatial) > 0:
            report_lines.append("\nGLCM Texture Features:")
            report_lines.append(f"  Contrast:       {df_spatial['GLCM_Contrast'].mean():.4f} ± {df_spatial['GLCM_Contrast'].std():.4f}")
            report_lines.append(f"  Dissimilarity:  {df_spatial['GLCM_Dissimilarity'].mean():.4f} ± {df_spatial['GLCM_Dissimilarity'].std():.4f}")
            report_lines.append(f"  Homogeneity:    {df_spatial['GLCM_Homogeneity'].mean():.4f} ± {df_spatial['GLCM_Homogeneity'].std():.4f}")
            report_lines.append(f"  Energy:         {df_spatial['GLCM_Energy'].mean():.4f} ± {df_spatial['GLCM_Energy'].std():.4f}")
            report_lines.append(f"  Correlation:    {df_spatial['GLCM_Correlation'].mean():.4f} ± {df_spatial['GLCM_Correlation'].std():.4f}")
        
        # Output files
        report_lines.append("\n" + "="*80)
        report_lines.append("OUTPUT FILES")
        report_lines.append("="*80)
        report_lines.append(f"\n✅ Features:")
        report_lines.append(f"  • temporal_features.csv ({len(df_temporal):,} rows)")
        report_lines.append(f"  • spatial_features.csv ({len(df_spatial):,} rows)")
        report_lines.append(f"  • aggregated_features.csv ({len(df_aggregated):,} rows)")
        
        report_lines.append(f"\n✅ Visualizations:")
        report_lines.append(f"  • sample_timeseries.png")
        report_lines.append(f"  • feature_distributions.png")
        report_lines.append(f"  • correlation_heatmap.png")
        
        # Next steps
        report_lines.append("\n" + "="*80)
        report_lines.append("NEXT STEPS: PHASE 3")
        report_lines.append("="*80)
        report_lines.append("\n✨ Ready for Phase 3: Pattern Discovery & Anomaly Detection")
        report_lines.append("\n  Use temporal_features.csv for:")
        report_lines.append("  • DTW-based K-Means clustering (group similar growth patterns)")
        report_lines.append("  • Anomaly detection with Isolation Forest")
        report_lines.append("\n  Use aggregated_features.csv for:")
        report_lines.append("  • Phase 4 predictive modeling (Random Forest, XGBoost, LSTM)")
        
        report_lines.append("\n" + "="*80)
        report_lines.append("PHASE 2 COMPLETE ✅")
        report_lines.append("="*80 + "\n")
        
        # Save report
        report_path = self.output_dir / 'phase2_report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        # Print to console
        print('\n'.join(report_lines))
        
        print(f"💾 Report saved to: {report_path}\n")
    
    def run_pipeline(self, max_patches=None):
        """
        Run the complete Phase 2 pipeline
        
        Args:
            max_patches: Maximum number of patches to process (None = all)
        """
        print("\n" + "🚀"*40)
        print("RUNNING PHASE 2 COMPLETE PIPELINE")
        print("🚀"*40 + "\n")
        
        start_time = datetime.now()
        
        # Step 1: Extract temporal features
        df_temporal = self.extract_temporal_features(max_patches=max_patches)
        
        # Step 2: Extract spatial features (GLCM)
        df_spatial = self.extract_spatial_features(max_patches=max_patches)
        
        # Step 3: Create aggregated features
        df_aggregated = self.create_aggregated_features(df_temporal)
        
        # Step 4: Visualize features
        self.visualize_features(df_temporal, df_spatial, df_aggregated)
        
        # Step 5: Generate report
        self.generate_report(df_temporal, df_spatial, df_aggregated)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print("\n" + "="*80)
        print("PHASE 2 PIPELINE COMPLETE ✅")
        print("="*80)
        print(f"\n⏱️  Total execution time: {duration:.2f} seconds ({duration/60:.2f} minutes)")
        print(f"📊 Parcels processed: {len(df_aggregated):,}")
        print(f"💾 Outputs saved to: {self.output_dir}")
        print(f"\n✨ Ready for Phase 3: DTW Clustering & Anomaly Detection\n")


def main():
    """Main execution function"""
    print("\n" + "🌾"*40)
    print("CSCE5380 - Crop Health Monitoring from Remote Sensing")
    print("PHASE 2: Parcel Segmentation & Temporal Feature Extraction")
    print("🌾"*40 + "\n")
    
    # Initialize processor
    processor = ParcelFeatureExtractor(
        input_dir="./outputs/phase1/processed_data/sample_patches",
        output_dir="./outputs/phase2"
    )
    
    # Run complete pipeline
    # Process all patches (set max_patches=10 for quick testing)
    processor.run_pipeline(max_patches=None)


if __name__ == "__main__":
    main()
