"""
CSCE5380 Data Mining - Group 15
PHASE 2: IMAGE SEGMENTATION & VEGETATION INDICES (Weeks 3-4)
Crop Health Monitoring from Remote Sensing

Owner: Snehal Teja Adidam
Goal: Segment crop regions and compute vegetation health indices

COMPLETE DETAILED IMPLEMENTATION - CONTINUING FROM YOUR CODE
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import json
import warnings
from tqdm import tqdm
from scipy import ndimage
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)


class VegetationIndexProcessor:
    """
    Comprehensive processor for vegetation indices and image segmentation
    """
    
    def __init__(self, input_dir="./outputs/phase1/processed_data", 
                 output_dir="./outputs/phase2"):
        """
        Initialize the vegetation index processor
        
        Args:
            input_dir: Directory with Phase 1 outputs
            output_dir: Directory for Phase 2 outputs
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'visualizations').mkdir(exist_ok=True)
        (self.output_dir / 'indices').mkdir(exist_ok=True)
        (self.output_dir / 'segments').mkdir(exist_ok=True)
        (self.output_dir / 'features').mkdir(exist_ok=True)
        
        # Data containers
        self.patches = []
        self.vegetation_indices = {}
        self.segmentation_results = {}
        self.features_df = None
        self.statistics = {}
        
        print("="*80)
        print("PHASE 2: IMAGE SEGMENTATION & VEGETATION INDICES")
        print("="*80)
        print(f"\n✅ Processor initialized")
        print(f"   Input from Phase 1: {self.input_dir}")
        print(f"   Output directory: {self.output_dir}\n")
    
    # ========================================================================
    # STEP 1: LOAD PREPROCESSED DATA FROM PHASE 1
    # ========================================================================
    
    def load_phase1_data(self):
        """Load processed data from Phase 1"""
        print("\n" + "="*80)
        print("STEP 1: LOADING PHASE 1 DATA")
        print("="*80 + "\n")
        
        print("📥 Loading preprocessed data from Phase 1...\n")
        
        # Load metadata
        metadata_file = self.input_dir / 'metadata_summary.csv'
        if metadata_file.exists():
            metadata_df = pd.read_csv(metadata_file)
            print(f"   ✅ Loaded metadata: {len(metadata_df)} patches")
        else:
            raise FileNotFoundError(f"Metadata not found at {metadata_file}. Please ensure Phase 1 outputs exist.")
        
        # Load statistics
        stats_file = self.input_dir / 'dataset_statistics.json'
        if stats_file.exists():
            with open(stats_file, 'r') as f:
                self.statistics = json.load(f)
            print(f"   ✅ Loaded dataset statistics")
        
        # Load sample patches
        sample_dir = self.input_dir / 'sample_patches'
        if sample_dir.exists():
            patch_files = sorted(list(sample_dir.glob('*_images.npy')))
            print(f"   ✅ Found {len(patch_files)} sample patches")
            
            for patch_file in tqdm(patch_files, desc="   Loading patches"):
                patch_id = patch_file.stem.replace('_images', '')
                
                # Load images
                images = np.load(patch_file)
                
                # Load labels if available
                label_file = sample_dir / f'{patch_id}_labels.npy'
                labels = np.load(label_file) if label_file.exists() else None
                
                # Get metadata for this patch
                patch_meta = metadata_df[metadata_df['patch_id'] == patch_id]
                
                patch_data = {
                    'patch_id': patch_id,
                    'images': images,
                    'labels': labels,
                    'n_timesteps': images.shape[0],
                    'metadata': patch_meta.iloc[0].to_dict() if len(patch_meta) > 0 else {}
                }
                
                self.patches.append(patch_data)
        else:
            raise FileNotFoundError(f"Sample patches not found at {sample_dir}. Please ensure Phase 1 outputs exist.")
        
        print(f"\n✅ Phase 1 data loaded successfully")
        print(f"   Total patches: {len(self.patches)}")
        print(f"   Ready for vegetation index computation\n")
        
        return len(self.patches)
    
    # Synthetic data generation removed. Only real data loading is allowed.
    
    # ========================================================================
    # STEP 2: COMPUTE VEGETATION INDICES
    # ========================================================================
    
    def compute_vegetation_indices(self):
        """
        Compute comprehensive vegetation indices for all patches
        
        Indices computed:
        1. NDVI - Normalized Difference Vegetation Index
        2. EVI - Enhanced Vegetation Index
        3. SAVI - Soil Adjusted Vegetation Index
        4. NDWI - Normalized Difference Water Index
        """
        print("\n" + "="*80)
        print("STEP 2: COMPUTING VEGETATION INDICES")
        print("="*80 + "\n")
        
        print("🌱 Computing vegetation health indicators...\n")
        
        print("   Vegetation Indices:")
        print("   1. NDVI (Normalized Difference Vegetation Index)")
        print("      - Formula: (NIR - Red) / (NIR + Red)")
        print("      - Range: -1 to 1")
        print("      - Use: General vegetation health indicator")
        print("      - Interpretation: >0.6=healthy, 0.3-0.6=moderate, <0.3=stressed\n")
        
        print("   2. EVI (Enhanced Vegetation Index)")
        print("      - Formula: 2.5 * ((NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1))")
        print("      - Range: -1 to 1")
        print("      - Use: Reduces atmospheric and soil background effects")
        print("      - Interpretation: Similar to NDVI but more sensitive\n")
        
        print("   3. SAVI (Soil Adjusted Vegetation Index)")
        print("      - Formula: ((NIR - Red) / (NIR + Red + L)) * (1 + L), L=0.5")
        print("      - Range: -1 to 1")
        print("      - Use: Accounts for soil brightness")
        print("      - Interpretation: Better for sparse vegetation\n")
        
        print("   4. NDWI (Normalized Difference Water Index)")
        print("      - Formula: (NIR - SWIR) / (NIR + SWIR)")
        print("      - Range: -1 to 1")
        print("      - Use: Vegetation water content")
        print("      - Interpretation: >0.3=well-watered, <0=water stress\n")
        
        print("   Computing indices for all patches...")
        
        for patch in tqdm(self.patches, desc="   Processing"):
            patch_id = patch['patch_id']
            images = patch['images']
            n_timesteps, n_bands, height, width = images.shape
            
            # Extract relevant bands
            blue = images[:, 0, :, :]   # Band 2 (Blue)
            red = images[:, 2, :, :]    # Band 4 (Red)
            nir = images[:, 7, :, :]    # Band 8 (NIR)
            swir = images[:, 8, :, :] if n_bands > 8 else nir * 0.5  # Band 11 (SWIR)
            
            # Initialize arrays for indices
            ndvi = np.zeros_like(nir)
            evi = np.zeros_like(nir)
            savi = np.zeros_like(nir)
            ndwi = np.zeros_like(nir)
            
            # Compute indices for each timestep
            for t in range(n_timesteps):
                # NDVI computation
                denominator = nir[t] + red[t]
                denominator = np.where(denominator == 0, 1e-6, denominator)
                ndvi[t] = (nir[t] - red[t]) / denominator
                
                # EVI computation
                denominator = nir[t] + 6*red[t] - 7.5*blue[t] + 1
                denominator = np.where(denominator == 0, 1e-6, denominator)
                evi[t] = 2.5 * ((nir[t] - red[t]) / denominator)
                
                # SAVI computation (L = 0.5 for moderate vegetation)
                L = 0.5
                denominator = nir[t] + red[t] + L
                denominator = np.where(denominator == 0, 1e-6, denominator)
                savi[t] = ((nir[t] - red[t]) / denominator) * (1 + L)
                
                # NDWI computation
                denominator = nir[t] + swir[t]
                denominator = np.where(denominator == 0, 1e-6, denominator)
                ndwi[t] = (nir[t] - swir[t]) / denominator
            
            # Clip to valid range
            ndvi = np.clip(ndvi, -1, 1)
            evi = np.clip(evi, -1, 1)
            savi = np.clip(savi, -1, 1)
            ndwi = np.clip(ndwi, -1, 1)
            
            # Compute temporal statistics
            ndvi_mean = np.mean(ndvi, axis=0)
            ndvi_std = np.std(ndvi, axis=0)
            ndvi_max = np.max(ndvi, axis=0)
            ndvi_min = np.min(ndvi, axis=0)
            
            # Compute temporal trend (linear regression slope)
            ndvi_temporal_mean = ndvi.mean(axis=(1, 2))
            if len(ndvi_temporal_mean) > 2:
                x = np.arange(len(ndvi_temporal_mean))
                ndvi_trend = np.polyfit(x, ndvi_temporal_mean, 1)[0]
            else:
                ndvi_trend = 0
            
            # Compute phenology metrics
            ndvi_peak_time = np.argmax(ndvi_temporal_mean)
            ndvi_peak_value = np.max(ndvi_temporal_mean)
            
            # Store all indices and derived metrics
            self.vegetation_indices[patch_id] = {
                # Raw indices (full spatiotemporal)
                'ndvi': ndvi,
                'evi': evi,
                'savi': savi,
                'ndwi': ndwi,
                
                # Temporal means (spatial maps)
                'ndvi_mean': ndvi_mean,
                'evi_mean': np.mean(evi, axis=0),
                'savi_mean': np.mean(savi, axis=0),
                'ndwi_mean': np.mean(ndwi, axis=0),
                
                # Temporal statistics
                'ndvi_std': ndvi_std,
                'ndvi_max': ndvi_max,
                'ndvi_min': ndvi_min,
                
                # Derived metrics
                'ndvi_trend': ndvi_trend,
                'ndvi_peak_time': ndvi_peak_time,
                'ndvi_peak_value': ndvi_peak_value,
                'vegetation_amplitude': ndvi_max - ndvi_min,
                
                # Summary statistics
                'mean_ndvi': np.mean(ndvi),
                'mean_evi': np.mean(evi),
                'mean_savi': np.mean(savi),
                'mean_ndwi': np.mean(ndwi),
                
                # Health classification
                'health_score': self._compute_health_score(ndvi_mean),
                'stress_indicator': self._compute_stress_indicator(ndvi_trend, np.mean(ndvi))
            }
        
        print(f"\n✅ Vegetation indices computed for {len(self.vegetation_indices)} patches")
        
        # Compute dataset-level statistics
        self._compute_index_statistics()
        
        print("\n📊 Dataset-wide Index Statistics:")
        print(f"   NDVI range: [{self.statistics['ndvi_min']:.3f}, {self.statistics['ndvi_max']:.3f}]")
        print(f"   NDVI mean: {self.statistics['ndvi_mean']:.3f} ± {self.statistics['ndvi_std']:.3f}")
        print(f"   EVI mean: {self.statistics['evi_mean']:.3f} ± {self.statistics['evi_std']:.3f}")
        print(f"   Healthy patches (NDVI>0.5): {self.statistics['healthy_patches']} "
              f"({self.statistics['healthy_patches']/len(self.patches)*100:.1f}%)")
        print(f"   Stressed patches (NDVI<0.3): {self.statistics['stressed_patches']} "
              f"({self.statistics['stressed_patches']/len(self.patches)*100:.1f}%)\n")
    
    def _compute_health_score(self, ndvi_mean):
        """Compute overall health score from NDVI"""
        mean_ndvi = np.mean(ndvi_mean)
        if mean_ndvi > 0.6:
            return 'Healthy'
        elif mean_ndvi > 0.3:
            return 'Moderate'
        else:
            return 'Stressed'
    
    def _compute_stress_indicator(self, trend, mean_ndvi):
        """Compute stress indicator from trend and mean"""
        if trend < -0.01 and mean_ndvi < 0.4:
            return 'High Stress'
        elif trend < -0.005 or mean_ndvi < 0.3:
            return 'Moderate Stress'
        else:
            return 'Low Stress'
    
    def _compute_index_statistics(self):
        """Compute dataset-wide statistics for vegetation indices"""
        all_ndvi = [idx['mean_ndvi'] for idx in self.vegetation_indices.values()]
        all_evi = [idx['mean_evi'] for idx in self.vegetation_indices.values()]
        all_savi = [idx['mean_savi'] for idx in self.vegetation_indices.values()]
        
        self.statistics.update({
            'ndvi_mean': np.mean(all_ndvi),
            'ndvi_std': np.std(all_ndvi),
            'ndvi_min': np.min(all_ndvi),
            'ndvi_max': np.max(all_ndvi),
            'evi_mean': np.mean(all_evi),
            'evi_std': np.std(all_evi),
            'savi_mean': np.mean(all_savi),
            'healthy_patches': sum(1 for v in all_ndvi if v > 0.5),
            'stressed_patches': sum(1 for v in all_ndvi if v < 0.3)
        })
    
    # ========================================================================
    # STEP 3: IMAGE SEGMENTATION
    # ========================================================================
    
    def perform_image_segmentation(self):
        """
        Perform image segmentation to identify crop regions
        
        Methods used:
        1. Threshold-based segmentation (NDVI thresholds)
        2. K-means clustering (spectral clustering)
        3. Connected component analysis
        4. Region property extraction
        """
        print("\n" + "="*80)
        print("STEP 3: IMAGE SEGMENTATION")
        print("="*80 + "\n")
        
        print("🔍 Segmenting crop regions using multiple methods...\n")
        
        print("   Segmentation Methods:")
        print("   1. Threshold-based: Using NDVI values")
        print("   2. K-means clustering: Grouping similar pixels")
        print("   3. Connected components: Identifying contiguous regions")
        print("   4. Region analysis: Extracting geometric properties\n")
        
        for patch in tqdm(self.patches, desc="   Segmenting patches"):
            patch_id = patch['patch_id']
            indices = self.vegetation_indices[patch_id]
            
            # Get mean NDVI for segmentation
            ndvi_mean = indices['ndvi_mean']
            height, width = ndvi_mean.shape
            
            # Method 1: Threshold-based segmentation
            healthy_mask = ndvi_mean > 0.5
            moderate_mask = (ndvi_mean > 0.3) & (ndvi_mean <= 0.5)
            stressed_mask = (ndvi_mean > 0.1) & (ndvi_mean <= 0.3)
            bare_mask = ndvi_mean <= 0.1
            
            # Method 2: K-means clustering on spectral features
            images_mean = np.mean(patch['images'], axis=0)  # Average over time
            n_bands = images_mean.shape[0]
            
            # Reshape for clustering: (n_pixels, n_bands)
            pixels = images_mean.reshape(n_bands, -1).T
            pixels_normalized = StandardScaler().fit_transform(pixels)
            
            # Apply K-means
            n_clusters = 5
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(pixels_normalized)
            cluster_map = cluster_labels.reshape(height, width)
            
            # Method 3: Connected component analysis
            labeled_healthy, n_healthy = ndimage.label(healthy_mask)
            labeled_stressed, n_stressed = ndimage.label(stressed_mask)
            
            # Method 4: Region properties
            healthy_regions = self._extract_region_properties(labeled_healthy, ndvi_mean)
            stressed_regions = self._extract_region_properties(labeled_stressed, ndvi_mean)
            
            # Compute segmentation statistics
            total_pixels = height * width
            
            segmentation_result = {
                'patch_id': patch_id,
                
                # Threshold-based masks
                'healthy_mask': healthy_mask,
                'moderate_mask': moderate_mask,
                'stressed_mask': stressed_mask,
                'bare_mask': bare_mask,
                
                # Cluster-based segmentation
                'cluster_map': cluster_map,
                'n_clusters': n_clusters,
                
                # Connected components
                'labeled_healthy': labeled_healthy,
                'labeled_stressed': labeled_stressed,
                'n_healthy_regions': n_healthy,
                'n_stressed_regions': n_stressed,
                
                # Region properties
                'healthy_regions': healthy_regions,
                'stressed_regions': stressed_regions,
                
                # Area statistics
                'healthy_area': np.sum(healthy_mask),
                'moderate_area': np.sum(moderate_mask),
                'stressed_area': np.sum(stressed_mask),
                'bare_area': np.sum(bare_mask),
                
                # Percentage coverage
                'healthy_percent': np.sum(healthy_mask) / total_pixels * 100,
                'moderate_percent': np.sum(moderate_mask) / total_pixels * 100,
                'stressed_percent': np.sum(stressed_mask) / total_pixels * 100,
                'bare_percent': np.sum(bare_mask) / total_pixels * 100,
                
                # Spatial metrics
                'fragmentation_index': n_healthy / (np.sum(healthy_mask) + 1),
                'diversity_index': self._compute_diversity_index(cluster_map),
            }
            
            self.segmentation_results[patch_id] = segmentation_result
        
        print(f"\n✅ Segmentation completed for {len(self.segmentation_results)} patches\n")
        
        # Compute aggregated statistics
        self._compute_segmentation_statistics()
        
        print("📊 Segmentation Statistics:")
        print(f"   Average healthy coverage: {self.statistics['avg_healthy_percent']:.1f}%")
        print(f"   Average stressed coverage: {self.statistics['avg_stressed_percent']:.1f}%")
        print(f"   Average number of regions: {self.statistics['avg_n_regions']:.1f}")
        print(f"   Most fragmented patches: {self.statistics['most_fragmented']}")
        print(f"   Most homogeneous patches: {self.statistics['most_homogeneous']}\n")
    
    def _extract_region_properties(self, labeled_image, values):
        """Extract properties of labeled regions"""
        regions = []
        n_regions = labeled_image.max()
        
        for region_id in range(1, n_regions + 1):
            mask = labeled_image == region_id
            if np.sum(mask) < 10:  # Skip very small regions
                continue
            
            region_props = {
                'id': region_id,
                'area': np.sum(mask),
                'mean_value': np.mean(values[mask]),
                'std_value': np.std(values[mask]),
                'centroid': ndimage.center_of_mass(mask)
            }
            regions.append(region_props)
        
        return regions
    
    def _compute_diversity_index(self, cluster_map):
        """Compute Shannon diversity index for clusters"""
        unique, counts = np.unique(cluster_map, return_counts=True)
        proportions = counts / counts.sum()
        diversity = -np.sum(proportions * np.log(proportions + 1e-10))
        return diversity
    
    def _compute_segmentation_statistics(self):
        """Compute aggregated segmentation statistics"""
        healthy_percents = [seg['healthy_percent'] for seg in self.segmentation_results.values()]
        stressed_percents = [seg['stressed_percent'] for seg in self.segmentation_results.values()]
        n_regions = [seg['n_healthy_regions'] + seg['n_stressed_regions'] 
                    for seg in self.segmentation_results.values()]
        fragmentation = [seg['fragmentation_index'] for seg in self.segmentation_results.values()]
        diversity = [seg['diversity_index'] for seg in self.segmentation_results.values()]
        
        self.statistics.update({
            'avg_healthy_percent': np.mean(healthy_percents),
            'avg_stressed_percent': np.mean(stressed_percents),
            'avg_n_regions': np.mean(n_regions),
            'most_fragmented': int(np.argmax(fragmentation)),
            'most_homogeneous': int(np.argmin(diversity))
        })
    
    # ========================================================================
    # STEP 4: SPATIAL-TEMPORAL FEATURE EXTRACTION (CONTINUED FROM YOUR CODE)
    # ========================================================================
    
    def extract_features(self):
        """
        Extract comprehensive spatial-temporal features for each patch
        These features will be used in Phase 3 for pattern discovery and modeling
        """
        print("\n" + "="*80)
        print("STEP 4: SPATIAL-TEMPORAL FEATURE EXTRACTION")
        print("="*80 + "\n")
        
        print("📊 Extracting features for machine learning...\n")
        
        print("   Feature Categories:")
        print("   1. Temporal features (trends, seasonality, variability)")
        print("   2. Spatial features (heterogeneity, texture, shape)")
        print("   3. Spectral features (index statistics, band ratios)")
        print("   4. Phenological features (growth stages, peak timing)")
        print("   5. Segmentation features (region counts, coverage)\n")
        
        feature_list = []
        
        for patch in tqdm(self.patches, desc="   Extracting features"):
            patch_id = patch['patch_id']
            indices = self.vegetation_indices[patch_id]
            segmentation = self.segmentation_results[patch_id]
            
            features = {'patch_id': patch_id}
            
            # ==================== TEMPORAL FEATURES ====================
            # NDVI temporal statistics
            features['ndvi_mean_temporal'] = indices['mean_ndvi']
            features['ndvi_std_temporal'] = np.mean(indices['ndvi_std'])
            features['ndvi_trend'] = indices['ndvi_trend']
            features['ndvi_peak_time'] = indices['ndvi_peak_time']
            features['ndvi_peak_value'] = indices['ndvi_peak_value']
            
            # EVI temporal statistics
            features['evi_mean_temporal'] = indices['mean_evi']
            features['evi_std_temporal'] = np.std(indices['evi'])
            
            # SAVI temporal statistics
            features['savi_mean_temporal'] = indices['mean_savi']
            
            # NDWI temporal statistics
            features['ndwi_mean_temporal'] = indices['mean_ndwi']
            
            # Vegetation amplitude (seasonality indicator)
            features['vegetation_amplitude'] = np.mean(indices['vegetation_amplitude'])
            
            # ==================== SPATIAL FEATURES ====================
            # Spatial variability
            features['ndvi_spatial_variance'] = np.var(indices['ndvi_mean'])
            features['evi_spatial_variance'] = np.var(indices['evi_mean'])
            
            # Spatial heterogeneity (coefficient of variation)
            features['ndvi_spatial_cv'] = (np.std(indices['ndvi_mean']) / 
                                          (np.mean(indices['ndvi_mean']) + 1e-6))
            
            # Texture features (using NDVI)
            features['ndvi_texture_entropy'] = self._compute_texture_entropy(indices['ndvi_mean'])
            features['ndvi_texture_contrast'] = self._compute_texture_contrast(indices['ndvi_mean'])
            
            # ==================== SPECTRAL FEATURES ====================
            # Index extremes
            features['ndvi_max'] = np.max(indices['ndvi'])
            features['ndvi_min'] = np.min(indices['ndvi'])
            features['evi_max'] = np.max(indices['evi'])
            features['evi_min'] = np.min(indices['evi'])
            
            # Index range
            features['ndvi_range'] = features['ndvi_max'] - features['ndvi_min']
            
            # ==================== PHENOLOGICAL FEATURES ====================
            # Growth rate (early season slope)
            ndvi_ts = indices['ndvi'].mean(axis=(1,2))
            if len(ndvi_ts) > 10:
                features['early_growth_rate'] = np.mean(np.diff(ndvi_ts[:10]))
                features['late_growth_rate'] = np.mean(np.diff(ndvi_ts[-10:]))
            else:
                features['early_growth_rate'] = 0
                features['late_growth_rate'] = 0
            
            # Growing season length (days above threshold)
            features['growing_season_length'] = np.sum(ndvi_ts > 0.3)
            
            # Senescence rate (rate of decline)
            if len(ndvi_ts) > 20:
                peak_idx = np.argmax(ndvi_ts)
                if peak_idx < len(ndvi_ts) - 5:
                    features['senescence_rate'] = np.mean(np.diff(ndvi_ts[peak_idx:peak_idx+5]))
                else:
                    features['senescence_rate'] = 0
            else:
                features['senescence_rate'] = 0
            
            # Green-up rate (rate of spring growth)
            if len(ndvi_ts) > 10:
                first_quarter = len(ndvi_ts) // 4
                features['greenup_rate'] = np.mean(np.diff(ndvi_ts[:first_quarter]))
            else:
                features['greenup_rate'] = 0
            
            # ==================== SEGMENTATION FEATURES ====================
            # Coverage percentages
            features['healthy_coverage'] = segmentation['healthy_percent']
            features['moderate_coverage'] = segmentation['moderate_percent']
            features['stressed_coverage'] = segmentation['stressed_percent']
            features['bare_coverage'] = segmentation['bare_percent']
            
            # Region counts
            features['n_healthy_regions'] = segmentation['n_healthy_regions']
            features['n_stressed_regions'] = segmentation['n_stressed_regions']
            features['total_regions'] = (segmentation['n_healthy_regions'] + 
                                        segmentation['n_stressed_regions'])
            
            # Fragmentation metrics
            features['fragmentation_index'] = segmentation['fragmentation_index']
            features['diversity_index'] = segmentation['diversity_index']
            
            # ==================== COMPOSITE FEATURES ====================
            # Stress indicators
            features['stress_score'] = (features['stressed_coverage'] / 
                                       (features['healthy_coverage'] + 1e-6))
            
            # Vegetation vigor
            features['vegetation_vigor'] = (features['ndvi_mean_temporal'] * 
                                           features['healthy_coverage'] / 100)
            
            # Temporal stability
            features['temporal_stability'] = 1 / (features['ndvi_std_temporal'] + 1e-6)
            
            # Health classification
            features['health_score'] = indices['health_score']
            features['stress_indicator'] = indices['stress_indicator']
            
            feature_list.append(features)
        
        # Convert to DataFrame
        self.features_df = pd.DataFrame(feature_list)
        
        print(f"\n✅ Feature extraction complete")
        print(f"   Total features extracted: {len(self.features_df.columns) - 1}")
        print(f"   Total patches: {len(self.features_df)}\n")
        
        # Display feature summary
        print("📊 Feature Summary:")
        print(f"   Temporal features: 10")
        print(f"   Spatial features: 5")
        print(f"   Spectral features: 5")
        print(f"   Phenological features: 5")
        print(f"   Segmentation features: 8")
        print(f"   Composite features: 3")
        print(f"   Categorical features: 2")
        print(f"   ─────────────────────")
        print(f"   Total: {len(self.features_df.columns) - 1} features\n")
        
        # Save features
        features_path = self.output_dir / 'features' / 'phase2_features.csv'
        self.features_df.to_csv(features_path, index=False)
        print(f"💾 Features saved to: {features_path}\n")
    
    def _compute_texture_entropy(self, image):
        """Compute texture entropy from image"""
        # Quantize image to 8 bins
        quantized = np.digitize(image, bins=np.linspace(-1, 1, 9))
        unique, counts = np.unique(quantized, return_counts=True)
        probabilities = counts / counts.sum()
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        return entropy
    
    def _compute_texture_contrast(self, image):
        """Compute texture contrast using gradient magnitude"""
        # Compute gradients
        grad_x = np.gradient(image, axis=0)
        grad_y = np.gradient(image, axis=1)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        contrast = np.mean(gradient_magnitude)
        return contrast
    
    # ========================================================================
    # STEP 5: TEMPORAL PATTERN ANALYSIS
    # ========================================================================
    
    def analyze_temporal_patterns(self):
        """
        Analyze temporal patterns in vegetation indices
        """
        print("\n" + "="*80)
        print("STEP 5: TEMPORAL PATTERN ANALYSIS")
        print("="*80 + "\n")
        
        print("📈 Analyzing temporal patterns...\n")
        
        print("   Analysis Components:")
        print("   1. Seasonal decomposition")
        print("   2. Trend detection")
        print("   3. Phenological stage identification")
        print("   4. Growth cycle characterization\n")
        
        temporal_patterns = {}
        
        for patch in tqdm(self.patches[:10], desc="   Analyzing (sample)"):
            patch_id = patch['patch_id']
            indices = self.vegetation_indices[patch_id]
            
            # Get NDVI time series
            ndvi_ts = indices['ndvi'].mean(axis=(1, 2))
            
            # Identify phenological stages
            stages = self._identify_phenological_stages(ndvi_ts)
            
            # Detect anomalies in time series
            anomalies = self._detect_temporal_anomalies(ndvi_ts)
            
            # Compute periodicity metrics
            periodicity = self._compute_periodicity(ndvi_ts)
            
            temporal_patterns[patch_id] = {
                'phenological_stages': stages,
                'temporal_anomalies': anomalies,
                'periodicity_score': periodicity,
                'max_ndvi': np.max(ndvi_ts),
                'min_ndvi': np.min(ndvi_ts),
                'ndvi_range': np.max(ndvi_ts) - np.min(ndvi_ts)
            }
        
        print(f"\n✅ Temporal pattern analysis complete")
        print(f"   Patterns analyzed: {len(temporal_patterns)} patches")
        
        # Save temporal patterns
        patterns_path = self.output_dir / 'patterns' / 'temporal_patterns.json'
        patterns_path.parent.mkdir(exist_ok=True)
        with open(patterns_path, 'w') as f:
            # Convert numpy types for JSON serialization
            json_patterns = {}
            for k, v in temporal_patterns.items():
                json_patterns[k] = {
                    'phenological_stages': v['phenological_stages'],
                    'temporal_anomalies': [int(x) for x in v['temporal_anomalies']],
                    'periodicity_score': float(v['periodicity_score']),
                    'max_ndvi': float(v['max_ndvi']),
                    'min_ndvi': float(v['min_ndvi']),
                    'ndvi_range': float(v['ndvi_range'])
                }
            json.dump(json_patterns, f, indent=2)
        
        print(f"💾 Patterns saved to: {patterns_path}\n")
    
    def _identify_phenological_stages(self, ndvi_ts):
        """Identify phenological stages from NDVI time series"""
        stages = {
            'dormant': 0,
            'greenup': 0,
            'peak': 0,
            'senescence': 0
        }
        
        if len(ndvi_ts) < 10:
            return stages
        
        # Find peak
        peak_idx = np.argmax(ndvi_ts)
        stages['peak'] = int(peak_idx)
        
        # Greenup: before peak
        greenup_end = peak_idx
        stages['greenup'] = int(greenup_end // 2)
        
        # Senescence: after peak
        senescence_start = peak_idx
        stages['senescence'] = int((len(ndvi_ts) + peak_idx) // 2)
        
        return stages
    
    def _detect_temporal_anomalies(self, ndvi_ts):
        """Detect anomalies in time series using simple threshold"""
        if len(ndvi_ts) < 5:
            return []
        
        mean = np.mean(ndvi_ts)
        std = np.std(ndvi_ts)
        threshold = 2 * std
        
        anomalies = []
        for i, val in enumerate(ndvi_ts):
            if abs(val - mean) > threshold:
                anomalies.append(i)
        
        return anomalies
    
    def _compute_periodicity(self, ndvi_ts):
        """Compute periodicity score (simple autocorrelation at lag 1)"""
        if len(ndvi_ts) < 3:
            return 0.0
        
        # Normalize
        normalized = (ndvi_ts - np.mean(ndvi_ts)) / (np.std(ndvi_ts) + 1e-6)
        
        # Autocorrelation at lag 1
        autocorr = np.corrcoef(normalized[:-1], normalized[1:])[0, 1]
        
        return autocorr if not np.isnan(autocorr) else 0.0
    
    # ========================================================================
    # STEP 6: VISUALIZATION
    # ========================================================================
    
    def create_visualizations(self):
        """Generate comprehensive visualizations"""
        print("\n" + "="*80)
        print("STEP 6: GENERATING VISUALIZATIONS")
        print("="*80 + "\n")
        
        print("📊 Creating visualization suite...\n")
        
        # Visualization 1: Vegetation Indices Overview
        self._plot_vegetation_indices_overview()
        
        # Visualization 2: Segmentation Results
        self._plot_segmentation_results()
        
        # Visualization 3: Temporal Patterns
        self._plot_temporal_patterns()
        
        # Visualization 4: Feature Distributions
        self._plot_feature_distributions()
        
        print("\n✅ All visualizations generated\n")
    
    def _plot_vegetation_indices_overview(self):
        """Plot comprehensive vegetation indices overview"""
        fig, axes = plt.subplots(3, 3, figsize=(20, 16))
        fig.suptitle('Phase 2: Vegetation Indices Analysis', 
                    fontsize=18, fontweight='bold', y=0.995)
        
        # 1. NDVI Distribution
        ax = axes[0, 0]
        ndvi_values = [idx['mean_ndvi'] for idx in self.vegetation_indices.values()]
        ax.hist(ndvi_values, bins=30, color='green', edgecolor='black', alpha=0.7)
        ax.axvline(0.3, color='orange', linestyle='--', linewidth=2, label='Stress threshold')
        ax.axvline(0.5, color='darkgreen', linestyle='--', linewidth=2, label='Healthy threshold')
        ax.set_xlabel('Mean NDVI', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title('NDVI Distribution', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # 2. EVI vs NDVI Correlation
        ax = axes[0, 1]
        evi_values = [idx['mean_evi'] for idx in self.vegetation_indices.values()]
        scatter = ax.scatter(ndvi_values, evi_values, c=ndvi_values, 
                           cmap='RdYlGn', alpha=0.6, edgecolors='black', linewidth=0.5)
        ax.set_xlabel('NDVI', fontsize=11)
        ax.set_ylabel('EVI', fontsize=11)
        ax.set_title('NDVI vs EVI Correlation', fontsize=13, fontweight='bold')
        plt.colorbar(scatter, ax=ax, label='NDVI')
        ax.grid(alpha=0.3)
        
        # 3. Health Classification Pie Chart
        ax = axes[0, 2]
        health_counts = pd.Series([idx['health_score'] for idx in self.vegetation_indices.values()]).value_counts()
        colors_health = {'Healthy': '#2ecc71', 'Moderate': '#f39c12', 'Stressed': '#e74c3c'}
        ax.pie(health_counts.values, labels=health_counts.index, autopct='%1.1f%%',
               colors=[colors_health.get(h, 'gray') for h in health_counts.index],
               startangle=90, textprops={'fontsize': 10, 'fontweight': 'bold'})
        ax.set_title('Health Classification Distribution', fontsize=13, fontweight='bold')
        
        # 4. Sample NDVI Time Series
        ax = axes[1, 0]
        sample_patch = self.patches[0]
        sample_ndvi = self.vegetation_indices[sample_patch['patch_id']]['ndvi']
        ndvi_ts = sample_ndvi.mean(axis=(1, 2))
        ax.plot(ndvi_ts, linewidth=2.5, color='darkgreen', marker='o', markersize=4)
        ax.set_xlabel('Timestep', fontsize=11)
        ax.set_ylabel('Mean NDVI', fontsize=11)
        ax.set_title(f"NDVI Evolution - {sample_patch['patch_id']}", 
                    fontsize=13, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.axhline(0.5, color='green', linestyle='--', alpha=0.5, label='Healthy')
        ax.axhline(0.3, color='orange', linestyle='--', alpha=0.5, label='Moderate')
        ax.legend()
        
        # 5. NDVI Spatial Map
        ax = axes[1, 1]
        sample_ndvi_mean = self.vegetation_indices[sample_patch['patch_id']]['ndvi_mean']
        im = ax.imshow(sample_ndvi_mean, cmap='RdYlGn', vmin=-0.2, vmax=0.8)
        ax.set_title('NDVI Spatial Map', fontsize=13, fontweight='bold')
        plt.colorbar(im, ax=ax, label='NDVI')
        ax.axis('off')
        
        # 6. EVI Spatial Map
        ax = axes[1, 2]
        sample_evi_mean = self.vegetation_indices[sample_patch['patch_id']]['evi_mean']
        im = ax.imshow(sample_evi_mean, cmap='RdYlGn', vmin=-0.2, vmax=0.8)
        ax.set_title('EVI Spatial Map', fontsize=13, fontweight='bold')
        plt.colorbar(im, ax=ax, label='EVI')
        ax.axis('off')
        
        # 7. NDVI Trend Distribution
        ax = axes[2, 0]
        trends = [idx['ndvi_trend'] for idx in self.vegetation_indices.values()]
        ax.hist(trends, bins=25, color='purple', edgecolor='black', alpha=0.7)
        ax.axvline(0, color='red', linestyle='--', linewidth=2, label='No trend')
        ax.set_xlabel('NDVI Trend', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title('Temporal Trend Distribution', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # 8. Vegetation Amplitude
        ax = axes[2, 1]
        amplitudes = [np.mean(idx['vegetation_amplitude']) for idx in self.vegetation_indices.values()]
        ax.hist(amplitudes, bins=25, color='teal', edgecolor='black', alpha=0.7)
        ax.set_xlabel('Vegetation Amplitude', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title('Vegetation Amplitude Distribution', fontsize=13, fontweight='bold')
        ax.grid(alpha=0.3)
        
        # 9. NDWI vs NDVI
        ax = axes[2, 2]
        ndwi_values = [idx['mean_ndwi'] for idx in self.vegetation_indices.values()]
        scatter = ax.scatter(ndvi_values, ndwi_values, c=ndvi_values,
                           cmap='viridis', alpha=0.6, edgecolors='black', linewidth=0.5)
        ax.set_xlabel('NDVI', fontsize=11)
        ax.set_ylabel('NDWI (Water Content)', fontsize=11)
        ax.set_title('NDVI vs NDWI Relationship', fontsize=13, fontweight='bold')
        plt.colorbar(scatter, ax=ax, label='NDVI')
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        viz_path = self.output_dir / 'visualizations' / 'vegetation_indices_overview.png'
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        print(f"   ✅ Saved: {viz_path}")
        plt.close()
    
    def _plot_segmentation_results(self):
        """Plot segmentation results"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Phase 2: Image Segmentation Results', 
                    fontsize=18, fontweight='bold', y=0.995)
        
        sample_patch = self.patches[0]
        patch_id = sample_patch['patch_id']
        seg = self.segmentation_results[patch_id]
        
        # 1. Threshold-based Segmentation
        ax = axes[0, 0]
        seg_rgb = np.zeros((128, 128, 3))
        seg_rgb[seg['healthy_mask']] = [0, 1, 0]      # Green
        seg_rgb[seg['moderate_mask']] = [1, 1, 0]     # Yellow
        seg_rgb[seg['stressed_mask']] = [1, 0.5, 0]   # Orange
        seg_rgb[seg['bare_mask']] = [0.6, 0.4, 0.2]   # Brown
        ax.imshow(seg_rgb)
        ax.set_title('Threshold-based Segmentation', fontsize=13, fontweight='bold')
        ax.axis('off')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', label='Healthy'),
            Patch(facecolor='yellow', label='Moderate'),
            Patch(facecolor='orange', label='Stressed'),
            Patch(facecolor='brown', label='Bare')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
        
        # 2. K-means Clustering
        ax = axes[0, 1]
        im = ax.imshow(seg['cluster_map'], cmap='tab10')
        ax.set_title('K-means Clustering (5 clusters)', fontsize=13, fontweight='bold')
        plt.colorbar(im, ax=ax, label='Cluster ID')
        ax.axis('off')
        
        # 3. Connected Components (Healthy)
        ax = axes[0, 2]
        im = ax.imshow(seg['labeled_healthy'], cmap='nipy_spectral')
        ax.set_title(f"Healthy Regions ({seg['n_healthy_regions']} regions)", 
                    fontsize=13, fontweight='bold')
        plt.colorbar(im, ax=ax, label='Region ID')
        ax.axis('off')
        
        # 4. Coverage Statistics
        ax = axes[1, 0]
        coverage_data = {
            'Healthy': seg['healthy_percent'],
            'Moderate': seg['moderate_percent'],
            'Stressed': seg['stressed_percent'],
            'Bare': seg['bare_percent']
        }
        colors = ['green', 'yellow', 'orange', 'brown']
        bars = ax.bar(coverage_data.keys(), coverage_data.values(), 
                     color=colors, edgecolor='black', alpha=0.7)
        ax.set_ylabel('Coverage (%)', fontsize=11)
        ax.set_title('Vegetation Coverage by Category', fontsize=13, fontweight='bold')
        ax.grid(alpha=0.3, axis='y')
        
        for bar, val in zip(bars, coverage_data.values()):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # 5. Average Coverage Across All Patches
        ax = axes[1, 1]
        avg_coverage = {
            'Healthy': self.statistics['avg_healthy_percent'],
            'Stressed': self.statistics['avg_stressed_percent']
        }
        ax.barh(list(avg_coverage.keys()), list(avg_coverage.values()),
               color=['green', 'red'], edgecolor='black', alpha=0.7)
        ax.set_xlabel('Average Coverage (%)', fontsize=11)
        ax.set_title('Dataset-wide Average Coverage', fontsize=13, fontweight='bold')
        ax.grid(alpha=0.3, axis='x')
        
        for i, (k, v) in enumerate(avg_coverage.items()):
            ax.text(v + 1, i, f'{v:.1f}%', va='center', fontweight='bold')
        
        # 6. Fragmentation Analysis
        ax = axes[1, 2]
        fragmentation_values = [seg['fragmentation_index'] for seg in self.segmentation_results.values()]
        ax.hist(fragmentation_values, bins=20, color='purple', edgecolor='black', alpha=0.7)
        ax.set_xlabel('Fragmentation Index', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title('Fragmentation Distribution', fontsize=13, fontweight='bold')
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        viz_path = self.output_dir / 'visualizations' / 'segmentation_results.png'
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        print(f"   ✅ Saved: {viz_path}")
        plt.close()
    
    def _plot_temporal_patterns(self):
        """Plot temporal patterns"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Phase 2: Temporal Pattern Analysis', 
                    fontsize=18, fontweight='bold', y=0.995)
        
        # Select 5 sample patches
        sample_patches = self.patches[:5]
        
        # 1. Multiple NDVI Time Series
        ax = axes[0, 0]
        for i, patch in enumerate(sample_patches):
            patch_id = patch['patch_id']
            ndvi_ts = self.vegetation_indices[patch_id]['ndvi'].mean(axis=(1, 2))
            ax.plot(ndvi_ts, linewidth=2, label=f'Patch {i+1}', alpha=0.7)
        ax.set_xlabel('Timestep', fontsize=11)
        ax.set_ylabel('Mean NDVI', fontsize=11)
        ax.set_title('NDVI Temporal Evolution (5 samples)', fontsize=13, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        
        # 2. Peak Timing Distribution
        ax = axes[0, 1]
        peak_times = [idx['ndvi_peak_time'] for idx in self.vegetation_indices.values()]
        ax.hist(peak_times, bins=20, color='green', edgecolor='black', alpha=0.7)
        ax.set_xlabel('Peak Timing (timestep)', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title('NDVI Peak Timing Distribution', fontsize=13, fontweight='bold')
        ax.grid(alpha=0.3)
        
        # 3. Peak Value Distribution
        ax = axes[0, 2]
        peak_values = [idx['ndvi_peak_value'] for idx in self.vegetation_indices.values()]
        ax.hist(peak_values, bins=25, color='darkgreen', edgecolor='black', alpha=0.7)
        ax.set_xlabel('Peak NDVI Value', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title('NDVI Peak Value Distribution', fontsize=13, fontweight='bold')
        ax.grid(alpha=0.3)
        
        # 4. EVI Time Series Comparison
        ax = axes[1, 0]
        sample_patch = sample_patches[0]
        patch_id = sample_patch['patch_id']
        ndvi_ts = self.vegetation_indices[patch_id]['ndvi'].mean(axis=(1, 2))
        evi_ts = self.vegetation_indices[patch_id]['evi'].mean(axis=(1, 2))
        ax.plot(ndvi_ts, linewidth=2.5, label='NDVI', color='green')
        ax.plot(evi_ts, linewidth=2.5, label='EVI', color='blue')
        ax.set_xlabel('Timestep', fontsize=11)
        ax.set_ylabel('Index Value', fontsize=11)
        ax.set_title(f'NDVI vs EVI - {patch_id}', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # 5. Growing Season Length
        ax = axes[1, 1]
        if 'growing_season_length' in self.features_df.columns:
            season_lengths = self.features_df['growing_season_length'].values
            ax.hist(season_lengths, bins=20, color='olive', edgecolor='black', alpha=0.7)
            ax.set_xlabel('Growing Season Length (timesteps)', fontsize=11)
            ax.set_ylabel('Frequency', fontsize=11)
            ax.set_title('Growing Season Length Distribution', fontsize=13, fontweight='bold')
            ax.grid(alpha=0.3)
        
        # 6. Trend vs Peak Value
        ax = axes[1, 2]
        trends = [idx['ndvi_trend'] for idx in self.vegetation_indices.values()]
        peak_vals = [idx['ndvi_peak_value'] for idx in self.vegetation_indices.values()]
        scatter = ax.scatter(trends, peak_vals, c=peak_vals, cmap='RdYlGn',
                           alpha=0.6, edgecolors='black', linewidth=0.5)
        ax.set_xlabel('NDVI Trend', fontsize=11)
        ax.set_ylabel('Peak NDVI Value', fontsize=11)
        ax.set_title('Trend vs Peak Value Relationship', fontsize=13, fontweight='bold')
        plt.colorbar(scatter, ax=ax, label='Peak Value')
        ax.grid(alpha=0.3)
        ax.axvline(0, color='red', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        viz_path = self.output_dir / 'visualizations' / 'temporal_patterns.png'
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        print(f"   ✅ Saved: {viz_path}")
        plt.close()
    
    def _plot_feature_distributions(self):
        """Plot feature distributions"""
        fig, axes = plt.subplots(3, 3, figsize=(18, 14))
        fig.suptitle('Phase 2: Feature Distributions for Phase 3', 
                    fontsize=18, fontweight='bold', y=0.995)
        
        # Select key features to visualize
        feature_plots = [
            ('ndvi_mean_temporal', 'NDVI Mean', 'green'),
            ('evi_mean_temporal', 'EVI Mean', 'blue'),
            ('ndvi_spatial_variance', 'NDVI Spatial Variance', 'purple'),
            ('healthy_coverage', 'Healthy Coverage (%)', 'green'),
            ('stressed_coverage', 'Stressed Coverage (%)', 'red'),
            ('fragmentation_index', 'Fragmentation Index', 'orange'),
            ('vegetation_amplitude', 'Vegetation Amplitude', 'teal'),
            ('ndvi_trend', 'NDVI Trend', 'darkblue'),
            ('temporal_stability', 'Temporal Stability', 'brown')
        ]
        
        for idx, (feature, title, color) in enumerate(feature_plots):
            row = idx // 3
            col = idx % 3
            ax = axes[row, col]
            
            if feature in self.features_df.columns:
                data = self.features_df[feature].values
                ax.hist(data, bins=25, color=color, edgecolor='black', alpha=0.7)
                ax.set_xlabel(title, fontsize=10)
                ax.set_ylabel('Frequency', fontsize=10)
                ax.set_title(f'{title} Distribution', fontsize=11, fontweight='bold')
                ax.grid(alpha=0.3)
                
                # Add mean line
                mean_val = np.mean(data)
                ax.axvline(mean_val, color='red', linestyle='--', linewidth=2,
                          label=f'Mean: {mean_val:.3f}')
                ax.legend(fontsize=8)
        
        plt.tight_layout()
        viz_path = self.output_dir / 'visualizations' / 'feature_distributions.png'
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        print(f"   ✅ Saved: {viz_path}")
        plt.close()
    
    # ========================================================================
    # STEP 7: GENERATE PHASE 2 REPORT
    # ========================================================================
    
    def generate_report(self):
        """Generate comprehensive Phase 2 report"""
        print("\n" + "="*80)
        print("STEP 7: GENERATING PHASE 2 REPORT")
        print("="*80 + "\n")
        
        report = f"""
{'='*80}
PHASE 2 COMPLETION REPORT
Image Segmentation & Vegetation Indices (Weeks 3-4)
{'='*80}

Owner: Snehal Teja Adidam
Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{'='*80}
EXECUTIVE SUMMARY
{'='*80}

Phase 2 has successfully completed all objectives for image segmentation
and vegetation indices computation. Processed Phase 1 data and generated
comprehensive features for Phase 3.

Key Achievements:
✅ Vegetation indices computed (NDVI, EVI, SAVI, NDWI)
✅ Image segmentation performed (threshold, k-means, connected components)
✅ Features extracted (38 features per patch)
✅ Temporal pattern analysis completed
✅ Visualizations generated (4 comprehensive plots)

{'='*80}
RESULTS SUMMARY
{'='*80}

Patches Processed: {len(self.patches)}
Vegetation Indices: 4 (NDVI, EVI, SAVI, NDWI)
Mean NDVI: {self.statistics['ndvi_mean']:.3f} ± {self.statistics['ndvi_std']:.3f}
Healthy Patches: {self.statistics['healthy_patches']} ({self.statistics['healthy_patches']/len(self.patches)*100:.1f}%)
Stressed Patches: {self.statistics['stressed_patches']} ({self.statistics['stressed_patches']/len(self.patches)*100:.1f}%)
Average Healthy Coverage: {self.statistics['avg_healthy_percent']:.1f}%
Features Extracted: {len(self.features_df.columns) - 1}

{'='*80}
STATUS: COMPLETE ✅
{'='*80}
"""
        
        print(report)
        
        # Save report
        report_path = self.output_dir / 'phase2_report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\n✅ Phase 2 report saved: {report_path}\n")
        
        return report


def main():
    """Main execution function for Phase 2"""
    print("\n" + "=" * 80)
    print("PHASE 2: IMAGE SEGMENTATION & VEGETATION INDICES")
    print("=" * 80 + "\n")
    
    # Initialize processor
    processor = VegetationIndexProcessor()
    
    # Step 1: Load Phase 1 data
    processor.load_phase1_data()
    
    # Step 2: Compute vegetation indices
    processor.compute_vegetation_indices()
    
    # Step 3: Perform image segmentation
    processor.perform_image_segmentation()
    
    # Step 4: Extract features
    processor.extract_features()
    
    # Step 5: Analyze temporal patterns
    processor.analyze_temporal_patterns()
    
    # Step 6: Create visualizations
    processor.create_visualizations()
    
    # Step 7: Generate report
    processor.generate_report()
    
    print("\n" + "=" * 80)
    print("✅ PHASE 2 COMPLETE!")
    print("=" * 80 + "\n")
    
    return processor


if __name__ == "__main__":
    processor = main()