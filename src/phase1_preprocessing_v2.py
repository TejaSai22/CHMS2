"""
CSCE5380 Data Mining - Group 15
PHASE 1: DATA ACQUISITION & PREPROCESSING (Weeks 1-2)
Crop Health Monitoring from Remote Sensing

This script handles:
1. Loading real PASTIS dataset (Sentinel-2 images and parcel annotations)
2. Computing NDVI and EVI vegetation indices for each timestep
3. Normalizing and cleaning data
4. Exporting processed data for Phase 2

NO SYNTHETIC DATA - Uses only real PASTIS data
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
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


class PASTISDatasetProcessor:
    """
    Processor for real PASTIS satellite imagery dataset
    Handles data loading, NDVI/EVI computation, and preprocessing
    """
    
    def __init__(self, data_dir="./data/PASTIS", output_dir="./outputs/phase1"):
        """
        Initialize the PASTIS dataset processor
        
        Args:
            data_dir: Directory containing PASTIS dataset
            output_dir: Directory for outputs and reports
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        
        # Create directory structure
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'visualizations').mkdir(exist_ok=True)
        (self.output_dir / 'processed_data').mkdir(exist_ok=True)
        
        # Data containers
        self.raw_patches = []
        self.metadata_df = None
        self.statistics = {}
        
        # PASTIS Sentinel-2 band indices (0-indexed)
        # PASTIS uses 10 bands: Blue, Green, Red, Red Edge 1-3, NIR, NIR narrow, SWIR 1-2
        self.BAND_BLUE = 0    # B2 (490nm)
        self.BAND_GREEN = 1   # B3 (560nm)
        self.BAND_RED = 2     # B4 (665nm)
        self.BAND_NIR = 6     # B8 (842nm)
        
        print("="*80)
        print("PHASE 1: DATASET ACQUISITION, CLEANING & PREPROCESSING")
        print("="*80)
        print(f"\n✅ Processor initialized")
        print(f"   Data directory: {self.data_dir}")
        print(f"   Output directory: {self.output_dir}\n")

    def load_or_generate_dataset(self, n_patches=100):
        """
        Load REAL PASTIS dataset (NO SYNTHETIC DATA)
        
        Args:
            n_patches: Number of patches to load from real data
        """
        print("\n" + "="*80)
        print("STEP 1: LOADING REAL PASTIS DATASET")
        print("="*80 + "\n")
        
        # Check for data directories
        s2_dir = self.data_dir / 'DATA_S2'
        annotations_dir = self.data_dir / 'ANNOTATIONS'
        
        if not s2_dir.exists():
            raise FileNotFoundError(
                f"❌ PASTIS DATA_S2 directory not found at {s2_dir}\n"
                f"Please ensure PASTIS dataset exists"
            )
        
        if not annotations_dir.exists():
            raise FileNotFoundError(
                f"❌ PASTIS ANNOTATIONS directory not found at {annotations_dir}\n"
                f"Please ensure PASTIS dataset exists"
            )
        
        # Load metadata if available
        metadata_file = self.data_dir / 'metadata.geojson'
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            print(f"   ✅ Loaded metadata with {len(metadata['features'])} entries")
            # Convert to DataFrame for easier access
            metadata_rows = []
            for feature in metadata['features']:
                props = feature['properties']
                metadata_rows.append({
                    'patch_id': props.get('ID_PATCH'),
                    'n_parcels': props.get('N_Parcel'),
                    'tile': props.get('TILE'),
                    'fold': props.get('Fold')
                })
            self.metadata_df = pd.DataFrame(metadata_rows)
        else:
            print(f"   ⚠️  metadata.geojson not found. Continuing without it.")
            self.metadata_df = None
        
        # Load normalization statistics
        norm_file = self.data_dir / 'NORM_S2_patch.json'
        norm_stats = None
        if norm_file.exists():
            with open(norm_file, 'r') as f:
                norm_stats = json.load(f)
            print(f"   ✅ Loaded normalization statistics")
        
        # Load data files
        image_files = sorted(list(s2_dir.glob('S2_*.npy')))[:n_patches]
        
        if len(image_files) == 0:
            raise FileNotFoundError(
                f"❌ No S2_*.npy files found in {s2_dir}\n"
                f"Expected files like: S2_10000.npy, S2_10001.npy, etc."
            )
        
        print(f"   Found {len(image_files)} patch files in dataset")
        print(f"   Loading first {min(n_patches, len(image_files))} patches...\n")
        
        # Statistics accumulators
        all_ndvi = []
        all_evi = []
        
        for img_file in tqdm(image_files, desc="   Loading real data"):
            patch_id = img_file.stem.replace('S2_', '')
            
            try:
                # Load satellite images: shape (T, C, H, W)
                # T = timesteps, C = channels/bands (10), H = height (128), W = width (128)
                images = np.load(img_file).astype(np.float32)
                
                # Load corresponding parcel IDs (not crop labels, but parcel boundaries)
                parcel_file = annotations_dir / f'ParcelIDs_{patch_id}.npy'
                parcel_ids = np.load(parcel_file) if parcel_file.exists() else None
                
                # Load crop type labels if available
                target_file = annotations_dir / f'TARGET_{patch_id}.npy'
                crop_labels = np.load(target_file) if target_file.exists() else None
                
                # Extract temporal information
                n_timesteps, n_bands, height, width = images.shape
                
                # Normalize images using PASTIS normalization stats
                if norm_stats is not None:
                    # Get fold for this patch
                    fold = 1  # default
                    if self.metadata_df is not None:
                        md = self.metadata_df[self.metadata_df['patch_id'] == int(patch_id)]
                        if not md.empty:
                            fold = md.iloc[0]['fold']
                    
                    fold_key = f"Fold_{fold}"
                    if fold_key in norm_stats:
                        means = np.array(norm_stats[fold_key]['mean'])
                        stds = np.array(norm_stats[fold_key]['std'])
                        # Normalize: (x - mean) / std for each band
                        images = (images - means[None, :, None, None]) / stds[None, :, None, None]
                
                # Compute NDVI and EVI for all timesteps
                # NDVI = (NIR - Red) / (NIR + Red)
                # EVI = 2.5 * ((NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1))
                
                nir = images[:, self.BAND_NIR, :, :]    # Shape: (T, H, W)
                red = images[:, self.BAND_RED, :, :]
                blue = images[:, self.BAND_BLUE, :, :]
                
                # Compute NDVI
                denom_ndvi = nir + red
                denom_ndvi = np.where(denom_ndvi == 0, 1e-10, denom_ndvi)  # Avoid division by zero
                ndvi = (nir - red) / denom_ndvi
                ndvi = np.clip(ndvi, -1, 1)  # NDVI values should be in [-1, 1]
                
                # Compute EVI
                denom_evi = nir + 6 * red - 7.5 * blue + 1
                denom_evi = np.where(denom_evi == 0, 1e-10, denom_evi)
                evi = 2.5 * ((nir - red) / denom_evi)
                evi = np.clip(evi, -1, 2)  # EVI values typically in [-1, 2]
                
                # Calculate mean NDVI and EVI across space and time for quality assessment
                mean_ndvi = np.nanmean(ndvi)
                mean_evi = np.nanmean(evi)
                
                all_ndvi.append(mean_ndvi)
                all_evi.append(mean_evi)
                
                # Simple health classification based on mean NDVI
                if mean_ndvi > 0.6:
                    health_status = "Healthy"
                elif mean_ndvi > 0.3:
                    health_status = "Moderate"
                else:
                    health_status = "Stressed"
                
                # Store processed patch
                patch_data = {
                    'patch_id': patch_id,
                    'images': images,                # Normalized Sentinel-2 data
                    'ndvi': ndvi,                    # NDVI time series
                    'evi': evi,                      # EVI time series
                    'parcel_ids': parcel_ids,        # Parcel boundaries
                    'crop_labels': crop_labels,      # Crop type labels
                    'n_timesteps': n_timesteps,
                    'n_bands': n_bands,
                    'shape': images.shape,
                    'mean_ndvi': float(mean_ndvi),
                    'mean_evi': float(mean_evi),
                    'health_status': health_status,
                    'file_path': str(img_file)
                }
                
                self.raw_patches.append(patch_data)
                
            except Exception as e:
                print(f"   ⚠️  Error loading {img_file.name}: {e}")
                continue
        
        print(f"\n   ✅ Loaded {len(self.raw_patches)} patches successfully")
        
        # Compute dataset statistics
        self.statistics = {
            'n_patches': len(self.raw_patches),
            'mean_ndvi_overall': float(np.mean(all_ndvi)),
            'std_ndvi_overall': float(np.std(all_ndvi)),
            'mean_evi_overall': float(np.mean(all_evi)),
            'std_evi_overall': float(np.std(all_evi)),
            'min_ndvi': float(np.min(all_ndvi)),
            'max_ndvi': float(np.max(all_ndvi)),
            'min_evi': float(np.min(all_evi)),
            'max_evi': float(np.max(evi)),
            'loaded_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        print(f"\n   📊 Dataset Statistics:")
        print(f"      Mean NDVI: {self.statistics['mean_ndvi_overall']:.3f} ± {self.statistics['std_ndvi_overall']:.3f}")
        print(f"      Mean EVI:  {self.statistics['mean_evi_overall']:.3f} ± {self.statistics['std_evi_overall']:.3f}")
        
        return True

    def export_phase1_outputs(self, sample_count=50):
        """
        Export Phase 1 artifacts that Phase 2 expects:
         - outputs/phase1/processed_data/metadata_summary.csv
         - outputs/phase1/processed_data/sample_patches/{patch_id}_*.npy
        """
        print("\n" + "="*80)
        print("STEP 2: EXPORTING PROCESSED DATA")
        print("="*80 + "\n")
        
        processed_dir = self.output_dir / 'processed_data'
        sample_dir = processed_dir / 'sample_patches'
        processed_dir.mkdir(parents=True, exist_ok=True)
        sample_dir.mkdir(parents=True, exist_ok=True)

        # Build metadata summary
        rows = []
        for p in self.raw_patches:
            row = {
                'patch_id': p.get('patch_id'),
                'file_path': p.get('file_path', ''),
                'n_timesteps': int(p.get('n_timesteps', 0)),
                'n_bands': int(p.get('n_bands', 0)),
                'mean_ndvi': float(p.get('mean_ndvi', 0.0)),
                'mean_evi': float(p.get('mean_evi', 0.0)),
                'health_status': p.get('health_status', 'Unknown')
            }
            rows.append(row)

        if len(rows) == 0:
            print("   ⚠️  No loaded patches to export.")
            return

        metadata_df = pd.DataFrame(rows)
        metadata_path = processed_dir / 'metadata_summary.csv'
        metadata_df.to_csv(metadata_path, index=False)
        print(f"   💾 Exported metadata summary: {metadata_path}")

        # Export statistics
        stats_path = processed_dir / 'dataset_statistics.json'
        with open(stats_path, 'w') as f:
            json.dump(self.statistics, f, indent=2)
        print(f"   💾 Exported statistics: {stats_path}")

        # Export sample patches
        sample_count = min(sample_count, len(self.raw_patches))
        print(f"\n   💾 Exporting {sample_count} sample patches...")
        
        for p in tqdm(self.raw_patches[:sample_count], desc="   Saving samples"):
            pid = str(p.get('patch_id'))
            
            # Save normalized images
            images = p.get('images')
            if images is not None:
                np.save(sample_dir / f"{pid}_images.npy", images)
            
            # Save NDVI
            ndvi = p.get('ndvi')
            if ndvi is not None:
                np.save(sample_dir / f"{pid}_ndvi.npy", ndvi)
            
            # Save EVI
            evi = p.get('evi')
            if evi is not None:
                np.save(sample_dir / f"{pid}_evi.npy", evi)
            
            # Save parcel IDs
            parcel_ids = p.get('parcel_ids')
            if parcel_ids is not None:
                np.save(sample_dir / f"{pid}_parcels.npy", parcel_ids)
            
            # Save crop labels
            crop_labels = p.get('crop_labels')
            if crop_labels is not None:
                np.save(sample_dir / f"{pid}_labels.npy", crop_labels)

        print(f"   ✅ Exported {sample_count} patches to: {sample_dir}")
        
        # Generate visualizations
        self.generate_visualizations(n_samples=min(5, len(self.raw_patches)))
        
        # Generate report
        self.generate_phase1_report()
        
        print("\n   ✅ Phase 1 Complete! Data ready for Phase 2.\n")

    def generate_visualizations(self, n_samples=5):
        """Generate visualizations of loaded data"""
        print(f"\n   📊 Generating visualizations...")
        
        vis_dir = self.output_dir / 'visualizations'
        
        try:
            # 1. NDVI distribution
            fig, ax = plt.subplots(figsize=(10, 6))
            ndvi_values = [p['mean_ndvi'] for p in self.raw_patches]
            ax.hist(ndvi_values, bins=30, edgecolor='black', alpha=0.7)
            ax.set_xlabel('Mean NDVI')
            ax.set_ylabel('Frequency')
            ax.set_title('Distribution of Mean NDVI Across Patches')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(vis_dir / 'ndvi_distribution.png', dpi=150, bbox_inches='tight')
            plt.close()
            
            # 2. Sample NDVI time series
            fig, axes = plt.subplots(n_samples, 1, figsize=(12, 3*n_samples))
            if n_samples == 1:
                axes = [axes]
            
            for i, patch in enumerate(self.raw_patches[:n_samples]):
                ndvi = patch['ndvi']  # Shape: (T, H, W)
                # Compute spatial mean for each timestep
                ndvi_temporal = np.nanmean(ndvi, axis=(1, 2))
                axes[i].plot(ndvi_temporal, marker='o', linewidth=2)
                axes[i].set_ylabel('Mean NDVI')
                axes[i].set_title(f"Patch {patch['patch_id']} - NDVI Time Series")
                axes[i].grid(True, alpha=0.3)
                axes[i].set_ylim(-0.2, 1.0)
            
            axes[-1].set_xlabel('Timestep')
            plt.tight_layout()
            plt.savefig(vis_dir / 'ndvi_timeseries_samples.png', dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"      ✅ Saved visualizations to {vis_dir}")
            
        except Exception as e:
            print(f"      ⚠️  Error generating visualizations: {e}")

    def generate_phase1_report(self):
        """Generate comprehensive Phase 1 report"""
        report_path = self.output_dir / 'phase1_report.txt'
        
        report = f"""
{'='*80}
PHASE 1: DATA ACQUISITION & PREPROCESSING - REPORT
{'='*80}

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

DATASET SUMMARY:
----------------
Total Patches Loaded: {self.statistics['n_patches']}
Data Source: REAL PASTIS Dataset
Data Directory: {self.data_dir}

VEGETATION INDEX STATISTICS:
----------------------------
NDVI:
  Mean: {self.statistics['mean_ndvi_overall']:.4f}
  Std:  {self.statistics['std_ndvi_overall']:.4f}
  Range: [{self.statistics['min_ndvi']:.4f}, {self.statistics['max_ndvi']:.4f}]

EVI:
  Mean: {self.statistics['mean_evi_overall']:.4f}
  Std:  {self.statistics['std_evi_overall']:.4f}
  Range: [{self.statistics['min_evi']:.4f}, {self.statistics['max_evi']:.4f}]

HEALTH STATUS DISTRIBUTION:
---------------------------
"""
        
        # Add health status counts
        health_counts = {}
        for p in self.raw_patches:
            status = p['health_status']
            health_counts[status] = health_counts.get(status, 0) + 1
        
        for status, count in health_counts.items():
            report += f"{status}: {count} patches ({100*count/len(self.raw_patches):.1f}%)\n"
        
        report += f"""
DELIVERABLES:
-------------
✅ Cleaned and normalized Sentinel-2 image time-series
✅ NDVI computed for all patches and timesteps
✅ EVI computed for all patches and timesteps
✅ Parcel boundaries extracted
✅ Metadata summary exported
✅ Sample patches saved for Phase 2

OUTPUT LOCATIONS:
-----------------
Metadata: {self.output_dir / 'processed_data' / 'metadata_summary.csv'}
Statistics: {self.output_dir / 'processed_data' / 'dataset_statistics.json'}
Samples: {self.output_dir / 'processed_data' / 'sample_patches'}
Visualizations: {self.output_dir / 'visualizations'}

NEXT STEPS:
-----------
Phase 1 is complete. The prepared dataset is ready for Phase 2:
- Segmentation & Feature Extraction
- Temporal pattern analysis
- GLCM texture feature computation

Run Phase 2 with: python src/phase2_segmentation.py

{'='*80}
"""
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\n   📄 Phase 1 report saved to: {report_path}")


def main():
    """Main execution function for Phase 1"""
    print("\n" + "🌾"*40)
    print("CROP HEALTH MONITORING FROM REMOTE SENSING")
    print("Phase 1: Data Acquisition & Preprocessing")
    print("🌾"*40 + "\n")
    
    # Initialize processor
    processor = PASTISDatasetProcessor(
        data_dir="./data/PASTIS",
        output_dir="./outputs/phase1"
    )
    
    # Load real data (no synthetic generation)
    processor.load_or_generate_dataset(n_patches=100)
    
    # Export processed data for Phase 2
    processor.export_phase1_outputs(sample_count=50)
    
    print("\n" + "="*80)
    print("✅ PHASE 1 COMPLETE!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
