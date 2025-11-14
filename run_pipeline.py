"""Run the full pipeline (Phase1 -> Phase2 -> Phase3 -> Phase4) using real PASTIS data.
Usage:
    python run_pipeline.py --n_patches 100
"""
import sys
from pathlib import Path
import argparse

# Ensure src/ is on path
ROOT = Path(__file__).resolve().parent
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from phase1_preprocessing import PASTISDatasetProcessor
from phase2_segmentation import VegetationIndexProcessor
from phase3_patterndiscovery_and_anomalydetection import PatternDiscoveryEngine

# Import Phase 4 (handle special character in module name)
import importlib.util
phase4_spec = importlib.util.spec_from_file_location("phase4", str(SRC / "phase4_predictivemodeling&evaluation.py"))
phase4_module = importlib.util.module_from_spec(phase4_spec)
phase4_spec.loader.exec_module(phase4_module)
PredictiveModelingEngine = phase4_module.PredictiveModelingEngine


def main(n_patches, sample_count):
    print('\n=== RUNNING PIPELINE ===\n')

    # Phase 1
    p1 = PASTISDatasetProcessor(data_dir=str(ROOT / 'data' / 'PASTIS'), output_dir=str(ROOT / 'outputs' / 'phase1'))
    p1.load_or_generate_dataset(n_patches=n_patches)
    p1.export_phase1_outputs(sample_count=sample_count)

    # Phase 2
    p2 = VegetationIndexProcessor(input_dir=str(ROOT / 'outputs' / 'phase1' / 'processed_data'), output_dir=str(ROOT / 'outputs' / 'phase2'))
    p2.load_phase1_data()
    p2.compute_vegetation_indices()
    p2.perform_image_segmentation()
    p2.extract_features()
    p2.analyze_temporal_patterns()

    # Phase 3
    p3 = PatternDiscoveryEngine(input_dir=str(ROOT / 'outputs' / 'phase2'), output_dir=str(ROOT / 'outputs' / 'phase3'))
    if not p3.load_phase2_features():
        print('Phase 3: failed to load Phase 2 features. Exiting.')
        return
    p3.preprocess_features()
    p3.perform_clustering_analysis()
    p3.perform_anomaly_detection()
    p3.discover_pattern_rules()
    p3.generate_early_warnings()
    p3.create_visualizations()
    p3.generate_phase3_report()

    # Phase 4
    p4 = PredictiveModelingEngine(input_dir=str(ROOT / 'outputs' / 'phase3'), output_dir=str(ROOT / 'outputs' / 'phase4'))
    if not p4.load_and_prepare_data():
        print('Phase 4: failed to load Phase 3 data. Exiting.')
        return
    p4.prepare_train_test_split()
    p4.train_regression_models()
    p4.evaluate_regression_models()

    print('\n=== PIPELINE COMPLETE ===\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_patches', type=int, default=100, help='Number of patches to process')
    parser.add_argument('--sample_count', type=int, default=50, help='Number of sample patches to export for Phase 2')
    args = parser.parse_args()
    main(args.n_patches, args.sample_count)
