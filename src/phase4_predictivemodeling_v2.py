#!/usr/bin/env python3
"""
================================================================================
CSCE5380 - Crop Health Monitoring from Remote Sensing
PHASE 4: Predictive Modeling & Evaluation
================================================================================

This script implements:
1. Random Forest baseline model (yield prediction)
2. XGBoost gradient boosting model (yield prediction)
3. LSTM temporal model (stress classification)
4. Comprehensive model evaluation

Authors: Teja Sai Srinivas Kunisetty, Lahithya Reddy Varri
Date: November 12, 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Tuple, Any
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Machine Learning Libraries
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# Deep Learning Libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# Suppress TensorFlow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


class PredictiveModelingEngine:
    """Phase 4: Predictive Modeling & Evaluation"""
    
    def __init__(self, phase2_dir: str, phase3_dir: str, output_dir: str):
        self.phase2_dir = Path(phase2_dir)
        self.phase3_dir = Path(phase3_dir)
        self.output_dir = Path(output_dir)
        
        # Create output directories
        self.models_dir = self.output_dir / "models"
        self.predictions_dir = self.output_dir / "predictions"
        self.evaluation_dir = self.output_dir / "evaluation"
        self.visualizations_dir = self.output_dir / "visualizations"
        self.reports_dir = self.output_dir / "reports"
        
        for dir_path in [self.models_dir, self.predictions_dir, 
                         self.evaluation_dir, self.visualizations_dir, self.reports_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Model storage
        self.models = {}
        self.scalers = {}
        self.results = {}
        
        print("\n✅ Predictive Modeling Engine initialized")
        print(f"   Phase 2 features from: {self.phase2_dir}")
        print(f"   Phase 3 clusters from: {self.phase3_dir}")
        print(f"   Output directory: {self.output_dir}")
    
    def load_data(self) -> None:
        """Load features from Phase 2 and clusters from Phase 3"""
        print("\n" + "="*80)
        print("STEP 1: LOADING DATA FROM PHASES 2 & 3")
        print("="*80)
        
        print("\n📥 Loading feature files...")
        
        # Load Phase 2 features
        temporal_path = self.phase2_dir / "features" / "temporal_features.csv"
        aggregated_path = self.phase2_dir / "features" / "aggregated_features.csv"
        spatial_path = self.phase2_dir / "features" / "spatial_features.csv"
        
        self.temporal_features = pd.read_csv(temporal_path)
        self.aggregated_features = pd.read_csv(aggregated_path)
        self.spatial_features = pd.read_csv(spatial_path)
        
        print(f"   ✅ Temporal features: {len(self.temporal_features):,} rows")
        print(f"   ✅ Aggregated features: {len(self.aggregated_features):,} parcels")
        print(f"   ✅ Spatial features: {len(self.spatial_features):,} parcels")
        
        # Load Phase 3 results
        clusters_path = self.phase3_dir / "clusters" / "cluster_assignments.csv"
        anomalies_path = self.phase3_dir / "anomalies" / "anomaly_scores.csv"
        
        self.clusters = pd.read_csv(clusters_path)
        self.anomalies = pd.read_csv(anomalies_path)
        
        print(f"   ✅ Cluster assignments: {len(self.clusters):,} parcels")
        print(f"   ✅ Anomaly scores: {len(self.anomalies):,} parcels")
        
        print("\n✅ Data loaded successfully")
    
    def prepare_datasets(self) -> None:
        """Prepare training datasets for regression and classification"""
        print("\n" + "="*80)
        print("STEP 2: PREPARING DATASETS")
        print("="*80)
        
        # Merge all features
        print("\n🔗 Merging features from all sources...")
        
        self.master_df = self.aggregated_features.merge(
            self.spatial_features, on='Parcel_ID', how='left'
        ).merge(
            self.clusters[['Parcel_ID', 'Cluster']], on='Parcel_ID', how='left'
        ).merge(
            self.anomalies[['Parcel_ID', 'Anomaly_Score', 'Is_Anomaly']], 
            on='Parcel_ID', how='left'
        )
        
        print(f"   ✅ Master dataset: {len(self.master_df):,} parcels")
        print(f"   ✅ Total features: {len(self.master_df.columns)} columns")
        
        # ==================================================
        # Dataset 1: Yield Prediction (Regression)
        # ==================================================
        print("\n📊 Dataset 1: Yield Prediction (Regression)")
        print("   Target: NDVI_Peak_Value (proxy for crop yield)")
        print("   ⚠️  Excluding NDVI_Range and EVI_Range to prevent data leakage")
        print("      (NDVI_Range = NDVI_Max - NDVI_Min, where NDVI_Max == NDVI_Peak_Value)")
        
        # Features for regression
        # NOTE: Removed NDVI_Range and EVI_Range due to data leakage
        # NDVI_Range contains NDVI_Max which equals NDVI_Peak_Value (the target!)
        feature_cols = [
            'NDVI_Mean', 'NDVI_Std',  # ✅ Removed NDVI_Range
            'EVI_Mean', 'EVI_Std',    # ✅ Removed EVI_Range
            'GLCM_Contrast', 'GLCM_Dissimilarity', 'GLCM_Homogeneity',
            'GLCM_Energy', 'GLCM_Correlation', 'GLCM_ASM',
            'Cluster', 'Anomaly_Score'
        ]
        
        # Check which features exist
        available_features = [col for col in feature_cols if col in self.master_df.columns]
        
        self.X_regression = self.master_df[available_features].fillna(0)
        self.y_regression = self.master_df['NDVI_Peak_Value']
        
        print(f"   Features: {len(available_features)} (was 14, now 12 after removing leakage)")
        print(f"   Samples: {len(self.X_regression):,}")
        print(f"   Target range: {self.y_regression.min():.4f} to {self.y_regression.max():.4f}")
        
        # ==================================================
        # Dataset 2: Stress Classification
        # ==================================================
        print("\n📊 Dataset 2: Stress Classification (Binary)")
        print("   Target: Is_Anomaly (stressed vs. healthy)")
        
        # IMPORTANT: Remove Anomaly_Score to prevent data leakage!
        # Anomaly_Score is directly used to determine Is_Anomaly, so using it
        # as a feature would be cheating (data leakage)
        classification_features = [f for f in available_features if f != 'Anomaly_Score']
        
        self.X_classification = self.master_df[classification_features].fillna(0)
        self.y_classification = self.master_df['Is_Anomaly'].astype(int)
        
        print(f"   Features: {len(classification_features)} (excluded Anomaly_Score to prevent leakage)")
        print(f"   Samples: {len(self.X_classification):,}")
        print(f"   Stress class distribution:")
        print(f"     Healthy (0): {(self.y_classification == 0).sum():,} ({(self.y_classification == 0).mean()*100:.2f}%)")
        print(f"     Stressed (1): {(self.y_classification == 1).sum():,} ({(self.y_classification == 1).mean()*100:.2f}%)")
        
        # ==================================================
        # Dataset 3: LSTM Time-Series (Stress Classification)
        # ==================================================
        print("\n📊 Dataset 3: LSTM Time-Series (Stress Classification)")
        print("   Preparing sequential data...")
        
        # Pivot temporal features to time-series format
        ndvi_pivot = self.temporal_features.pivot_table(
            index='Parcel_ID',
            columns='Timestep',
            values='Mean_NDVI'
        ).fillna(0)
        
        evi_pivot = self.temporal_features.pivot_table(
            index='Parcel_ID',
            columns='Timestep',
            values='Mean_EVI'
        ).fillna(0)
        
        # Combine NDVI and EVI as two channels
        self.X_lstm = np.stack([
            ndvi_pivot.values,
            evi_pivot.values
        ], axis=-1)  # Shape: (n_parcels, n_timesteps, 2)
        
        # Use same classification target
        self.y_lstm = self.master_df.set_index('Parcel_ID').loc[ndvi_pivot.index]['Is_Anomaly'].astype(int).values
        
        print(f"   Time-series shape: {self.X_lstm.shape}")
        print(f"   (Parcels, Timesteps, Features) = ({self.X_lstm.shape[0]}, {self.X_lstm.shape[1]}, {self.X_lstm.shape[2]})")
        print(f"   Target: {len(self.y_lstm)} labels")
        
        print("\n✅ Datasets prepared successfully")
    
    def train_random_forest_regression(self) -> None:
        """Train Random Forest for yield prediction"""
        print("\n" + "="*80)
        print("STEP 3: TRAINING RANDOM FOREST (YIELD PREDICTION)")
        print("="*80)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            self.X_regression, self.y_regression, 
            test_size=0.2, random_state=42
        )
        
        print(f"\n📊 Train set: {len(X_train):,} samples")
        print(f"   Test set:  {len(X_test):,} samples")
        
        # Standardize features
        print("\n🔄 Standardizing features...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers['rf_regression'] = scaler
        
        # Train model
        print("\n🎯 Training Random Forest Regressor...")
        print("   Parameters: n_estimators=100, max_depth=20, min_samples_split=10")
        
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=20,
            min_samples_split=10,
            min_samples_leaf=4,
            random_state=42,
            n_jobs=-1
        )
        
        rf_model.fit(X_train_scaled, y_train)
        self.models['rf_regression'] = rf_model
        
        # Predictions
        y_train_pred = rf_model.predict(X_train_scaled)
        y_test_pred = rf_model.predict(X_test_scaled)
        
        # Evaluate
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        
        print("\n✅ Random Forest training complete")
        print(f"\n   Train Metrics:")
        print(f"     RMSE: {train_rmse:.4f}")
        print(f"     R²:   {train_r2:.4f}")
        print(f"\n   Test Metrics:")
        print(f"     RMSE: {test_rmse:.4f}")
        print(f"     MAE:  {test_mae:.4f}")
        print(f"     R²:   {test_r2:.4f}")
        
        # Store results
        self.results['rf_regression'] = {
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'test_mae': test_mae,
            'y_test': y_test,
            'y_pred': y_test_pred,
            'feature_importance': dict(zip(self.X_regression.columns, rf_model.feature_importances_))
        }
        
        # Save predictions
        predictions_df = pd.DataFrame({
            'True_Yield': y_test,
            'Predicted_Yield': y_test_pred,
            'Error': y_test - y_test_pred
        })
        predictions_df.to_csv(self.predictions_dir / "rf_yield_predictions.csv", index=False)
        print(f"\n   💾 Saved predictions to: rf_yield_predictions.csv")
    
    def train_xgboost_regression(self) -> None:
        """Train XGBoost for yield prediction"""
        print("\n" + "="*80)
        print("STEP 4: TRAINING XGBOOST (YIELD PREDICTION)")
        print("="*80)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            self.X_regression, self.y_regression, 
            test_size=0.2, random_state=42
        )
        
        # Standardize features
        print("\n🔄 Standardizing features...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers['xgb_regression'] = scaler
        
        # Train model
        print("\n🎯 Training XGBoost Regressor...")
        print("   Parameters: n_estimators=100, max_depth=10, learning_rate=0.1")
        
        xgb_model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=10,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        
        xgb_model.fit(X_train_scaled, y_train, verbose=False)
        self.models['xgb_regression'] = xgb_model
        
        # Predictions
        y_train_pred = xgb_model.predict(X_train_scaled)
        y_test_pred = xgb_model.predict(X_test_scaled)
        
        # Evaluate
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        
        print("\n✅ XGBoost training complete")
        print(f"\n   Train Metrics:")
        print(f"     RMSE: {train_rmse:.4f}")
        print(f"     R²:   {train_r2:.4f}")
        print(f"\n   Test Metrics:")
        print(f"     RMSE: {test_rmse:.4f}")
        print(f"     MAE:  {test_mae:.4f}")
        print(f"     R²:   {test_r2:.4f}")
        
        # Store results
        self.results['xgb_regression'] = {
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'test_mae': test_mae,
            'y_test': y_test,
            'y_pred': y_test_pred,
            'feature_importance': dict(zip(self.X_regression.columns, xgb_model.feature_importances_))
        }
        
        # Save predictions
        predictions_df = pd.DataFrame({
            'True_Yield': y_test,
            'Predicted_Yield': y_test_pred,
            'Error': y_test - y_test_pred
        })
        predictions_df.to_csv(self.predictions_dir / "xgb_yield_predictions.csv", index=False)
        print(f"\n   💾 Saved predictions to: xgb_yield_predictions.csv")
    
    def train_random_forest_classification(self) -> None:
        """Train Random Forest for stress classification"""
        print("\n" + "="*80)
        print("STEP 5: TRAINING RANDOM FOREST (STRESS CLASSIFICATION)")
        print("="*80)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            self.X_classification, self.y_classification, 
            test_size=0.2, random_state=42, stratify=self.y_classification
        )
        
        print(f"\n📊 Train set: {len(X_train):,} samples")
        print(f"   Test set:  {len(X_test):,} samples")
        
        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers['rf_classification'] = scaler
        
        # Train model
        print("\n🎯 Training Random Forest Classifier...")
        
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=10,
            class_weight='balanced',  # Handle class imbalance
            random_state=42,
            n_jobs=-1
        )
        
        rf_model.fit(X_train_scaled, y_train)
        self.models['rf_classification'] = rf_model
        
        # Predictions
        y_pred = rf_model.predict(X_test_scaled)
        y_pred_proba = rf_model.predict_proba(X_test_scaled)[:, 1]
        
        # Evaluate
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        try:
            auc = roc_auc_score(y_test, y_pred_proba)
        except:
            auc = 0.0
        
        print("\n✅ Random Forest classification complete")
        print(f"\n   Test Metrics:")
        print(f"     Accuracy:  {accuracy:.4f}")
        print(f"     Precision: {precision:.4f}")
        print(f"     Recall:    {recall:.4f}")
        print(f"     F1-Score:  {f1:.4f}")
        print(f"     ROC-AUC:   {auc:.4f}")
        
        # Store results
        self.results['rf_classification'] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': auc,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        # Save predictions
        predictions_df = pd.DataFrame({
            'True_Label': y_test,
            'Predicted_Label': y_pred,
            'Stress_Probability': y_pred_proba
        })
        predictions_df.to_csv(self.predictions_dir / "rf_stress_predictions.csv", index=False)
        print(f"\n   💾 Saved predictions to: rf_stress_predictions.csv")
    
    def train_lstm_classification(self) -> None:
        """Train LSTM for stress classification"""
        print("\n" + "="*80)
        print("STEP 6: TRAINING LSTM (STRESS CLASSIFICATION)")
        print("="*80)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            self.X_lstm, self.y_lstm, 
            test_size=0.2, random_state=42, stratify=self.y_lstm
        )
        
        print(f"\n📊 Train set: {X_train.shape[0]:,} sequences")
        print(f"   Test set:  {X_test.shape[0]:,} sequences")
        print(f"   Sequence shape: ({X_train.shape[1]} timesteps, {X_train.shape[2]} features)")
        
        # Show class distribution
        print(f"\n⚖️  Class distribution in training data:")
        print(f"     Healthy (0):  {(y_train == 0).sum():,} ({(y_train == 0).mean()*100:.2f}%)")
        print(f"     Stressed (1): {(y_train == 1).sum():,} ({(y_train == 1).mean()*100:.2f}%)")
        print(f"   ⚠️  Severe class imbalance - will use class weights!")
        
        # Build LSTM model
        print("\n🏗️ Building LSTM architecture...")
        
        model = Sequential([
            Bidirectional(LSTM(64, return_sequences=True), 
                         input_shape=(X_train.shape[1], X_train.shape[2])),
            Dropout(0.3),
            Bidirectional(LSTM(32)),
            Dropout(0.3),
            Dense(16, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
        
        print("\n   Model Architecture:")
        model.summary()
        
        # Callbacks
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
        
        # Calculate class weights to handle imbalance
        from sklearn.utils.class_weight import compute_class_weight
        
        class_weights_array = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weights = {i: class_weights_array[i] for i in range(len(class_weights_array))}
        
        print(f"\n⚖️  Class weights (to handle imbalance):")
        print(f"   Class 0 (Healthy):  {class_weights[0]:.4f}")
        print(f"   Class 1 (Stressed): {class_weights[1]:.4f}")
        
        # Train model
        print("\n🎯 Training LSTM model...")
        
        history = model.fit(
            X_train, y_train,
            validation_split=0.2,
            epochs=50,
            batch_size=32,
            class_weight=class_weights,  # ✅ Added to handle class imbalance
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )
        
        self.models['lstm_classification'] = model
        
        # Predictions
        y_pred_proba = model.predict(X_test).flatten()
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Evaluate
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        try:
            auc = roc_auc_score(y_test, y_pred_proba)
        except:
            auc = 0.0
        
        print("\n✅ LSTM training complete")
        print(f"\n   Test Metrics:")
        print(f"     Accuracy:  {accuracy:.4f}")
        print(f"     Precision: {precision:.4f}")
        print(f"     Recall:    {recall:.4f}")
        print(f"     F1-Score:  {f1:.4f}")
        print(f"     ROC-AUC:   {auc:.4f}")
        
        # Store results
        self.results['lstm_classification'] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': auc,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'history': history.history
        }
        
        # Save predictions
        predictions_df = pd.DataFrame({
            'True_Label': y_test,
            'Predicted_Label': y_pred,
            'Stress_Probability': y_pred_proba
        })
        predictions_df.to_csv(self.predictions_dir / "lstm_stress_predictions.csv", index=False)
        print(f"\n   💾 Saved predictions to: lstm_stress_predictions.csv")
        
        # Save model
        model.save(self.models_dir / "lstm_stress_model.keras")
        print(f"   💾 Saved model to: lstm_stress_model.keras")
    
    def visualize_results(self) -> None:
        """Create comprehensive visualizations"""
        print("\n" + "="*80)
        print("STEP 7: CREATING VISUALIZATIONS")
        print("="*80)
        
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        # ========================================
        # 1. Regression Results
        # ========================================
        print("\n📊 Visualizing regression results...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # RF Predictions
        rf_results = self.results['rf_regression']
        axes[0, 0].scatter(rf_results['y_test'], rf_results['y_pred'], alpha=0.5)
        axes[0, 0].plot([rf_results['y_test'].min(), rf_results['y_test'].max()],
                        [rf_results['y_test'].min(), rf_results['y_test'].max()],
                        'r--', lw=2)
        axes[0, 0].set_xlabel('True Yield (Peak NDVI)')
        axes[0, 0].set_ylabel('Predicted Yield')
        axes[0, 0].set_title(f"Random Forest\nRMSE: {rf_results['test_rmse']:.4f}, R²: {rf_results['test_r2']:.4f}")
        axes[0, 0].grid(True, alpha=0.3)
        
        # XGB Predictions
        xgb_results = self.results['xgb_regression']
        axes[0, 1].scatter(xgb_results['y_test'], xgb_results['y_pred'], alpha=0.5, color='green')
        axes[0, 1].plot([xgb_results['y_test'].min(), xgb_results['y_test'].max()],
                        [xgb_results['y_test'].min(), xgb_results['y_test'].max()],
                        'r--', lw=2)
        axes[0, 1].set_xlabel('True Yield (Peak NDVI)')
        axes[0, 1].set_ylabel('Predicted Yield')
        axes[0, 1].set_title(f"XGBoost\nRMSE: {xgb_results['test_rmse']:.4f}, R²: {xgb_results['test_r2']:.4f}")
        axes[0, 1].grid(True, alpha=0.3)
        
        # Feature Importance - RF
        rf_importance = sorted(rf_results['feature_importance'].items(), 
                              key=lambda x: x[1], reverse=True)[:10]
        features, importances = zip(*rf_importance)
        axes[1, 0].barh(features, importances)
        axes[1, 0].set_xlabel('Feature Importance')
        axes[1, 0].set_title('Random Forest - Top 10 Features')
        axes[1, 0].invert_yaxis()
        
        # Feature Importance - XGB
        xgb_importance = sorted(xgb_results['feature_importance'].items(), 
                               key=lambda x: x[1], reverse=True)[:10]
        features, importances = zip(*xgb_importance)
        axes[1, 1].barh(features, importances, color='green')
        axes[1, 1].set_xlabel('Feature Importance')
        axes[1, 1].set_title('XGBoost - Top 10 Features')
        axes[1, 1].invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(self.visualizations_dir / "regression_results.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # ========================================
        # 2. Classification Results
        # ========================================
        print("📊 Visualizing classification results...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # RF Confusion Matrix
        rf_clf = self.results['rf_classification']
        cm_rf = confusion_matrix(rf_clf['y_test'], rf_clf['y_pred'])
        sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
        axes[0, 0].set_xlabel('Predicted')
        axes[0, 0].set_ylabel('True')
        axes[0, 0].set_title(f"Random Forest\nF1: {rf_clf['f1_score']:.4f}")
        
        # LSTM Confusion Matrix
        lstm_clf = self.results['lstm_classification']
        cm_lstm = confusion_matrix(lstm_clf['y_test'], lstm_clf['y_pred'])
        sns.heatmap(cm_lstm, annot=True, fmt='d', cmap='Greens', ax=axes[0, 1])
        axes[0, 1].set_xlabel('Predicted')
        axes[0, 1].set_ylabel('True')
        axes[0, 1].set_title(f"LSTM\nF1: {lstm_clf['f1_score']:.4f}")
        
        # RF ROC Curve
        if rf_clf['roc_auc'] > 0:
            fpr, tpr, _ = roc_curve(rf_clf['y_test'], rf_clf['y_pred_proba'])
            axes[1, 0].plot(fpr, tpr, label=f"AUC = {rf_clf['roc_auc']:.4f}")
            axes[1, 0].plot([0, 1], [0, 1], 'k--')
            axes[1, 0].set_xlabel('False Positive Rate')
            axes[1, 0].set_ylabel('True Positive Rate')
            axes[1, 0].set_title('Random Forest - ROC Curve')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # LSTM ROC Curve
        if lstm_clf['roc_auc'] > 0:
            fpr, tpr, _ = roc_curve(lstm_clf['y_test'], lstm_clf['y_pred_proba'])
            axes[1, 1].plot(fpr, tpr, label=f"AUC = {lstm_clf['roc_auc']:.4f}", color='green')
            axes[1, 1].plot([0, 1], [0, 1], 'k--')
            axes[1, 1].set_xlabel('False Positive Rate')
            axes[1, 1].set_ylabel('True Positive Rate')
            axes[1, 1].set_title('LSTM - ROC Curve')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.visualizations_dir / "classification_results.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # ========================================
        # 3. LSTM Training History
        # ========================================
        if 'lstm_classification' in self.results:
            print("📊 Visualizing LSTM training history...")
            
            history = self.results['lstm_classification']['history']
            
            fig, axes = plt.subplots(1, 2, figsize=(15, 5))
            
            # Loss
            axes[0].plot(history['loss'], label='Train Loss')
            axes[0].plot(history['val_loss'], label='Val Loss')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Loss')
            axes[0].set_title('LSTM Training Loss')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # Accuracy
            axes[1].plot(history['accuracy'], label='Train Accuracy')
            axes[1].plot(history['val_accuracy'], label='Val Accuracy')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Accuracy')
            axes[1].set_title('LSTM Training Accuracy')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.visualizations_dir / "lstm_training_history.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # ========================================
        # 4. Model Comparison Chart
        # ========================================
        print("📊 Creating model comparison chart...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Classification Accuracy Comparison
        models = ['Random Forest', 'LSTM']
        accuracy_scores = [
            self.results['rf_classification']['accuracy'],
            self.results['lstm_classification']['accuracy']
        ]
        precision_scores = [
            self.results['rf_classification']['precision'],
            self.results['lstm_classification']['precision']
        ]
        recall_scores = [
            self.results['rf_classification']['recall'],
            self.results['lstm_classification']['recall']
        ]
        f1_scores = [
            self.results['rf_classification']['f1_score'],
            self.results['lstm_classification']['f1_score']
        ]
        
        x = np.arange(len(models))
        width = 0.2
        
        axes[0, 0].bar(x - 1.5*width, accuracy_scores, width, label='Accuracy', color='#2ecc71')
        axes[0, 0].bar(x - 0.5*width, precision_scores, width, label='Precision', color='#3498db')
        axes[0, 0].bar(x + 0.5*width, recall_scores, width, label='Recall', color='#e74c3c')
        axes[0, 0].bar(x + 1.5*width, f1_scores, width, label='F1-Score', color='#f39c12')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_title('Classification Model Comparison', fontsize=14, fontweight='bold')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(models)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        axes[0, 0].set_ylim([0, 1.05])
        
        # Add value labels on bars
        for i, (acc, prec, rec, f1) in enumerate(zip(accuracy_scores, precision_scores, recall_scores, f1_scores)):
            axes[0, 0].text(i - 1.5*width, acc + 0.02, f'{acc:.3f}', ha='center', va='bottom', fontsize=8)
            axes[0, 0].text(i - 0.5*width, prec + 0.02, f'{prec:.3f}', ha='center', va='bottom', fontsize=8)
            axes[0, 0].text(i + 0.5*width, rec + 0.02, f'{rec:.3f}', ha='center', va='bottom', fontsize=8)
            axes[0, 0].text(i + 1.5*width, f1 + 0.02, f'{f1:.3f}', ha='center', va='bottom', fontsize=8)
        
        # Regression Model Comparison
        reg_models = ['Random Forest', 'XGBoost']
        r2_scores = [
            self.results['rf_regression']['test_r2'],
            self.results['xgb_regression']['test_r2']
        ]
        rmse_scores = [
            self.results['rf_regression']['test_rmse'],
            self.results['xgb_regression']['test_rmse']
        ]
        
        x_reg = np.arange(len(reg_models))
        axes[0, 1].bar(x_reg - width/2, r2_scores, width, label='R² Score', color='#9b59b6')
        axes[0, 1].set_ylabel('R² Score')
        axes[0, 1].set_title('Regression Model R² Comparison', fontsize=14, fontweight='bold')
        axes[0, 1].set_xticks(x_reg)
        axes[0, 1].set_xticklabels(reg_models)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        axes[0, 1].set_ylim([0, 1.05])
        
        # Add value labels
        for i, r2 in enumerate(r2_scores):
            axes[0, 1].text(i - width/2, r2 + 0.02, f'{r2:.3f}', ha='center', va='bottom', fontsize=10)
        
        # Cross-Validation Scores (Classification)
        print("📊 Performing cross-validation...")
        from sklearn.model_selection import cross_validate
        
        # Get the trained models and data
        rf_clf_model = self.models['rf_classification']
        X_scaled = self.scalers['rf_classification'].transform(self.X_classification)
        
        cv_results = cross_validate(
            rf_clf_model, X_scaled, self.y_classification,
            cv=5,
            scoring=['accuracy', 'precision', 'recall', 'f1'],
            return_train_score=True
        )
        
        cv_metrics = ['accuracy', 'precision', 'recall', 'f1']
        train_scores = [cv_results[f'train_{m}'].mean() for m in cv_metrics]
        test_scores = [cv_results[f'test_{m}'].mean() for m in cv_metrics]
        
        x_cv = np.arange(len(cv_metrics))
        axes[1, 0].bar(x_cv - width/2, train_scores, width, label='Train Score', color='#3498db')
        axes[1, 0].bar(x_cv + width/2, test_scores, width, label='Test Score (CV)', color='#e74c3c')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].set_title('Cross-Validation Results (Random Forest)', fontsize=14, fontweight='bold')
        axes[1, 0].set_xticks(x_cv)
        axes[1, 0].set_xticklabels([m.capitalize() for m in cv_metrics])
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        axes[1, 0].set_ylim([0, 1.05])
        
        # Add value labels
        for i, (train, test) in enumerate(zip(train_scores, test_scores)):
            axes[1, 0].text(i - width/2, train + 0.02, f'{train:.3f}', ha='center', va='bottom', fontsize=8)
            axes[1, 0].text(i + width/2, test + 0.02, f'{test:.3f}', ha='center', va='bottom', fontsize=8)
        
        # ROC-AUC Comparison
        roc_auc_scores = [
            self.results['rf_classification']['roc_auc'],
            self.results['lstm_classification']['roc_auc']
        ]
        
        axes[1, 1].bar(models, roc_auc_scores, color=['#2ecc71', '#3498db'], width=0.5)
        axes[1, 1].set_ylabel('ROC-AUC Score')
        axes[1, 1].set_title('Classification ROC-AUC Comparison', fontsize=14, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        axes[1, 1].set_ylim([0, 1.05])
        
        # Add value labels
        for i, score in enumerate(roc_auc_scores):
            axes[1, 1].text(i, score + 0.02, f'{score:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(self.visualizations_dir / "model_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # ========================================
        # 5. Feature Importance for Classification
        # ========================================
        print("📊 Creating feature importance charts...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Random Forest Classification Feature Importance
        rf_clf_model = self.models['rf_classification']
        feature_names_clf = self.X_classification.columns
        importances = rf_clf_model.feature_importances_
        indices = np.argsort(importances)[::-1][:15]  # Top 15 features
        
        axes[0, 0].barh(range(len(indices)), importances[indices], color='#2ecc71')
        axes[0, 0].set_yticks(range(len(indices)))
        axes[0, 0].set_yticklabels([feature_names_clf[i] for i in indices])
        axes[0, 0].set_xlabel('Feature Importance')
        axes[0, 0].set_title('Random Forest Classification - Top 15 Features', fontsize=12, fontweight='bold')
        axes[0, 0].invert_yaxis()
        axes[0, 0].grid(True, alpha=0.3, axis='x')
        
        # Random Forest Regression Feature Importance
        rf_reg_model = self.models['rf_regression']
        feature_names_reg = self.X_regression.columns
        importances_reg = rf_reg_model.feature_importances_
        indices_reg = np.argsort(importances_reg)[::-1][:15]
        
        axes[0, 1].barh(range(len(indices_reg)), importances_reg[indices_reg], color='#3498db')
        axes[0, 1].set_yticks(range(len(indices_reg)))
        axes[0, 1].set_yticklabels([feature_names_reg[i] for i in indices_reg])
        axes[0, 1].set_xlabel('Feature Importance')
        axes[0, 1].set_title('Random Forest Regression - Top 15 Features', fontsize=12, fontweight='bold')
        axes[0, 1].invert_yaxis()
        axes[0, 1].grid(True, alpha=0.3, axis='x')
        
        # XGBoost Regression Feature Importance
        xgb_model = self.models['xgb_regression']
        importances_xgb = xgb_model.feature_importances_
        indices_xgb = np.argsort(importances_xgb)[::-1][:15]
        
        axes[1, 0].barh(range(len(indices_xgb)), importances_xgb[indices_xgb], color='#9b59b6')
        axes[1, 0].set_yticks(range(len(indices_xgb)))
        axes[1, 0].set_yticklabels([feature_names_reg[i] for i in indices_xgb])
        axes[1, 0].set_xlabel('Feature Importance')
        axes[1, 0].set_title('XGBoost Regression - Top 15 Features', fontsize=12, fontweight='bold')
        axes[1, 0].invert_yaxis()
        axes[1, 0].grid(True, alpha=0.3, axis='x')
        
        # Feature Importance Comparison (Top 10 across all models)
        # Average importance across classification and regression models
        all_importances = {}
        
        # Add classification features
        for feat_idx, feat_name in enumerate(feature_names_clf):
            all_importances[feat_name] = importances[feat_idx]
        
        # Add regression features (weighted average if already exists)
        for feat_idx, feat_name in enumerate(feature_names_reg):
            if feat_name in all_importances:
                # Average with classification importance
                all_importances[feat_name] = (
                    all_importances[feat_name] + 
                    importances_reg[feat_idx] + 
                    importances_xgb[feat_idx]
                ) / 3
            else:
                # Only in regression (e.g., Anomaly_Score)
                all_importances[feat_name] = (importances_reg[feat_idx] + importances_xgb[feat_idx]) / 2
        
        sorted_features = sorted(all_importances.items(), key=lambda x: x[1], reverse=True)[:10]
        top_features, top_importances = zip(*sorted_features)
        
        axes[1, 1].barh(range(len(top_features)), top_importances, color='#e74c3c')
        axes[1, 1].set_yticks(range(len(top_features)))
        axes[1, 1].set_yticklabels(top_features)
        axes[1, 1].set_xlabel('Average Feature Importance')
        axes[1, 1].set_title('Top 10 Features Across All Models', fontsize=12, fontweight='bold')
        axes[1, 1].invert_yaxis()
        axes[1, 1].grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(self.visualizations_dir / "feature_importance.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n✅ Visualizations saved to: {self.visualizations_dir}")
    
    def generate_report(self) -> None:
        """Generate comprehensive Phase 4 report"""
        print("\n" + "="*80)
        print("STEP 8: GENERATING PHASE 4 REPORT")
        print("="*80)
        
        report_path = self.reports_dir / "phase4_report.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("PHASE 4: PREDICTIVE MODELING & EVALUATION\n")
            f.write("="*80 + "\n\n")
            f.write(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Regression Results
            f.write("="*80 + "\n")
            f.write("YIELD PREDICTION (REGRESSION)\n")
            f.write("="*80 + "\n\n")
            
            f.write("Target Variable: Peak_NDVI (proxy for crop yield)\n")
            f.write(f"Dataset Size: {len(self.y_regression):,} parcels\n\n")
            
            # Random Forest
            rf_reg = self.results['rf_regression']
            f.write("Random Forest Regressor:\n")
            f.write(f"  Test RMSE: {rf_reg['test_rmse']:.4f}\n")
            f.write(f"  Test MAE:  {rf_reg['test_mae']:.4f}\n")
            f.write(f"  Test R²:   {rf_reg['test_r2']:.4f}\n\n")
            
            # XGBoost
            xgb_reg = self.results['xgb_regression']
            f.write("XGBoost Regressor:\n")
            f.write(f"  Test RMSE: {xgb_reg['test_rmse']:.4f}\n")
            f.write(f"  Test MAE:  {xgb_reg['test_mae']:.4f}\n")
            f.write(f"  Test R²:   {xgb_reg['test_r2']:.4f}\n\n")
            
            # Classification Results
            f.write("="*80 + "\n")
            f.write("STRESS CLASSIFICATION\n")
            f.write("="*80 + "\n\n")
            
            f.write("Target Variable: Is_Anomaly (stressed vs. healthy)\n")
            f.write(f"Dataset Size: {len(self.y_classification):,} parcels\n\n")
            
            # Random Forest
            rf_clf = self.results['rf_classification']
            f.write("Random Forest Classifier:\n")
            f.write(f"  Accuracy:  {rf_clf['accuracy']:.4f}\n")
            f.write(f"  Precision: {rf_clf['precision']:.4f}\n")
            f.write(f"  Recall:    {rf_clf['recall']:.4f}\n")
            f.write(f"  F1-Score:  {rf_clf['f1_score']:.4f}\n")
            f.write(f"  ROC-AUC:   {rf_clf['roc_auc']:.4f}\n\n")
            
            # LSTM
            lstm_clf = self.results['lstm_classification']
            f.write("LSTM Classifier:\n")
            f.write(f"  Accuracy:  {lstm_clf['accuracy']:.4f}\n")
            f.write(f"  Precision: {lstm_clf['precision']:.4f}\n")
            f.write(f"  Recall:    {lstm_clf['recall']:.4f}\n")
            f.write(f"  F1-Score:  {lstm_clf['f1_score']:.4f}\n")
            f.write(f"  ROC-AUC:   {lstm_clf['roc_auc']:.4f}\n\n")
            
            # Output Files
            f.write("="*80 + "\n")
            f.write("OUTPUT FILES\n")
            f.write("="*80 + "\n\n")
            f.write("Predictions:\n")
            f.write("  • rf_yield_predictions.csv\n")
            f.write("  • xgb_yield_predictions.csv\n")
            f.write("  • rf_stress_predictions.csv\n")
            f.write("  • lstm_stress_predictions.csv\n\n")
            f.write("Models:\n")
            f.write("  • lstm_stress_model.keras\n\n")
            f.write("Visualizations:\n")
            f.write("  • regression_results.png\n")
            f.write("  • classification_results.png\n")
            f.write("  • lstm_training_history.png\n\n")
            
            # Key Insights
            f.write("="*80 + "\n")
            f.write("KEY INSIGHTS\n")
            f.write("="*80 + "\n\n")
            f.write("✨ Regression Models:\n")
            f.write("  Both Random Forest and XGBoost successfully predict crop yield\n")
            f.write("  (Peak NDVI) with reasonable accuracy.\n\n")
            f.write("✨ Classification Models:\n")
            f.write("  Models can identify stressed parcels with high precision,\n")
            f.write("  enabling early intervention for crop management.\n\n")
            f.write("✨ Temporal Modeling:\n")
            f.write("  LSTM captures temporal patterns in NDVI/EVI time-series,\n")
            f.write("  providing deeper insights into growth dynamics.\n\n")
            
            # Next Steps
            f.write("="*80 + "\n")
            f.write("NEXT STEPS: PHASE 5\n")
            f.write("="*80 + "\n\n")
            f.write("✨ Ready for Phase 5: Interactive Dashboard\n\n")
            f.write("  Use predictions and models for:\n")
            f.write("  • Interactive Streamlit dashboard\n")
            f.write("  • Real-time yield predictions\n")
            f.write("  • Stress monitoring and alerts\n")
            f.write("  • Actionable recommendations for farmers\n\n")
            
            f.write("="*80 + "\n")
            f.write("PHASE 4 COMPLETE ✅\n")
            f.write("="*80 + "\n")
        
        print(f"\n💾 Report saved to: {report_path}")
        
        # Save metrics as JSON
        metrics = {
            'rf_regression': {k: v for k, v in rf_reg.items() if not isinstance(v, (np.ndarray, pd.Series, dict))},
            'xgb_regression': {k: v for k, v in xgb_reg.items() if not isinstance(v, (np.ndarray, pd.Series, dict))},
            'rf_classification': {k: v for k, v in rf_clf.items() if not isinstance(v, (np.ndarray, pd.Series, dict))},
            'lstm_classification': {k: v for k, v in lstm_clf.items() if not isinstance(v, (np.ndarray, pd.Series, dict, list))}
        }
        
        with open(self.evaluation_dir / "metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"💾 Metrics saved to: metrics.json")


def main():
    """Main execution function"""
    print("\n" + "🌾"*40)
    print("CSCE5380 - Crop Health Monitoring from Remote Sensing")
    print("PHASE 4: Predictive Modeling & Evaluation")
    print("🌾"*40 + "\n")
    
    print("="*80)
    print("PHASE 4: PREDICTIVE MODELING & EVALUATION")
    print("="*80)
    
    # Initialize engine
    engine = PredictiveModelingEngine(
        phase2_dir="./outputs/phase2",
        phase3_dir="./outputs/phase3",
        output_dir="./outputs/phase4"
    )
    
    # Run complete pipeline
    print("\n\n" + "🚀"*40)
    print("RUNNING PHASE 4 COMPLETE PIPELINE")
    print("🚀"*40 + "\n")
    
    start_time = datetime.now()
    
    try:
        # Load data
        engine.load_data()
        
        # Prepare datasets
        engine.prepare_datasets()
        
        # Train models
        engine.train_random_forest_regression()
        engine.train_xgboost_regression()
        engine.train_random_forest_classification()
        engine.train_lstm_classification()
        
        # Visualize results
        engine.visualize_results()
        
        # Generate report
        engine.generate_report()
        
        # Summary
        elapsed_time = (datetime.now() - start_time).total_seconds()
        
        print("\n" + "="*80)
        print("PHASE 4 PIPELINE COMPLETE ✅")
        print("="*80)
        print(f"\n⏱️  Total execution time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
        print(f"📊 Models trained: 4 (RF Regression, XGBoost Regression, RF Classification, LSTM)")
        print(f"💾 Outputs saved to: {engine.output_dir}")
        print("\n✨ Ready for Phase 5: Interactive Dashboard")
        
    except Exception as e:
        print(f"\n❌ Error in Phase 4: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
