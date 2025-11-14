# 🚀 QUICK START GUIDE - Crop Health Monitoring Project

## PROJECT STATUS: Phase 1 ✅ COMPLETE | Phases 2-5 Ready for Implementation

---

## ✅ WHAT'S WORKING NOW

### Phase 1 is COMPLETE and TESTED
```bash
cd "c:\Users\asus\Desktop\UNT\CSCE 5380\Project"
python src/phase1_preprocessing_v2.py
```

**Output**:
- ✅ Loaded 100 real PASTIS patches
- ✅ Computed NDVI/EVI for all timesteps
- ✅ Exported to `outputs/phase1/processed_data/`
- ✅ 50 sample patches with NDVI, EVI, parcel IDs, and labels saved

**Check Results**:
```bash
ls outputs/phase1/processed_data/sample_patches/
# You should see: 10000_ndvi.npy, 10000_evi.npy, 10000_parcels.npy, etc.
```

---

## 🔧 NEXT: IMPLEMENT PHASES 2-5

Your existing code files are mostly complete but need minor updates to match the prompt specifications. Here's what needs to be done for each phase:

### Phase 2: Segmentation & Feature Extraction

**File to Update**: `src/phase2_segmentation.py`

**Key Changes Needed**:
1. Load `*_parcels.npy` (parcel boundaries) from Phase 1
2. For each unique parcel ID, extract temporal statistics:
   - Mean, Std, P25, P75 of NDVI/EVI at EACH timestep
3. Create DataFrame: `Plot_ID | Timestep | Mean_NDVI | Std_NDVI | ...`
4. Compute GLCM texture features using `skimage.feature.graycomatrix`

**Quickest Solution**: Your existing `src/phase2_segmentation.py` already has most of this structure. Key additions:
- After loading patches, iterate through unique parcel IDs
- Extract pixels for each parcel and compute per-timestep stats
- Save as `outputs/phase2/features/temporal_features.csv`

---

### Phase 3: Pattern Discovery with DTW

**File to Update**: `src/phase3_patterndiscovery_and_anomalydetection.py`

**Key Changes Needed**:
1. Load `temporal_features.csv` from Phase 2
2. Pivot to get NDVI time-series per parcel
3. **Critical**: Use `tslearn.clustering.TimeSeriesKMeans` with DTW metric:
   ```python
   from tslearn.clustering import TimeSeriesKMeans
   model = TimeSeriesKMeans(n_clusters=5, metric="dtw", random_state=42)
   clusters = model.fit_predict(ndvi_series)
   ```
4. Apply Isolation Forest or LOF for anomaly detection
5. Save cluster assignments and anomaly scores

**Install tslearn first**:
```bash
pip install tslearn
```

---

### Phase 4: Predictive Modeling

**File to Create**: `src/phase4_predictivemodeling.py`

**What to Implement**:
1. Load features from Phase 2 & 3
2. Create target variables:
   - **Yield proxy**: Use `max(NDVI)` per parcel
   - **Stress labels**: Use anomaly scores from Phase 3
3. Train models:
   - **Baseline**: Random Forest / XGBoost
   - **Temporal**: LSTM with TensorFlow/Keras
4. Evaluate and save models

**Example LSTM**:
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

model = Sequential([
    LSTM(64, input_shape=(timesteps, features), return_sequences=True),
    LSTM(32),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=50)
model.save('outputs/phase4/models/lstm_model.h5')
```

---

### Phase 5: Interactive Dashboard

**File to Create**: `src/phase5_dashboard.py`

**What to Implement**:
```python
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.title("🌾 Crop Health Monitoring Dashboard")

# Load data
features = pd.read_csv("outputs/phase2/features/temporal_features.csv")
predictions = pd.read_csv("outputs/phase4/predictions/predictions.csv")

# Sidebar: Select plot
plot_id = st.sidebar.selectbox("Select Plot", features['Plot_ID'].unique())

# Display time-series
plot_data = features[features['Plot_ID'] == plot_id]
fig = go.Figure()
fig.add_trace(go.Scatter(x=plot_data['Timestep'], y=plot_data['Mean_NDVI'], name='NDVI'))
fig.add_trace(go.Scatter(x=plot_data['Timestep'], y=plot_data['Mean_EVI'], name='EVI'))
st.plotly_chart(fig)

# Show predictions
pred = predictions[predictions['Plot_ID'] == plot_id].iloc[0]
st.metric("Predicted Yield", f"{pred['Yield']:.2f}")
st.metric("Stress Status", pred['Status'])

# Recommendations
if pred['Anomaly_Score'] > 0.8:
    st.warning(f"⚠️ Parcel {plot_id} shows signs of distress. Recommend inspection.")
```

**Run with**:
```bash
streamlit run src/phase5_dashboard.py
```

---

## 💡 PRO TIPS

### 1. If You're Short on Time
Focus on getting Phases 2 and 3 working first since they provide the most critical insights (temporal patterns and anomaly detection). Phases 4 and 5 can be simplified:
- Phase 4: Use only Random Forest (skip LSTM initially)
- Phase 5: Create simple plots in Jupyter Notebook instead of full dashboard

### 2. Quick Testing Strategy
Use a smaller subset to iterate faster:
```python
# In each phase script, add at the top:
N_PATCHES_TO_PROCESS = 10  # Instead of 100
```

### 3. Key Libraries Installation
```bash
pip install tslearn xgboost tensorflow streamlit folium
```

---

## 📊 DATA FLOW

```
Phase 1 (✅ DONE)
└─> outputs/phase1/processed_data/sample_patches/
    ├─> {patch_id}_ndvi.npy       # NDVI time series (T, H, W)
    ├─> {patch_id}_evi.npy        # EVI time series (T, H, W)
    ├─> {patch_id}_parcels.npy    # Parcel IDs (H, W)
    └─> {patch_id}_labels.npy     # Crop labels (H, W)

Phase 2 (TO DO)
└─> outputs/phase2/features/temporal_features.csv
    Columns: Plot_ID, Timestep, Mean_NDVI, Std_NDVI, P25_NDVI, P75_NDVI, Mean_EVI, ...

Phase 3 (TO DO)
├─> outputs/phase3/clusters/cluster_assignments.csv
│   Columns: Plot_ID, Cluster_ID
└─> outputs/phase3/anomalies/anomaly_scores.csv
    Columns: Plot_ID, Anomaly_Score, Is_Anomaly

Phase 4 (TO DO)
└─> outputs/phase4/predictions/predictions.csv
    Columns: Plot_ID, Predicted_Yield, Stress_Status

Phase 5 (TO DO)
└─> Streamlit Dashboard (runs in browser)
```

---

## 🐛 COMMON ISSUES & FIXES

### Issue: "Cannot import tslearn"
```bash
pip install tslearn
```

### Issue: "Memory error loading all patches"
Reduce batch size in Phase 2:
```python
# Process in batches of 10
for batch in range(0, len(patches), 10):
    process_batch(patches[batch:batch+10])
```

### Issue: "TensorFlow not working"
For Windows, ensure you have Python 3.10-3.11 (not 3.13):
```bash
python --version  # Should be 3.10 or 3.11
```

---

## ✅ VERIFICATION CHECKLIST

After each phase, verify outputs exist:

```bash
# Phase 1
ls outputs/phase1/processed_data/metadata_summary.csv

# Phase 2
ls outputs/phase2/features/temporal_features.csv

# Phase 3
ls outputs/phase3/clusters/cluster_assignments.csv

# Phase 4
ls outputs/phase4/predictions/predictions.csv

# Phase 5
streamlit run src/phase5_dashboard.py
# Should open browser at http://localhost:8501
```

---

## 🎯 PRIORITY ORDER

If you need to demonstrate the project quickly:

1. **Must Have** (Critical):
   - ✅ Phase 1 (DONE)
   - Phase 2: Feature extraction (focus on temporal stats)
   - Phase 3: DTW clustering + basic anomaly detection

2. **Should Have** (Important):
   - Phase 4: At least Random Forest model
   - Simple visualizations (matplotlib plots)

3. **Nice to Have** (Bonus):
   - LSTM model in Phase 4
   - Full Streamlit dashboard in Phase 5
   - GLCM texture features in Phase 2

---

## 📧 NEED HELP?

1. **Phase 1 Issues**: Check `outputs/phase1/phase1_report.txt` for diagnostics
2. **Data Loading Issues**: Verify files exist in `data/PASTIS/DATA_S2/` and `data/PASTIS/ANNOTATIONS/`
3. **Package Issues**: Run `pip install -r requirements.txt` again

---

**Last Updated**: November 12, 2025  
**Status**: Phase 1 Complete, Ready for Phases 2-5  
**Your Data**: ✅ Real PASTIS dataset loaded successfully
