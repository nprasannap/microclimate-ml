import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import brier_score_loss, mean_absolute_error
from sklearn.impute import SimpleImputer

# Set publication-ready style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8

# ==========================================
# 1. DIRECTORY CONFIGURATION & METADATA
# ==========================================
base_dir = r"E:\WF\CrossSite"
master_dir = os.path.join(base_dir, "Master_Files")
output_dir = os.path.join(base_dir, "Analysis_Final_Verified")
os.makedirs(output_dir, exist_ok=True)

locations = ["Badulla", "Colombo", "Galle", "Kandy"]

spatial_metadata = {
    "Colombo": {"Elevation": 7.0,   "Dist_Coast": 1.0,   "Slope_Index": 0.05, "Climate_Zone": "Wet_Lowland"},
    "Galle":   {"Elevation": 15.0,  "Dist_Coast": 1.0,   "Slope_Index": 0.08, "Climate_Zone": "Wet_Lowland"},
    "Kandy":   {"Elevation": 500.0, "Dist_Coast": 75.0,  "Slope_Index": 0.45, "Climate_Zone": "Wet_Midcountry"},
    "Badulla": {"Elevation": 680.0, "Dist_Coast": 100.0, "Slope_Index": 0.62, "Climate_Zone": "Intermediate_Upcountry"}
}

pop_cols_24h = ['Ggl_PoP', 'Bug_Composite_24h_PoP', 'TWC_Composite_24h_PoP', 'Accu_Composite_24h_PoP']
maxt_cols = ['Ggl_MaxT', 'Bug_MaxT', 'TWC_MaxT', 'Accu_MaxT']
mint_cols = ['Ggl_MinT', 'Bug_MinT', 'TWC_MinT', 'Accu_MinT']
rh_cols = ['Ggl_RH', 'Bug_Day_RH', 'TWC_Day_RH']

base_feature_cols = [
    'Ggl_PoP', 'Ggl_RH', 'Ggl_MaxT', 'Ggl_MinT', 'Ggl_Wind',
    'Bug_Day_PoP', 'Bug_Night_PoP', 'Bug_MaxT', 'Bug_MinT', 'Bug_Day_RH', 'Bug_Night_RH',
    'TWC_Day_PoP', 'TWC_Night_PoP', 'TWC_MaxT', 'TWC_MinT', 'TWC_Day_RH', 'TWC_Night_RH',
    'Accu_Day_PoP', 'Accu_Night_PoP', 'Accu_MaxT', 'Accu_MinT',
    'Bug_Composite_24h_PoP', 'TWC_Composite_24h_PoP', 'Accu_Composite_24h_PoP'
]

spatial_num_cols = ['Elevation', 'Dist_Coast', 'Slope_Index']
climate_zone_cols = ['Climate_Zone_Wet_Lowland', 'Climate_Zone_Wet_Midcountry', 'Climate_Zone_Intermediate_Upcountry']

# Target observation standards
target_rain_col = 'Observed_Rain_Binary_AccuBug'
target_maxt_col = 'Observed_MaxT_AccuBug'

# ==========================================
# 2. DATA LOADING & FEATURE ENGINEERING
# ==========================================
pooled_data_list = []

for loc in locations:
    # Updated to match actual filename convention: Master_2025_.xlsx
    file_path = os.path.join(master_dir, f"Master_2025_{loc}.xlsx")
    if os.path.exists(file_path):
        df = pd.read_excel(file_path, sheet_name='ML_Master_Sheet')
        
        df['Elevation'] = spatial_metadata[loc]['Elevation']
        df['Dist_Coast'] = spatial_metadata[loc]['Dist_Coast']
        df['Slope_Index'] = spatial_metadata[loc]['Slope_Index']
        df['Climate_Zone_Wet_Lowland'] = 1 if spatial_metadata[loc]['Climate_Zone'] == 'Wet_Lowland' else 0
        df['Climate_Zone_Wet_Midcountry'] = 1 if spatial_metadata[loc]['Climate_Zone'] == 'Wet_Midcountry' else 0
        df['Climate_Zone_Intermediate_Upcountry'] = 1 if spatial_metadata[loc]['Climate_Zone'] == 'Intermediate_Upcountry' else 0
        df['Location_Name'] = loc
        
        # Microclimate interaction features
        df['Ensemble_PoP_Mean'] = df[pop_cols_24h].mean(axis=1)
        df['Ensemble_PoP_Std'] = df[pop_cols_24h].std(axis=1)
        df['Ensemble_PoP_Max'] = df[pop_cols_24h].max(axis=1)
        df['Ensemble_PoP_Range'] = df['Ensemble_PoP_Max'] - df[pop_cols_24h].min(axis=1)
        
        df['Ensemble_MaxT_Mean'] = df[maxt_cols].mean(axis=1)
        df['Ensemble_MaxT_Std'] = df[maxt_cols].std(axis=1)
        df['Ensemble_MinT_Mean'] = df[mint_cols].mean(axis=1)
        df['Ensemble_RH_Mean'] = df[rh_cols].mean(axis=1)
        
        df['Diurnal_Temp_Range'] = df['Ensemble_MaxT_Mean'] - df['Ensemble_MinT_Mean']
        df['PoP_Elevation_Interaction'] = df['Ensemble_PoP_Mean'] * (df['Elevation'] / 100.0)
        df['PoP_Coastal_Interaction'] = df['Ensemble_PoP_Mean'] / (df['Dist_Coast'] + 1.0)
        df['Thermal_Lapse_Proxy'] = df['Ensemble_MaxT_Mean'] - (0.0065 * df['Elevation'])
        
        pooled_data_list.append(df)
    else:
        print(f"Warning: File not found - {file_path}")

df_master_pooled = pd.concat(pooled_data_list, ignore_index=True).dropna(subset=[target_rain_col, target_maxt_col])

interaction_cols = [
    'Ensemble_PoP_Mean', 'Ensemble_PoP_Std', 'Ensemble_PoP_Max', 'Ensemble_PoP_Range',
    'Ensemble_MaxT_Mean', 'Ensemble_MaxT_Std', 'Ensemble_MinT_Mean', 'Ensemble_RH_Mean',
    'Diurnal_Temp_Range', 'PoP_Elevation_Interaction', 'PoP_Coastal_Interaction', 'Thermal_Lapse_Proxy'
]
full_feature_cols = base_feature_cols + spatial_num_cols + climate_zone_cols + interaction_cols

imputer = SimpleImputer(strategy='mean')

# ==========================================
# 3. LEAVE-ONE-LOCATION-OUT EVALUATION
# ==========================================
baseline_results = []
interaction_results = []
feature_importances = []

for test_loc in locations:
    df_train = df_master_pooled[df_master_pooled['Location_Name'] != test_loc]
    df_test = df_master_pooled[df_master_pooled['Location_Name'] == test_loc]
    
    y_train_r = df_train[target_rain_col].values
    y_test_r = df_test[target_rain_col].values
    y_train_m = df_train[target_maxt_col].values
    y_test_m = df_test[target_maxt_col].values
    
    # Baseline Model (Raw Features)
    X_train_b = imputer.fit_transform(df_train[base_feature_cols])
    X_test_b = imputer.transform(df_test[base_feature_cols])
    
    rf_b_rain = RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=5, random_state=42)
    rf_b_rain.fit(X_train_b, y_train_r)
    p_b_rain = rf_b_rain.predict_proba(X_test_b)[:, 1]
    bs_b = brier_score_loss(y_test_r, p_b_rain)
    bs_ref = brier_score_loss(y_test_r, np.full_like(y_test_r, np.mean(y_test_r), dtype=float))
    bss_b = 1.0 - (bs_b / bs_ref) if bs_ref > 0 else 0.0
    
    rf_b_maxt = RandomForestRegressor(n_estimators=200, max_depth=8, min_samples_leaf=5, random_state=42)
    rf_b_maxt.fit(X_train_b, y_train_m)
    p_b_maxt = rf_b_maxt.predict(X_test_b)
    mae_b = mean_absolute_error(y_test_m, p_b_maxt)
    
    baseline_results.append({'Location': test_loc, 'BSS': bss_b, 'MAE': mae_b, 'Type': 'Baseline Model'})
    
    # Interaction Model (Full Engineered Features)
    X_train_i = imputer.fit_transform(df_train[full_feature_cols])
    X_test_i = imputer.transform(df_test[full_feature_cols])
    
    rf_i_rain = RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=5, random_state=42)
    rf_i_rain.fit(X_train_i, y_train_r)
    p_i_rain = rf_i_rain.predict_proba(X_test_i)[:, 1]
    bs_i = brier_score_loss(y_test_r, p_i_rain)
    bss_i = 1.0 - (bs_i / bs_ref) if bs_ref > 0 else 0.0
    
    rf_i_maxt = RandomForestRegressor(n_estimators=200, max_depth=8, min_samples_leaf=5, random_state=42)
    rf_i_maxt.fit(X_train_i, y_train_m)
    p_i_maxt = rf_i_maxt.predict(X_test_i)
    mae_i = mean_absolute_error(y_test_m, p_i_maxt)
    
    interaction_results.append({'Location': test_loc, 'BSS': bss_i, 'MAE': mae_i, 'Type': 'Verified Interaction Model'})
    
    # Track Feature Importances
    fi_df = pd.DataFrame({'Feature': full_feature_cols, 'Importance': rf_i_rain.feature_importances_})
    feature_importances.append(fi_df)

df_comp = pd.concat([pd.DataFrame(baseline_results), pd.DataFrame(interaction_results)])

# Export Table Summaries
df_pivot = df_comp.pivot(index='Location', columns='Type', values=['BSS', 'MAE'])
df_pivot.to_excel(os.path.join(output_dir, "Verified_Master_Model_Comparison.xlsx"))

# ==========================================
# 4. GENERATE PUBLICATION FIGURES
# ==========================================

# Figure 1: BSS Benchmark Comparison
plt.figure(figsize=(9, 5))
ax = sns.barplot(data=df_comp, x='Location', y='BSS', hue='Type', palette=['#7f8c8d', '#2ecc71'])
plt.title('Zero-Shot Brier Skill Score (BSS) Across Microclimates', fontsize=12, fontweight='bold', pad=12)
plt.ylabel('Brier Skill Score (BSS)', fontsize=10)
plt.xlabel('Unseen Target Location', fontsize=10)
plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
for p in ax.patches:
    if p.get_height() != 0:
        ax.annotate(f"{p.get_height():.3f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom' if p.get_height() > 0 else 'top', fontsize=8, xytext=(0, 3), textcoords='offset points')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "Verified_BSS_Benchmark_Comparison.png"), dpi=300)
plt.close()

# Figure 2: Max Temperature MAE Comparison
plt.figure(figsize=(9, 5))
ax = sns.barplot(data=df_comp, x='Location', y='MAE', hue='Type', palette=['#95a5a6', '#3498db'])
plt.title('Max Temperature MAE (°C) Across Unseen Locations', fontsize=12, fontweight='bold', pad=12)
plt.ylabel('MAE (°C)', fontsize=10)
plt.xlabel('Unseen Target Location', fontsize=10)
for p in ax.patches:
    if p.get_height() > 0:
        ax.annotate(f"{p.get_height():.2f}°C", (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=8, xytext=(0, 3), textcoords='offset points')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "Verified_MaxT_MAE_Comparison.png"), dpi=300)
plt.close()

# Figure 3: Mean Feature Importance
df_fi_all = pd.concat(feature_importances).groupby('Feature').mean().reset_index()
df_fi_top = df_fi_all.sort_values(by='Importance', ascending=False).head(12)

plt.figure(figsize=(10, 5))
sns.barplot(data=df_fi_top, x='Importance', y='Feature', palette='Blues_r')
plt.title('Top 12 Feature Importances in Verified Universal Model', fontsize=12, fontweight='bold')
plt.xlabel('Gini Feature Importance', fontsize=10)
plt.ylabel('')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "Verified_Top_Feature_Importances.png"), dpi=300)
plt.close()

# ==========================================
# 5. SHAP INTERPRETABILITY (GALLE CASE STUDY)
# ==========================================
target_loc = "Galle"
df_train_galle = df_master_pooled[df_master_pooled['Location_Name'] != target_loc]
df_test_galle = df_master_pooled[df_master_pooled['Location_Name'] == target_loc]

X_train_galle = imputer.fit_transform(df_train_galle[full_feature_cols])
y_train_galle = df_train_galle[target_rain_col].values

X_test_galle = imputer.transform(df_test_galle[full_feature_cols])
X_test_galle_df = pd.DataFrame(X_test_galle, columns=full_feature_cols)

rf_shap = RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=5, random_state=42)
rf_shap.fit(X_train_galle, y_train_galle)

explainer = shap.TreeExplainer(rf_shap)
shap_values = explainer.shap_values(X_test_galle_df)

if isinstance(shap_values, list):
    shap_vals_rain = shap_values[1]
elif len(shap_values.shape) == 3:
    shap_vals_rain = shap_values[:, :, 1]
else:
    shap_vals_rain = shap_values

plt.figure(figsize=(10, 6))
shap.summary_plot(shap_vals_rain, X_test_galle_df, max_display=12, show=False)
plt.title(f"SHAP Feature Attribution: {target_loc} (Zero-Shot)", fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f"SHAP_Summary_{target_loc}_Final.png"), dpi=300)
plt.close()

print(f"Pipeline executed successfully! Output folder updated at: {output_dir}")