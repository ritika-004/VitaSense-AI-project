import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import xgboost as xgb
import os

# ===== Step 1: Load dataset =====
file_path = r"PPG-BP dataset.xlsx"
try:
    df = pd.read_excel(file_path, header=1)
    print("✅ Dataset loaded successfully! Shape:", df.shape)
except Exception as e:
    print("❌ Error loading dataset:", e)
    exit()

# ===== Step 2: Clean column names =====
df.columns = df.columns.str.strip()
rename_map = {
    'Systolic Blood Pressure(mmHg)': 'SBP',
    'Diastolic Blood Pressure(mmHg)': 'DBP',
    'Sex(M/F)': 'Sex',
    'Age(year)': 'Age',
    'Height(cm)': 'Height',
    'Weight(kg)': 'Weight',
    'Heart Rate(b/m)': 'HeartRate',
    'BMI(kg/m^2)': 'BMI'
}
df = df.rename(columns=rename_map)
print("\n🧹 Cleaned column names:")
print(df.columns.tolist())

# ===== Step 3: Drop missing target rows =====
if 'SBP' not in df.columns or 'DBP' not in df.columns:
    print("❌ Target columns SBP and DBP not found. Please check dataset.")
    exit()
df = df.dropna(subset=['SBP', 'DBP'])

# ===== Step 4: Encode categorical column =====
if 'Sex' in df.columns:
    df['Sex'] = df['Sex'].map({'Male': 1, 'Female': 0})

# ===== Step 5: Select features =====
features = ['Age', 'Height', 'Weight', 'HeartRate', 'BMI']
available_features = [f for f in features if f in df.columns]
if not available_features:
    print("❌ No valid feature columns found.")
    exit()

X = df[available_features]
y_sbp = df['SBP']
y_dbp = df['DBP']

print(f"\n✅ Features used: {available_features}")
print(f"Dataset size after cleaning: {X.shape}")

# ===== Step 6: Train-test split =====
X_train, X_test, y_train_sbp, y_test_sbp = train_test_split(X, y_sbp, test_size=0.2, random_state=42)
_, _, y_train_dbp, y_test_dbp = train_test_split(X, y_dbp, test_size=0.2, random_state=42)

# ===== Step 7: Define base models =====
base_models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(random_state=42, n_estimators=100),
    "GradientBoosting": GradientBoostingRegressor(random_state=42),
    "XGBoost": xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
}

# ===== Step 8: Create Ensemble model =====
ensemble_estimators = [
    ('lr', LinearRegression()),
    ('rf', RandomForestRegressor(random_state=42, n_estimators=50)),
    ('gb', GradientBoostingRegressor(random_state=42, n_estimators=50)),
    ('xgb', xgb.XGBRegressor(objective='reg:squarederror', random_state=42, n_estimators=50))
]
ensemble_model = VotingRegressor(estimators=ensemble_estimators)

# Combine all models
models = {
    **base_models,
    "Ensemble": ensemble_model
}

results = []
trained_models = {}

# ===== Step 9: Train and evaluate each model =====
for name, model in models.items():
    print(f"\n🚀 Training {name} model for SBP...")
    model_sbp = model  # Clone for SBP
    model_sbp.fit(X_train, y_train_sbp)
    pred_sbp = model_sbp.predict(X_test)
    mae_sbp = mean_absolute_error(y_test_sbp, pred_sbp)
    r2_sbp = r2_score(y_test_sbp, pred_sbp)
    
    print(f"🚀 Training {name} model for DBP...")
    # Create new instance for DBP (avoid overwriting SBP model)
    if name == "LinearRegression":
        model_dbp = LinearRegression()
    elif name == "RandomForest":
        model_dbp = RandomForestRegressor(random_state=42, n_estimators=100)
    elif name == "GradientBoosting":
        model_dbp = GradientBoostingRegressor(random_state=42)
    elif name == "XGBoost":
        model_dbp = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    elif name == "Ensemble":
        model_dbp = VotingRegressor(estimators=[
            ('lr', LinearRegression()),
            ('rf', RandomForestRegressor(random_state=42, n_estimators=50)),
            ('gb', GradientBoostingRegressor(random_state=42, n_estimators=50)),
            ('xgb', xgb.XGBRegressor(objective='reg:squarederror', random_state=42, n_estimators=50))
        ])
    
    model_dbp.fit(X_train, y_train_dbp)
    pred_dbp = model_dbp.predict(X_test)
    mae_dbp = mean_absolute_error(y_test_dbp, pred_dbp)
    r2_dbp = r2_score(y_test_dbp, pred_dbp)
    
    avg_r2 = (r2_sbp + r2_dbp) / 2
    
    results.append({
        "Model": name,
        "MAE (SBP)": round(mae_sbp, 2),
        "R² (SBP)": round(r2_sbp, 3),
        "MAE (DBP)": round(mae_dbp, 2),
        "R² (DBP)": round(r2_dbp, 3),
        "Avg R²": round(avg_r2, 3)
    })
    
    # Save both models (SBP and DBP)
    trained_models[name] = {
        'sbp': model_sbp,
        'dbp': model_dbp
    }
    
    # Save to disk
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model_sbp, os.path.join(model_dir, f"{name}_SBP.pkl"))
    joblib.dump(model_dbp, os.path.join(model_dir, f"{name}_DBP.pkl"))
    print(f"💾 {name} models (SBP & DBP) saved successfully!")

# ===== Step 10: Create comparison table =====
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="Avg R²", ascending=False)

print("\n📊 Model Comparison Summary:")
print(results_df)

# Save summary
comparison_path = "model_comparison.xlsx"
results_df.to_excel(comparison_path, index=False)
print(f"\n📁 Comparison summary saved at: {comparison_path}")

# ===== Step 11: Identify best model =====
best_model_row = results_df.iloc[0]
print("\n🏆 Best Model Based on Average R²:")
print(best_model_row)

print("\n✅ All models trained and saved successfully!")
print(f"📂 Models saved in: {os.path.abspath(model_dir)}")
print(f"📊 Comparison saved in: {os.path.abspath(comparison_path)}")
