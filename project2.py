import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             fbeta_score, classification_report)
import xgboost as xgb
import optuna
import joblib
import shap
import warnings
warnings.filterwarnings("ignore")
base_path=r"C:\Users\Prachi\OneDrive\Desktop\Project"
admissions=pd.read_csv(f"{base_path}\\ADMISSIONS.csv", parse_dates=["admittime", "dischtime", "deathtime"])
patients=pd.read_csv(f"{base_path}\\PATIENTS.csv", parse_dates=["dob", "dod"])
diagnoses=pd.read_csv(f"{base_path}\\DIAGNOSES_ICD.csv")
labevents=pd.read_csv(f"{base_path}\\LABEVENTS.csv", parse_dates=["charttime"])
icustays=pd.read_csv(f"{base_path}\\ICUSTAYS.csv")
prescription=pd.read_csv(f"{base_path}\\PRESCRIPTIONS.csv")
procedures=pd.read_csv(f"{base_path}\\PROCEDURES_ICD.csv")

df=admissions[["subject_id","hadm_id","admittime","admission_type","insurance","language","religion","marital_status","ethnicity","hospital_expire_flag"]].copy()
df=df.merge(patients[["subject_id","dob","gender"]],on="subject_id",how="left")
df["age"]=(df["admittime"]-df["dob"]).dt.days/365
df["age"]=df["age"].clip(0, 90)
df=df[df["age"]>=18].drop(columns=["dob"]).reset_index(drop=True)
print(f"Cohort size(adults):{len(df):,}")
print(f"Mortality rate:{df['hospital_expire_flag'].mean()*100:.1f}%\n")
icd_map={"0":"infectious", "1":"neoplasm", "2":"blood", "3":"endocrine", "4":"mental", "5":"nervous","6":"circulatory", "7":"respiratory", "8":"digestive", "9":"genitourinary", "E":"external", "V":"supplementary"
}
primary_diag=diagnoses[diagnoses["seq_num"]==1][["hadm_id","icd9_code"]].copy()
primary_diag["icd_chapter"]=primary_diag["icd9_code"].astype(str).str[0].map(icd_map).fillna("other")
df=df.merge(primary_diag[["hadm_id","icd_chapter"]],on="hadm_id",how="left")
df["icd_chapter"]=df["icd_chapter"].fillna("other")
comorbidity=diagnoses.groupby("hadm_id")["icd9_code"].count().rename("comorbidity_count").reset_index()
proc_count=procedures.groupby("hadm_id")["icd9_code"].count().rename("procedure_count").reset_index()
drug_count=prescription.groupby("hadm_id")["drug"].nunique().rename("unique_drug_count").reset_index()
for count_df in [comorbidity, proc_count, drug_count]:
    df=df.merge(count_df,on="hadm_id",how="left")
df[["comorbidity_count","procedure_count","unique_drug_count"]]=df[["comorbidity_count","procedure_count","unique_drug_count"]].fillna(0)
icu_agg=icustays.groupby("hadm_id").agg(icu_flag=("icustay_id","count"),icu_los=("los","sum")).reset_index()
icu_agg["icu_flag"]=(icu_agg["icu_flag"]>0).astype(int)
df=df.merge(icu_agg,on="hadm_id",how="left")
df["icu_flag"]=df["icu_flag"].fillna(0).astype(int)
df["icu_los"]=df["icu_los"].fillna(0)
LAB_ITEMS={50912:"creatinine",51222:"hemoglobin",51300:"wbc",50820:"ph",50802:"base_excess",51006:"urea_nitrogen"}
labs=labevents[labevents["itemid"].isin(LAB_ITEMS.keys())].copy()
labs["lab_name"]=labs["itemid"].map(LAB_ITEMS)
labs=labs.dropna(subset=["valuenum"])
labs_first=labs.sort_values("charttime").groupby(["hadm_id","lab_name"])["valuenum"].first().unstack().reset_index()
labs_first.columns.name=None
lab_cols=[c for c in LAB_ITEMS.values() if c in labs_first.columns]
for col in lab_cols:
    labs_first[f"{col}_missing"]=labs_first[col].isna().astype(int)
if "creatinine" in labs_first.columns:
    labs_first["creatinine_abnormal"]=((labs_first["creatinine"]<0.7)|(labs_first["creatinine"]>1.3)).astype(int)
if "wbc" in labs_first.columns:
    labs_first["wbc_abnormal"]=((labs_first["wbc"] < 4.5) | (labs_first["wbc"] > 11.0)).astype(int)
if "hemoglobin" in labs_first.columns:
    labs_first["hemoglobin_abnormal"]=(labs_first["hemoglobin"]<8.0).astype(int)
df=df.merge(labs_first,on="hadm_id",how="left")
CAT_COLS=["icd_chapter","insurance","admission_type","gender"]
NUM_COLS=["age","comorbidity_count","procedure_count","icu_flag","icu_los","unique_drug_count","creatinine","hemoglobin","wbc","ph"]
NUM_COLS=[c for c in NUM_COLS if c in df.columns]
FLAG_COLS=[c for c in df.columns if c.endswith("_missing") or c.endswith("_abnormal")]
x=pd.get_dummies(df[CAT_COLS+NUM_COLS+FLAG_COLS].copy(),columns=CAT_COLS)
y=df["hospital_expire_flag"]
for col in NUM_COLS:
    if f"{col}_missing" not in x.columns:
        x[f"{col}_missing"]=x[col].isna().astype(int)
x_train, x_test, y_train, y_test=train_test_split(x,y,random_state=42,test_size=0.2,stratify=y)
train_medians=x_train[NUM_COLS].median()
x_train[NUM_COLS]=x_train[NUM_COLS].fillna(train_medians)
x_test[NUM_COLS]=x_test[NUM_COLS].fillna(train_medians)
x_train=x_train.fillna(0)
x_test=x_test.fillna(0)
scale_weight=int((y_train==0).sum()/(y_train==1).sum())
def objective(trial):
    params={
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 2, 6),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 5),
        "scale_pos_weight": scale_weight,
        "random_state": 42, 
        "n_jobs": -1, 
        "eval_metric": "auc"
    }
    model=xgb.XGBClassifier(**params)
    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores=cross_val_score(model, x_train, y_train, cv=cv, scoring="roc_auc")
    return scores.mean()

optuna.logging.set_verbosity(optuna.logging.WARNING)
print("Running Optuna Hyperparameter Tuning...")
study=optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)
print(f"Best CV AUROC: {study.best_value:.4f}")

# Train Best Model and Calibrate Probabilities
best_xgb=xgb.XGBClassifier(**study.best_params, scale_pos_weight=scale_weight, eval_metric="auc", random_state=42)
calibrated=CalibratedClassifierCV(best_xgb, method="isotonic", cv=3)
calibrated.fit(x_train, y_train)

# Evaluation
y_proba=calibrated.predict_proba(x_test)[:, 1]
y_pred=(y_proba >= 0.4).astype(int)

print(f"\nTest AUROC:  {roc_auc_score(y_test, y_proba):.4f}")
print(f"Test PR-AUC: {average_precision_score(y_test, y_proba):.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=["Survived", "Died"]))

# --- 6. SHAP Explanations ---
os.makedirs("reports/figures", exist_ok=True)

# We refit the raw best model because SHAP cannot interpret the wrapper from CalibratedClassifierCV
base_model=xgb.XGBClassifier(**study.best_params, scale_pos_weight=scale_weight, eval_metric="auc", random_state=42)
base_model.fit(x_train, y_train)

explainer=shap.TreeExplainer(base_model)
shap_values=explainer.shap_values(x_test)

# SHAP Bar Plot
shap.summary_plot(shap_values, x_test, plot_type="bar", show=False)
plt.title("Feature Importance — SHAP (Global)")
plt.tight_layout()
plt.savefig("reports/figures/shap_global.png", dpi=150, bbox_inches="tight")
plt.show()

# SHAP Beeswarm Plot
shap.summary_plot(shap_values, x_test, show=False)
plt.tight_layout()
plt.savefig("reports/figures/shap_beeswarm.png", dpi=150, bbox_inches="tight")
plt.show()

# Save Artifacts
os.makedirs("models", exist_ok=True)
joblib.dump(calibrated, "models/mortality_model.pkl")
joblib.dump(train_medians, "models/train_medians.pkl")
joblib.dump(x_train.columns.tolist(), "models/feature_names.pkl")
joblib.dump(explainer, "models/shap_explainer.pkl")