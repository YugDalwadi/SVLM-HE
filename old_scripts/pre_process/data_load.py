import os
import pandas as pd

# ---- 1. Paths ----
data_dir = "/raid/deeksha/mimic/mimic-iv-3.1"   # <- update to your extracted MIMIC-IV folder

# ---- 2. Load ICD dictionary ----
d_icd = pd.read_csv(os.path.join(data_dir, "hosp/d_icd_diagnoses.csv.gz"))
print("ICD dictionary shape:", d_icd.shape)

# Find all ICD codes related to hepatic encephalopathy
he_codes = d_icd[d_icd['long_title'].str.contains("Hepatic encephalopathy", case=False, na=False)]
print("Hepatic encephalopathy codes:\n", he_codes)

# ---- 3. Load diagnoses and filter ----
diagnoses = pd.read_csv(os.path.join(data_dir, "hosp/diagnoses_icd.csv.gz"))
print("Diagnoses shape:", diagnoses.shape)

# Merge to get only HE diagnoses
he_diagnoses = diagnoses.merge(he_codes, on=['icd_code','icd_version'], how='inner')
print("HE patients shape:", he_diagnoses.shape)

# ---- 4. Get unique patient IDs ----
he_subjects = he_diagnoses['subject_id'].unique()
print("Number of unique HE patients:", len(he_subjects))

# ---- 5. Extract related patient-level data ----

# Patients
patients = pd.read_csv(os.path.join(data_dir, "hosp/patients.csv.gz"))
he_patients = patients[patients['subject_id'].isin(he_subjects)]

# Admissions
admissions = pd.read_csv(os.path.join(data_dir, "hosp/admissions.csv.gz"))
he_admissions = admissions[admissions['subject_id'].isin(he_subjects)]

# Labs (example: bilirubin, ammonia, INR, albumin)
labevents = pd.read_csv(os.path.join(data_dir, "hosp/labevents.csv.gz"), low_memory=False)
he_labs = labevents[labevents['subject_id'].isin(he_subjects)]

# ---- 6. Save filtered data ----
out_dir = "/raid/deeksha/mimic/he_extracted"
os.makedirs(out_dir, exist_ok=True)

he_patients.to_csv(os.path.join(out_dir, "he_patients.csv"), index=False)
he_admissions.to_csv(os.path.join(out_dir, "he_admissions.csv"), index=False)
he_diagnoses.to_csv(os.path.join(out_dir, "he_diagnoses.csv"), index=False)
he_labs.to_csv(os.path.join(out_dir, "he_labs.csv"), index=False)

print("✅ Extracted HE cohort saved to", out_dir)
