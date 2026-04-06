"""
retrain_model.py

Retrains the RandomForest scot-free prediction model from scratch using the
pre-processed murder data parquet file (260224_murderdata.parquet).

This script reproduces the full data-preparation and model-training pipeline
from ex_analysis.ipynb so that the saved joblib file is always compatible with
the current scikit-learn / Python environment.

The final model is the "thin" model described in the notebook: a pipeline
trained on a reduced column set (13 low-importance columns removed) with
hyperparameters tuned via GridSearchCV.

Usage (run from the repo root or from NIBRS-data/):
    python NIBRS-data/retrain_model.py

The trained model is written to:
    web_app/scotfreerandomforestmodel.joblib
"""

import os
import joblib
import numpy as np
import polars as pl

from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit, GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OrdinalEncoder

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARQUET_PATH = os.path.join(SCRIPT_DIR, "260224_murderdata.parquet")
MODEL_OUT = os.path.join(SCRIPT_DIR, "..", "web_app", "scotfreerandomforestmodel.joblib")

# ---------------------------------------------------------------------------
# 1. Load raw data
# ---------------------------------------------------------------------------
print("Loading parquet …")
murderdata = pl.read_parquet(PARQUET_PATH)

# ---------------------------------------------------------------------------
# 2. Property-loss CDEs  (Destroyed / Seized / Looted)
# ---------------------------------------------------------------------------
print("Building property CDEs …")

destroyedpropertyCDE = murderdata.select(pl.col(["incident_id", "prop_desc_name", "prop_loss_name"])).unique()
destroyedpropertyCDE = destroyedpropertyCDE.with_columns(
    pl.when(pl.col("prop_loss_name").is_in(["Destroyed/Damaged/Vandalized", "Burned"]))
    .then(1)
    .when(pl.col("prop_loss_name").is_null())
    .then(None)
    .otherwise(0)
    .alias("prop_destruction")
)
destroyedpropertyCDE = destroyedpropertyCDE.group_by("incident_id").agg(pl.col("prop_destruction").sum())

seizedpropertyCDE = murderdata.select(pl.col(["incident_id", "prop_desc_name", "prop_loss_name"])).unique()
seizedpropertyCDE = seizedpropertyCDE.with_columns(
    pl.when(pl.col("prop_loss_name") == "Seized")
    .then(1)
    .when(pl.col("prop_loss_name").is_null())
    .then(None)
    .otherwise(0)
    .alias("taken_by_cops")
)
seizedpropertyCDE = seizedpropertyCDE.group_by("incident_id").agg(pl.col("taken_by_cops").sum())

lootedpropertyCDE = murderdata.select(pl.col(["incident_id", "prop_desc_name", "prop_loss_name"])).unique()
lootedpropertyCDE = lootedpropertyCDE.with_columns(
    pl.when(pl.col("prop_loss_name") == "Stolen/Etc")
    .then(1)
    .when(pl.col("prop_loss_name").is_null())
    .then(None)
    .otherwise(0)
    .alias("looted_remains")
)
lootedpropertyCDE = lootedpropertyCDE.group_by("incident_id").agg(pl.col("looted_remains").sum())

murderdata = (
    murderdata
    .drop(["prop_loss_name", "prop_loss_id", "stolen_count", "property_value", "date_recovered", "prop_desc_name"])
    .join(destroyedpropertyCDE, on="incident_id", how="left")
    .join(seizedpropertyCDE, on="incident_id", how="left")
    .join(lootedpropertyCDE, on="incident_id", how="left")
    .unique()
)

# ---------------------------------------------------------------------------
# 3. Weapon CDEs
# ---------------------------------------------------------------------------
print("Building weapon CDEs …")

def makedummycol(existingcolumn: str, inclusion_criteria: list, dummy_col_name: str):
    """Return a CDE (incident_id + dummy column) from murderdata."""
    nameCDE = murderdata.select(pl.col(["incident_id", existingcolumn])).unique()
    nameCDE = nameCDE.with_columns(
        pl.when(pl.col(existingcolumn).is_in(inclusion_criteria))
        .then(1)
        .when(pl.col(existingcolumn).is_null())
        .then(None)
        .otherwise(0)
        .alias(dummy_col_name)
    )
    nameCDE = nameCDE.group_by("incident_id").agg(pl.col(dummy_col_name).sum())
    return nameCDE


weaponsCDE = murderdata.select(pl.col(["incident_id", "weapon_name"])).unique()

gunweapCDE = weaponsCDE.with_columns(
    pl.when(
        pl.col("weapon_name").is_in(
            ["Firearm (Automatic)", "Firearm", "Handgun (Automatic)", "Rifle", "Handgun", "Other Firearm", "Shotgun"]
        )
    )
    .then(1)
    .when(pl.col("weapon_name").is_null())
    .then(None)
    .otherwise(0)
    .alias("firearms")
)
gunweapCDE = gunweapCDE.group_by("incident_id").agg(pl.col("firearms").sum())

beatingweapCDE = weaponsCDE.with_columns(
    pl.when(pl.col("weapon_name") == "Personal Weapons")
    .then(1)
    .when(pl.col("weapon_name").is_null())
    .then(None)
    .otherwise(0)
    .alias("feet_and_fists")
)
beatingweapCDE = beatingweapCDE.group_by("incident_id").agg(pl.col("feet_and_fists").sum())

bladeweapCDE    = makedummycol("weapon_name", ["Knife/Cutting Instrument"], "edged_weapon")
carweapCDE      = makedummycol("weapon_name", ["Motor Vehicle/Vessel"], "vehicle_as_weapon")
clubweapCDE     = makedummycol("weapon_name", ["Blunt Object"], "blunt_weapon")
immolationweapCDE = makedummycol("weapon_name", ["Fire/Incendiary Device"], "weaponized_fire")
unknownweapCDE  = makedummycol("weapon_name", ["Unknown"], "weapons_unknown")
strangleweapCDE = makedummycol("weapon_name", ["Asphyxiation"], "choked_or_strangled")
poisonweapCDE   = makedummycol("weapon_name", ["Drugs/Narcotics/Sleeping Pills", "Poison"], "poison")
otherweapCDE    = makedummycol("weapon_name", ["Other"], "other_weapon")

weaplist = [
    gunweapCDE, beatingweapCDE, bladeweapCDE, carweapCDE, clubweapCDE,
    immolationweapCDE, unknownweapCDE, strangleweapCDE, poisonweapCDE, otherweapCDE,
]
for cde in weaplist:
    murderdata = murderdata.join(cde, on="incident_id", how="left").unique()

murderdata = murderdata.drop(["weapon_name", "weapon_code"]).unique()

# ---------------------------------------------------------------------------
# 4. Suspect-using CDEs
# ---------------------------------------------------------------------------
print("Building suspect-using CDEs …")

hackersuspectCDE = makedummycol("suspect_using_name", ["Computer Equipment (Handheld Devices)"], "suspect_using_phone")
drunksuspectCDE  = makedummycol("suspect_using_name", ["Alcohol"], "suspect_drinking_alc")
highsuspectCDE   = makedummycol("suspect_using_name", ["Drugs/Narcotics"], "suspect_illegal_drug")

for cde in [hackersuspectCDE, drunksuspectCDE, highsuspectCDE]:
    murderdata = murderdata.join(cde, on="incident_id", how="left").unique()

murderdata = murderdata.drop(["suspect_using_name"]).unique()

# ---------------------------------------------------------------------------
# 5. Column cleanup & date engineering
# ---------------------------------------------------------------------------
print("Cleaning columns …")

murderdata = murderdata.drop(
    [
        "location_id", "agency_id", "cleared_except_id", "yearly_agency_id",
        "data_year", "ucr_agency_name", "ncic_agency_name", "pub_agency_name",
        "pub_agency_unit", "population_group_code", "cleared_except_name",
        "submission_date", "race_id", "ethnicity_id", "cargo_theft_flag",
    ]
)

murderdata = murderdata.with_columns(
    pl.when(pl.col("report_date_flag") == "f")
    .then(pl.col("incident_date").dt.weekday())
    .otherwise(None)
    .alias("incident_weekday"),
    pl.when(pl.col("report_date_flag") == "f")
    .then(pl.col("incident_date").dt.day())
    .otherwise(None)
    .alias("incident_monthday"),
    pl.when(pl.col("report_date_flag") == "f")
    .then(pl.col("incident_date").dt.month())
    .otherwise(None)
    .alias("incident_month"),
    pl.when(pl.col("report_date_flag") == "f")
    .then(pl.col("incident_date").dt.ordinal_day())
    .otherwise(None)
    .alias("incident_yearday"),
)

murderdata = murderdata.with_columns(
    pl.when(pl.col("report_date_flag") == "f")
    .then(pl.col("incident_year"))
    .otherwise(None)
    .alias("incident_year"),
    pl.when(pl.col("report_date_flag") == "f")
    .then(pl.col("incident_hour"))
    .otherwise(None)
    .alias("incident_hour"),
)

murderdata = murderdata.with_columns(
    pl.when(pl.col("report_date_flag") == "f").then(0).otherwise(1).alias("report_date_flag")
)

murderdata = murderdata.with_columns(
    pl.when(pl.col("population") == 0).then(None).otherwise(pl.col("population")).alias("population"),
    pl.when(pl.col("officer_rate") == 0).then(None).otherwise(pl.col("officer_rate")).alias("officer_rate"),
    pl.when(pl.col("employee_rate") == 0).then(None).otherwise(pl.col("employee_rate")).alias("employee_rate"),
)

# Age lookup table
age_id_to_age_dec = pl.DataFrame(
    {
        "age_id": [i for i in range(1, 105)],
        "age_decimal": [0.5 / 365, 3.5 / 365, 178.5 / 365] + [i for i in range(1, 100)] + [None, None],
    }
)

murderdata = (
    murderdata
    .join(age_id_to_age_dec, on="age_id", how="left")
    .join(age_id_to_age_dec, left_on="klr_age_id", right_on="age_id", how="left", suffix="_klr")
)

# Victim age status
murderdata = murderdata.with_columns(
    pl.when(
        (pl.col("age_id") == 103)
        | ((pl.col("age_range_low_num") == 1) & (pl.col("age_range_high_num") == 99))
    )
    .then(pl.lit("unknown"))
    .when(pl.col("age_range_low_num").is_not_null() & pl.col("age_range_high_num").is_not_null())
    .then(pl.lit("range_known"))
    .when(
        ((pl.col("age_range_low_num").is_not_null()) & (pl.col("age_range_high_num").is_null()))
        | (pl.col("age_id") <= 3)
    )
    .then(pl.lit("age_known"))
    .when(pl.col("age_id") == 104)
    .then(pl.lit("age_not_specified"))
    .otherwise(None)
    .alias("victim_age_status")
)

# Killer age status
murderdata = murderdata.with_columns(
    pl.when(
        (pl.col("klr_age_id") == 103)
        | ((pl.col("klr_age_range_low_num") == 1) & (pl.col("klr_age_range_high_num") == 99))
    )
    .then(pl.lit("age_unknown"))
    .when(pl.col("klr_age_range_low_num").is_not_null() & pl.col("klr_age_range_high_num").is_not_null())
    .then(pl.lit("range_known"))
    .when(
        ((pl.col("klr_age_range_low_num").is_not_null()) & (pl.col("klr_age_range_high_num").is_null()))
        | (pl.col("klr_age_id") <= 3)
    )
    .then(pl.lit("age_known"))
    .when(pl.col("klr_age_id") == 104)
    .then(pl.lit("age_not_specified"))
    .otherwise(None)
    .alias("klr_age_status")
)

# Combine state + county to avoid cross-state name collisions
murderdata = murderdata.with_columns(
    pl.concat_str(pl.col("state_abbr"), pl.col("county_name"), separator="_").alias("county_name")
)

# Victim / killer counts per incident
murderdata = murderdata.with_columns(
    num_of_victims=pl.col("victim_id").n_unique().over("incident_id"),
    num_of_killers=pl.col("offender_id").n_unique().over("incident_id"),
)

# ---------------------------------------------------------------------------
# 6. Select columns for sklearn
# ---------------------------------------------------------------------------
print("Selecting model columns …")

murderdata_cleaned = murderdata.select(
    [
        "incident_id", "incident_year", "incident_month", "incident_monthday",
        "incident_hour", "incident_weekday", "incident_yearday", "report_date_flag",
        "state_abbr", "agency_type_name", "population", "suburban_area_flag",
        "population_group_desc", "pop_sort_order", "male_officer", "male_civilian",
        "male_total", "female_officer", "female_civilian", "female_total",
        "officer_rate", "employee_rate", "county_name", "msa_name", "sex_code",
        "resident_status_code", "age_range_low_num", "age_range_high_num",
        "age_decimal", "victim_age_status", "ethnicity_name", "race_desc",
        "victim_type_name", "bias_category", "bias_desc", "criminal_act_name",
        "klr_sex_code", "klr_age_range_low_num", "klr_age_range_high_num",
        "age_decimal_klr", "klr_age_status", "ethnicity_name_klr", "race_desc_klr",
        "location_name", "relationship_name", "prop_destruction", "taken_by_cops",
        "looted_remains", "firearms", "feet_and_fists", "edged_weapon",
        "vehicle_as_weapon", "blunt_weapon", "weaponized_fire", "weapons_unknown",
        "choked_or_strangled", "poison", "other_weapon", "suspect_using_phone",
        "suspect_drinking_alc", "suspect_illegal_drug", "num_of_victims",
        "num_of_killers", "scot_free",
    ]
)

# ---------------------------------------------------------------------------
# 7. Add is_null indicator columns
# ---------------------------------------------------------------------------
print("Adding null-indicator columns …")

null_columns = [col for col in murderdata_cleaned.columns if murderdata_cleaned[col].has_nulls()]
already_docced = (
    [item for item in null_columns if "incident_" in item]
    + [item for item in null_columns if "age_" in item]
)
for ncol in already_docced:
    null_columns.remove(ncol)

newcols = []
for coln in null_columns:
    murderdata_cleaned = murderdata_cleaned.with_columns(
        pl.when(pl.col(coln).is_null()).then(1).otherwise(0).alias(f"{coln}_is_null")
    )
    newcols.append(f"{coln}_is_null")

murderdata_cleaned = murderdata_cleaned.select(
    [
        "incident_id", "incident_year", "incident_month", "incident_monthday",
        "incident_hour", "incident_weekday", "incident_yearday", "report_date_flag",
        "state_abbr", "agency_type_name", "population", "suburban_area_flag",
        "population_group_desc", "pop_sort_order", "male_officer", "male_civilian",
        "male_total", "female_officer", "female_civilian", "female_total",
        "officer_rate", "employee_rate", "county_name", "msa_name", "sex_code",
        "resident_status_code", "age_range_low_num", "age_range_high_num",
        "age_decimal", "victim_age_status", "ethnicity_name", "race_desc",
        "victim_type_name", "bias_category", "bias_desc", "criminal_act_name",
        "klr_sex_code", "klr_age_range_low_num", "klr_age_range_high_num",
        "age_decimal_klr", "klr_age_status", "ethnicity_name_klr", "race_desc_klr",
        "relationship_name", "location_name", "prop_destruction", "taken_by_cops",
        "looted_remains", "firearms", "feet_and_fists", "edged_weapon",
        "vehicle_as_weapon", "blunt_weapon", "weaponized_fire", "weapons_unknown",
        "choked_or_strangled", "poison", "other_weapon", "suspect_using_phone",
        "suspect_drinking_alc", "suspect_illegal_drug", "num_of_victims",
        "num_of_killers",
    ]
    + newcols
    + ["scot_free"]
)

# ---------------------------------------------------------------------------
# 8. Drop leaky / post-hoc columns  (same as notebook cell 58c4e55b)
# ---------------------------------------------------------------------------
print("Dropping leaky columns …")

murderdata_cleaned = murderdata_cleaned.drop(
    [
        "bias_category", "bias_desc", "klr_sex_code", "klr_age_range_low_num",
        "klr_age_range_high_num", "age_decimal_klr", "klr_age_status",
        "ethnicity_name_klr", "race_desc_klr", "relationship_name",
        "relationship_name_is_null", "suspect_using_phone", "suspect_drinking_alc",
        "suspect_illegal_drug",
    ]
)

# ---------------------------------------------------------------------------
# 9. Build X / y and train/holdout split
# ---------------------------------------------------------------------------
print("Splitting data …")

X = murderdata_cleaned.drop("scot_free", "incident_id")
y = murderdata_cleaned["scot_free"]
inc_groups = murderdata_cleaned["incident_id"]

cate_cols = [col for col in X.columns if X[col].dtype == pl.datatypes.String]
# (not used directly below but kept for reference)

train_indices, holdout_indices = next(
    GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=1).split(X, y, inc_groups)
)
inc_groups_train = inc_groups[train_indices]

# ---------------------------------------------------------------------------
# 10. Drop the 13 low-importance columns identified in the notebook
#     (columns_pruned[:13] from the permutation-importance pruning loop)
# ---------------------------------------------------------------------------
COLUMNS_TO_DROP = [
    "age_range_low_num",
    "age_decimal",
    "edged_weapon",
    "location_name",
    "firearms",
    "ethnicity_name",
    "resident_status_code",
    "incident_weekday",
    "msa_name",
    "incident_month",
    "race_desc",
    "incident_hour",
    "resident_status_code_is_null",
]

X_thin = X.drop(COLUMNS_TO_DROP)
X_train_thin = X_thin[train_indices]
y_train = y[train_indices]
y_holdout = y[holdout_indices]

cate_cols_thin = [col for col in X_thin.columns if X_thin[col].dtype == pl.datatypes.String]
nume_cols_thin = [col for col in X_thin.columns if col not in cate_cols_thin]

# ---------------------------------------------------------------------------
# 11. Build pipeline
# ---------------------------------------------------------------------------
print("Building pipeline …")

nume_trans = Pipeline(steps=[("num_impute", SimpleImputer(strategy="mean"))])

cate_trans = Pipeline(
    steps=[
        ("cat_impute", SimpleImputer(strategy="constant", fill_value="null_value")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num_prepro", nume_trans, nume_cols_thin),
        ("cat_prepro", cate_trans, cate_cols_thin),
    ]
)

rforest_cal = CalibratedClassifierCV(
    RandomForestClassifier(n_jobs=-1),
    cv=5,
    method="isotonic",
    ensemble=True,
    n_jobs=-1,
)
rforest_pipe = make_pipeline(preprocessor, rforest_cal)

# ---------------------------------------------------------------------------
# 12. GridSearchCV with best hyperparameters from the notebook
#     (thin_grid from notebook cell 4520f722)
# ---------------------------------------------------------------------------
print("Running GridSearchCV …  (this may take a while)")

thin_grid = {
    "calibratedclassifiercv__estimator__criterion": ["gini"],
    "calibratedclassifiercv__estimator__max_depth": [15, 20],
    "calibratedclassifiercv__estimator__max_features": ["sqrt", 10, 7],
    "calibratedclassifiercv__estimator__min_samples_split": [10],
    "calibratedclassifiercv__estimator__n_estimators": [500, 700],
    "calibratedclassifiercv__method": ["sigmoid"],
    "columntransformer__num_prepro__num_impute__strategy": ["median"],
}

group_kf = GroupKFold(n_splits=4)

thinsearch = GridSearchCV(
    rforest_pipe,
    thin_grid,
    scoring="d2_brier_score",
    cv=group_kf,
).fit(X_train_thin, y_train, groups=inc_groups_train)

print(f"Best params: {thinsearch.best_params_}")
print(f"Best score:  {thinsearch.best_score_}")

# ---------------------------------------------------------------------------
# 13. Save best_estimator_ (the pipeline, not the search object)
#     This matches what the notebook saves:
#         thinmodel = thinsearch.best_estimator_
#         joblib.dump(thinmodel, "scotfreerandomforestmodel.joblib")
# ---------------------------------------------------------------------------
thinmodel = thinsearch.best_estimator_

out_path = os.path.normpath(MODEL_OUT)
print(f"Saving model to {out_path} …")
joblib.dump(thinmodel, out_path, protocol=5)
print("Done.")
