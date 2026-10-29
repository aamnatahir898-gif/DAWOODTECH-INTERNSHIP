import argparse
import re
import sys
from pathlib import Path
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REFERENCE_YEAR = 2022


REFERENCE_YEAR = 2022

REQUIRED_COLUMNS = [
    "price",
    "mileage",
    "engine_capacity",
    "model_year",
]

ID_COLUMNS = ["Unnamed: 0", "ad_url"]

TEXT_COLUMNS = [
    "location",
    "engine_type",
    "transmission",
    "registered_in",
    "color",
    "assembly",
    "body_type",
]

cleaning_log = []


def log_step(step, before, after, note=""):
    cleaning_log.append({
        "step": step,
        "rows_before": before,
        "rows_after": after,
        "rows_removed": before - after,
        "note": note,
    })

    print(
        f"[{step}] {before:,} -> {after:,} rows "
        f"({before - after:,} removed)"
    )


def default_project_path():
    colab_path = Path("/content/drive/MyDrive/PriceSense")

    if colab_path.exists():
        return colab_path

    try:
        return Path(__file__).resolve().parent.parent
    except NameError:
        return Path.cwd()


def load_raw(raw_path):
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found: {raw_path}"
        )

    last_error = None

    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            df = pd.read_csv(
                raw_path,
                encoding=encoding,
                low_memory=False
            )

            print(
                f"Raw dataset loaded using {encoding}: "
                f"{df.shape[0]:,} rows x {df.shape[1]} columns"
            )

            return df

        except UnicodeDecodeError as err:
            last_error = err

    raise ValueError(
        f"Could not read CSV using supported encodings: {last_error}"
    )


def check_columns(df):
    missing = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing:
        raise KeyError(
            f"Required columns are missing: {missing}"
        )


def convert_price_to_pkr(value):
    if pd.isna(value):
        return np.nan

    text = str(value).strip().lower().replace(",", "")

    if not text or "call for price" in text:
        return np.nan

    text = re.sub(r"\bpkr\b", "", text).strip()

    match = re.search(
        r"^\s*(\d+(?:\.\d+)?)\s*(.*?)\s*$",
        text
    )

    if not match:
        return np.nan

    amount = float(match.group(1))
    unit = match.group(2).strip()

    if re.search(r"\b(crore|crores|cr)\b", unit):
        return amount * 10_000_000

    if re.search(r"\b(lac|lacs|lakh|lakhs)\b", unit):
        return amount * 100_000

    if unit == "":
        return amount

    return np.nan

def parse_numeric_columns(df):
    df = df.copy()

    df["price_pkr"] = df["price"].apply(
        convert_price_to_pkr
    )

    df["mileage_km"] = pd.to_numeric(
        df["mileage"]
        .astype("string")
        .str.replace(",", "", regex=False)
        .str.replace(r"(?i)\s*km\s*", "", regex=True)
        .str.strip(),
        errors="coerce"
    )

    df["engine_capacity_cc"] = pd.to_numeric(
        df["engine_capacity"]
        .astype("string")
        .str.replace(",", "", regex=False)
        .str.replace(r"(?i)\s*cc\s*", "", regex=True)
        .str.strip(),
        errors="coerce"
    )
    cc = df["engine_capacity_cc"]

    bad_cc = cc.notna() & ~cc.between(500, 6000)

    print(
        "Invalid engine capacities changed to missing:",
        int(bad_cc.sum())
    )

    df.loc[bad_cc, "engine_capacity_cc"] = np.nan

    bad_mileage = (
        df["mileage_km"].notna()
        & (df["mileage_km"] <= 0)
    )

    print(
        "Non-positive mileage changed to missing:",
        int(bad_mileage.sum())
    )

    df.loc[bad_mileage, "mileage_km"] = np.nan


    for col in TEXT_COLUMNS:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype("string")
                .str.strip()
                .replace("", pd.NA)
            )

    print(
        f"Usable prices: {df['price_pkr'].notna().sum():,}"
    )
    print(
        f"Usable mileage values: {df['mileage_km'].notna().sum():,}"
    )
    print(
        "Usable engine capacities:",
        f"{df['engine_capacity_cc'].notna().sum():,}"
    )

    return df


def add_model_year_and_age(df):
    df = df.copy()

    year = pd.to_numeric(
        df["model_year"],
        errors="coerce"
    )

    bad_year = (
        year.notna()
        & ~year.between(1990, REFERENCE_YEAR + 1)
    )

    print(
        "Invalid model years changed to missing:",
        int(bad_year.sum())
    )

    year = year.where(~bad_year)

    df["model_year_clean"] = year.round().astype("Int64")

    df["car_age"] = (
        REFERENCE_YEAR - df["model_year_clean"].astype("float")
    ).clip(lower=0)


    denominator = df["car_age"].where(df["car_age"] > 0)

    df["mileage_per_year"] = df["mileage_km"] / denominator

    print(
        "Missing model years:",
        f"{df['model_year_clean'].isna().sum():,}"
    )

    print(
        "Missing car ages:",
        f"{df['car_age'].isna().sum():,}"
    )

    return df



def drop_missing_price(df):
    before = len(df)

    df = df[
        df["price_pkr"].notna()
        & (df["price_pkr"] > 0)
    ].copy()

    log_step(
        "Drop missing/invalid prices",
        before,
        len(df),
        "Rows without a valid positive target price cannot be "
        "used for supervised price prediction."
    )

    return df

def drop_duplicates(df, original_columns):

    dup_cols = [
        col for col in original_columns
        if col not in ID_COLUMNS
    ]

    before = len(df)

    df = df.drop_duplicates(
        subset=dup_cols,
        keep="first"
    ).copy()

    log_step(
        "Drop duplicate listings",
        before,
        len(df),
        "Exact duplicate listing content checked while excluding "
        "row ID and URL."
    )

    return df

def iqr_bounds(values, k=1.5):
    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1

    return (
        q1 - k * iqr,
        q3 + k * iqr
    )


def flag_iqr_outliers(df, k=1.5):


    df = df.copy()

    audit_rows = []
    flagged_masks = {}

    for col in ["price_pkr", "mileage_km"]:
        values = df[col].dropna()

        if values.empty:
            df[f"{col}_iqr_outlier"] = False
            continue

        log_values = np.log1p(values)

        low, high = iqr_bounds(log_values, k)

        full_log_values = np.log1p(df[col])

        mask = (
            df[col].notna()
            & (
                (full_log_values < low)
                | (full_log_values > high)
            )
        )

        df[f"{col}_iqr_outlier"] = mask
        flagged_masks[col] = mask

        lower_limit = max(0, int(round(np.expm1(low))))
        upper_limit = int(round(np.expm1(high)))

        audit_rows.append({
            "column": col,
            "iqr_k": k,
            "lower_limit": lower_limit,
            "upper_limit": upper_limit,
            "flagged_rows": int(mask.sum()),
            "flagged_percent": round(
                mask.sum() / max(df[col].notna().sum(), 1) * 100,
                2
            ),
        })


    price_flag = df.get(
        "price_pkr_iqr_outlier",
        pd.Series(False, index=df.index)
    )

    mileage_flag = df.get(
        "mileage_km_iqr_outlier",
        pd.Series(False, index=df.index)
    )

    df["any_iqr_outlier"] = (
        price_flag | mileage_flag
    )

    audit_table = pd.DataFrame(audit_rows)

    print("\nIQR OUTLIER REPORT (LOG SCALE)")
    print(audit_table.to_string(index=False))

    print(
        "\nRows flagged in at least one column:",
        f"{df['any_iqr_outlier'].sum():,}"
    )

    print(
        "These rows are RETAINED in the modeling dataset."
    )

    return df, audit_table

def save_outlier_audit(df, reports_dir):
    reports_dir.mkdir(parents=True, exist_ok=True)

    audit_path = reports_dir / "iqr_outlier_audit.csv"

    flagged = df[df["any_iqr_outlier"]].copy()

    flagged.to_csv(
        audit_path,
        index=False
    )

    print(
        f"Outlier audit saved: {audit_path}"
    )

    return audit_path



def main(argv=None):
    parser = argparse.ArgumentParser(
        description="PriceSense final data cleaning"
    )

    parser.add_argument(
        "--project",
        type=Path,
        default=default_project_path()
    )

    parser.add_argument(
        "--iqr-k",
        type=float,
        default=1.5
    )

    args, _unknown = parser.parse_known_args(argv)

    if args.iqr_k <= 0:
        raise ValueError("--iqr-k must be greater than zero.")

    project = args.project

    raw_path = (
        project / "data" / "raw" / "pakwheels.csv"
    )


    cleaned_path = (
        project / "data" / "processed"
        / "pakwheels_cleaned.csv"
    )

    reports_dir = project / "reports"

    log_path = (
        reports_dir / "cleaning_log_reviewed.csv"
    )

    iqr_report_path = (
        reports_dir / "iqr_diagnostic.csv"
    )

    step = "start"

    try:
        print("PRICESENSE FINAL DATA CLEANING")

        print(f"Project folder: {project}\n")

        step = "load raw data"

        raw_df = load_raw(raw_path)

        check_columns(raw_df)

        original_columns = list(raw_df.columns)

        df = raw_df.copy()


        step = "parse numeric columns"

        df = parse_numeric_columns(df)

        step = "model year and feature engineering"

        df = add_model_year_and_age(df)

        step = "drop missing price"

        df = drop_missing_price(df)


        step = "drop duplicates"

        df = drop_duplicates(
            df,
            original_columns
        )


        step = "IQR outlier detection"

        df, iqr_table = flag_iqr_outliers(
            df,
            k=args.iqr_k
        )


        step = "save outputs"

        cleaned_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        reports_dir.mkdir(
            parents=True,
            exist_ok=True
        )


        df = df.drop(
            columns=["Unnamed: 0"],
            errors="ignore"
        )

        df.to_csv(
            cleaned_path,
            index=False
        )

        pd.DataFrame(cleaning_log).to_csv(
            log_path,
            index=False
        )

        iqr_table.to_csv(
            iqr_report_path,
            index=False
        )

        save_outlier_audit(
            df,
            reports_dir
        )

    except Exception as err:
        print(
            f"\nERROR (step: {step}): "
            f"{type(err).__name__}: {err}"
        )
        raise



    print("CLEANING COMPLETE")

    print(f"Raw dataset shape: {raw_df.shape}")
    print(f"Final dataset shape: {df.shape}")
    print(f"Rows removed overall: {len(raw_df) - len(df):,}")

    print(f"\nModeling dataset:\n{cleaned_path}")
    print(f"\nCleaning log:\n{log_path}")
    print(f"\nIQR diagnostic report:\n{iqr_report_path}")

    print("\nMissing values in important columns:")

    important_cols = [
        "price_pkr",
        "mileage_km",
        "engine_capacity_cc",
        "model_year_clean",
        "car_age",
        "mileage_per_year",
        "engine_type",
        "transmission",
        "body_type",
        "color",
    ]

    important_cols = [
        col for col in important_cols
        if col in df.columns
    ]

    print(
        df[important_cols]
        .isna()
        .sum()
        .to_string()
    )

    print("\nPrice summary (PKR):")
    print(
        df["price_pkr"]
        .describe()
        .to_string()
    )

    print("\nOutlier flags:")
    print(
        df["any_iqr_outlier"]
        .value_counts(dropna=False)
        .to_string()
    )

    if "assembly" in df.columns:
        print("\nMedian price by assembly:")
        print(
            df.groupby("assembly", dropna=False)["price_pkr"]
            .agg(["count", "median"])
            .to_string()
        )



    return df


if __name__ == "__main__":
    main()