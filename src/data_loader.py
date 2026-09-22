"""InsightAI Data Loader Module
Handles automated downloading, multi-sheet Excel parsing, schema normalization,
and high-speed Parquet caching for the UCI Online Retail II dataset and custom uploads.
"""

from pathlib import Path
from typing import Callable, Optional, Tuple
import os
import requests
import pandas as pd
import streamlit as st

DATA_DIR = Path("data")
EXCEL_PATH = DATA_DIR / "online_retail_II.xlsx"
PARQUET_PATH = DATA_DIR / "online_retail_II.parquet"
UCI_DOWNLOAD_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00502/online_retail_II.xlsx"

# Standard target schema for InsightAI
STANDARD_COLUMNS = [
    "Invoice",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "Price",
    "Customer ID",
    "Country"
]

COLUMN_MAPPING = {
    "invoiceno": "Invoice",
    "invoice": "Invoice",
    "stockcode": "StockCode",
    "description": "Description",
    "quantity": "Quantity",
    "invoicedate": "InvoiceDate",
    "price": "Price",
    "unitprice": "Price",
    "unit_price": "Price",
    "customer id": "Customer ID",
    "customer_id": "Customer ID",
    "customerid": "Customer ID",
    "country": "Country"
}


def ensure_data_dir() -> Path:
    """Ensures that the data directory exists."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR


def normalize_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Normalizes column names to standard InsightAI schema."""
    new_cols = {}
    for col in df.columns:
        clean_col = str(col).strip().lower()
        if clean_col in COLUMN_MAPPING:
            new_cols[col] = COLUMN_MAPPING[clean_col]
        else:
            new_cols[col] = str(col).strip()
    
    df = df.rename(columns=new_cols)
    return df


def download_uci_dataset(
    dest_path: Path = EXCEL_PATH,
    progress_callback: Optional[Callable[[float, str], None]] = None
) -> Path:
    """Downloads the official UCI Online Retail II Excel dataset with progress reporting."""
    ensure_data_dir()
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) InsightAI/1.0"}
    
    if progress_callback:
        progress_callback(0.05, "Connecting to UCI Machine Learning Repository...")
        
    response = requests.get(UCI_DOWNLOAD_URL, stream=True, headers=headers, timeout=120)
    response.raise_for_status()
    
    total_size = int(response.headers.get("content-length", 45 * 1024 * 1024))
    downloaded = 0
    chunk_size = 1024 * 1024  # 1MB chunks
    
    last_reported = 0
    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if progress_callback and (downloaded - last_reported >= 1024 * 1024 or downloaded >= total_size):
                    last_reported = downloaded
                    pct = min(0.95, downloaded / total_size)
                    progress_callback(
                        pct,
                        f"Downloading dataset: {downloaded / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB"
                    )
                    
    if progress_callback:
        progress_callback(1.0, "Download completed successfully.")
        
    return dest_path


def parse_excel_to_df(excel_path: Path) -> pd.DataFrame:
    """Parses both sheets from the UCI Online Retail II Excel file."""
    xls = pd.ExcelFile(excel_path)
    sheets = xls.sheet_names
    
    dfs = []
    for sheet in sheets:
        df_sheet = pd.read_excel(xls, sheet_name=sheet)
        df_sheet = normalize_schema(df_sheet)
        df_sheet["SourceSheet"] = sheet
        dfs.append(df_sheet)
        
    if not dfs:
        raise ValueError(f"No valid sheets found in {excel_path}")
        
    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df


def save_parquet_cache(df: pd.DataFrame, parquet_path: Path = PARQUET_PATH) -> Path:
    """Saves parsed dataset to Apache Parquet format for sub-second future loads."""
    ensure_data_dir()
    df_to_save = df.copy()
    
    # Cast all potential text/identifier columns cleanly to string to prevent PyArrow mixed-type errors
    string_cols = ["Invoice", "StockCode", "Description", "Country", "SourceSheet"]
    for col in string_cols:
        if col in df_to_save.columns:
            df_to_save[col] = df_to_save[col].fillna("").astype(str).str.strip()

    if "Customer ID" in df_to_save.columns:
        # Normalize Customer IDs to clean strings or None
        df_to_save["Customer ID"] = (
            df_to_save["Customer ID"]
            .astype(str)
            .str.replace(".0", "", regex=False)
            .str.strip()
            .replace({"nan": None, "None": None, "<NA>": None, "": None})
        )

    if "InvoiceDate" in df_to_save.columns:
        df_to_save["InvoiceDate"] = pd.to_datetime(df_to_save["InvoiceDate"], errors="coerce")

    if "Quantity" in df_to_save.columns:
        df_to_save["Quantity"] = pd.to_numeric(df_to_save["Quantity"], errors="coerce").fillna(0)
    if "Price" in df_to_save.columns:
        df_to_save["Price"] = pd.to_numeric(df_to_save["Price"], errors="coerce").fillna(0.0)

    df_to_save.to_parquet(parquet_path, engine="pyarrow", index=False)
    return parquet_path


def load_dataset(
    force_download: bool = False,
    progress_callback: Optional[Callable[[float, str], None]] = None
) -> Tuple[pd.DataFrame, str]:
    """Loads the UCI Online Retail II dataset, downloading and caching as necessary.
    
    Returns:
        Tuple[pd.DataFrame, str]: (Dataset DataFrame, Source description)
    """
    ensure_data_dir()
    
    # 1. Try Parquet Cache (Fastest: ~0.5s)
    if not force_download and PARQUET_PATH.exists():
        if progress_callback:
            progress_callback(0.5, "Loading cached Parquet dataset (1M rows)...")
        df = pd.read_parquet(PARQUET_PATH, engine="pyarrow")
        df = normalize_schema(df)
        if progress_callback:
            progress_callback(1.0, f"Loaded {len(df):,} transactions from cache.")
        return df, "Cached Parquet"

    # 2. Try Local Excel File
    if not force_download and EXCEL_PATH.exists():
        if progress_callback:
            progress_callback(0.3, "Parsing local Excel workbook (Year 2009-2010 & Year 2010-2011)...")
        df = parse_excel_to_df(EXCEL_PATH)
        if progress_callback:
            progress_callback(0.8, "Creating high-speed Parquet cache for future launches...")
        save_parquet_cache(df)
        if progress_callback:
            progress_callback(1.0, f"Processed {len(df):,} transactions.")
        return df, "Local Excel File"

    # 3. Download from UCI
    if progress_callback:
        progress_callback(0.1, "Initiating download of official UCI Online Retail II dataset...")
    download_uci_dataset(EXCEL_PATH, progress_callback=progress_callback)
    
    if progress_callback:
        progress_callback(0.6, "Parsing downloaded workbook...")
    df = parse_excel_to_df(EXCEL_PATH)
    
    if progress_callback:
        progress_callback(0.85, "Saving to high-speed Parquet cache...")
    save_parquet_cache(df)
    
    if progress_callback:
        progress_callback(1.0, f"Ready! {len(df):,} transactions loaded.")
    return df, "Official UCI Download"


def load_custom_file(uploaded_file) -> pd.DataFrame:
    """Parses a user-uploaded CSV or Excel file."""
    file_name = uploaded_file.name.lower()
    if file_name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif file_name.endswith((".xlsx", ".xls")):
        xls = pd.ExcelFile(uploaded_file)
        dfs = [pd.read_excel(xls, sheet) for sheet in xls.sheet_names]
        df = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]
    elif file_name.endswith(".parquet"):
        df = pd.read_parquet(uploaded_file)
    else:
        raise ValueError(f"Unsupported file format: {uploaded_file.name}")
        
    df = normalize_schema(df)
    return df
