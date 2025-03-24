"""
Download multiple datasets from configured URLs or resources.
Stores them in TMP/raw/<dataset_name>/ or as specified in the config.
"""

import os
import sys
import yaml
import requests
import shutil
import zipfile
from corus import load_lenta

def load_config(config_path: str) -> dict:
    """
    Loads YAML config file into a dictionary.
    """
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        sys.exit(1)
    with open(config_path, "r", encoding="utf-8") as file:
        config_data = yaml.safe_load(file)
    return config_data

def download_file(url: str, output_path: str) -> None:
    """
    Downloads a file from the given URL to the specified path.
    """
    response = requests.get(url, stream=True)
    response.raise_for_status()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as out_file:
        for chunk in response.iter_content(chunk_size=8192):
            out_file.write(chunk)

def extract_zip(zip_path: str, extract_to: str) -> None:
    """
    Extracts a ZIP file to the given folder.
    """
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)

def download_iam(iam_url: str, iam_dir: str) -> None:
    """
    Downloads the IAM dataset from iam_url into iam_dir.
    """
    if not iam_url:
        print("IAM URL not provided. Skipping.")
        return
    zip_name = os.path.join(iam_dir, "iam_dataset.zip")
    download_file(iam_url, zip_name)
    extract_zip(zip_name, iam_dir)
    os.remove(zip_name)

def download_mlt19(mlt19_url: str, mlt19_dir: str) -> None:
    """
    Downloads the MLT19 dataset from mlt19_url into mlt19_dir.
    """
    if not mlt19_url:
        print("MLT19 URL not provided. Skipping.")
        return
    zip_name = os.path.join(mlt19_dir, "mlt19_dataset.zip")
    download_file(mlt19_url, zip_name)
    extract_zip(zip_name, mlt19_dir)
    os.remove(zip_name)

def download_smile_twitter(smile_url: str, smile_dir: str) -> None:
    """
    Downloads the Smile Twitter Emotion dataset from smile_url into smile_dir.
    """
    if not smile_url:
        print("Smile Twitter URL not provided. Skipping.")
        return
    zip_name = os.path.join(smile_dir, "smile_twitter.zip")
    download_file(smile_url, zip_name)
    extract_zip(zip_name, smile_dir)
    os.remove(zip_name)

def download_rusentiment(rusentiment_url: str, rusentiment_dir: str) -> None:
    """
    Downloads and extracts RuSentiment dataset from rusentiment_url into rusentiment_dir.
    """
    if not rusentiment_url:
        print("RuSentiment URL not provided. Skipping.")
        return
    zip_name = os.path.join(rusentiment_dir, "rusentiment.zip")
    download_file(rusentiment_url, zip_name)
    extract_zip(zip_name, rusentiment_dir)
    os.remove(zip_name)

def download_gutenberg(gutenberg_ids: list, base_url: str, gutenberg_dir: str) -> None:
    """
    Downloads text files from Project Gutenberg based on provided IDs.
    """
    if not gutenberg_ids or not base_url:
        print("No Gutenberg IDs or base URL. Skipping.")
        return
    for book_id in gutenberg_ids:
        url = base_url.replace("{book_id}", book_id)
        file_name = f"{book_id}.txt"
        output_path = os.path.join(gutenberg_dir, file_name)
        download_file(url, output_path)

def download_lenta(lenta_input_path: str, lenta_output_path: str, limit: int) -> None:
    """
    Processes the Lenta.ru dataset using corus. Writes a limited subset to lenta_output_path.
    """
    if not os.path.exists(lenta_input_path):
        print(f"Lenta file not found: {lenta_input_path}")
        return
    os.makedirs(os.path.dirname(lenta_output_path), exist_ok=True)
    count = 0
    with open(lenta_output_path, "w", encoding="utf-8") as out_file:
        for record in load_lenta(lenta_input_path):
            if count >= limit:
                break
            out_file.write(record.text + "\n\n")
            count += 1

def main(config_path: str) -> None:
    """
    Main entry: downloads or processes all required datasets based on config.
    """
    config_data = load_config(config_path)
    raw_dir = config_data.get("download_paths", {}).get("raw_data_dir", "TMP/raw")

    # IAM
    iam_url = config_data.get("datasets", {}).get("iam_url", "")
    iam_dir = os.path.join(raw_dir, "iam")
    download_iam(iam_url, iam_dir)

    # MLT19
    mlt19_url = config_data.get("datasets", {}).get("mlt19_url", "")
    mlt19_dir = os.path.join(raw_dir, "mlt19")
    download_mlt19(mlt19_url, mlt19_dir)

    # Smile Twitter
    smile_url = config_data.get("datasets", {}).get("smile_twitter_url", "")
    smile_dir = os.path.join(raw_dir, "smile_twitter")
    download_smile_twitter(smile_url, smile_dir)

    # RuSentiment
    rusentiment_url = config_data.get("datasets", {}).get("rusentiment_url", "")
    rusentiment_dir = os.path.join(raw_dir, "rusentiment")
    download_rusentiment(rusentiment_url, rusentiment_dir)

    # Gutenberg
    gutenberg_ids = config_data.get("datasets", {}).get("gutenberg_ids", [])
    gutenberg_base = config_data.get("datasets", {}).get("gutenberg_base_url", "")
    gutenberg_dir = os.path.join(raw_dir, "gutenberg")
    os.makedirs(gutenberg_dir, exist_ok=True)
    download_gutenberg(gutenberg_ids, gutenberg_base, gutenberg_dir)

    # Lenta
    lenta_input = config_data.get("datasets", {}).get("lenta_input_path", "")
    lenta_output = config_data.get("datasets", {}).get("lenta_output_path", "")
    lenta_limit = config_data.get("datasets", {}).get("lenta_limit", 10)
    if lenta_input and lenta_output:
        download_lenta(lenta_input, lenta_output, lenta_limit)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python download_data.py <config_path>")
        sys.exit(1)
    config_file = sys.argv[1]
    main(config_file)