import os
import requests
import zipfile
import io
import shutil
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Folder paths
raw_folder = "data/raw/"
os.makedirs(raw_folder, exist_ok=True)

# Download MovieLens latest-small dataset
url = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
print("Downloading MovieLens dataset...")

r = requests.get(url, verify=False)
z = zipfile.ZipFile(io.BytesIO(r.content))
z.extractall("data/movielens")
print("Dataset downloaded and extracted!")

# Detect actual folder containing CSV files
top_folder = z.namelist()[0].split('/')[0]
extract_path = os.path.join("data/movielens", top_folder)
print(f"Using extract_path: {extract_path}")

# Copy CSV files to raw folder
for file_name in ["movies.csv", "ratings.csv", "tags.csv"]:
    src = os.path.join(extract_path, file_name)
    if os.path.exists(src):
        shutil.copy(src, os.path.join(raw_folder, file_name))
        print(f"{file_name} copied to data/raw/")
    else:
        print(f"{file_name} not found in {extract_path}")

print("All done! CSV files are in data/raw/")
