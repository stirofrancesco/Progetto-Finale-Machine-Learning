import gdown
import zipfile
import os

url = "https://drive.google.com/uc?id=16TT7jlbubYNJ-bfIkR3qcpMQ_JpzQok3"  
zip_path = "dataset.zip"
extract_folder = "dataset"

gdown.download(url, zip_path, quiet=False)

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_folder)

os.remove(zip_path)

print(f"Dataset estratto in '{extract_folder}' e file ZIP rimosso.")
