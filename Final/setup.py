# setup.py
def ensure_dataset_ready():
    import os
    import zipfile
    import requests
    from io import BytesIO

    dataset_url = "https://www.dropbox.com/scl/fi/67ke65f78o2oz7mc65fhm/Dataset.zip?rlkey=nvjngju1o2iii3h3mo4lvh2d4&st=yiapt1ml&dl=1"
    extract_to = "Dataset"

    if os.path.exists(extract_to):
        print(f"✓ Dataset already exists at: {extract_to}")
        return

    print("↓ Downloading dataset from Dropbox...")
    headers = {"User-Agent": "Mozilla/5.0"}

    response = requests.get(dataset_url, headers=headers, stream=True)
    if response.status_code != 200:
        raise Exception(f"✗ Failed to download dataset (HTTP {response.status_code})")

    if 'html' in response.headers.get('Content-Type', '').lower():
        raise Exception("✗ Dropbox returned HTML, not a ZIP file.")

    print("✓ Extracting dataset...")
    with zipfile.ZipFile(BytesIO(response.content)) as zip_ref:
        zip_ref.extractall(extract_to)

    print(f"✓ Dataset extracted to: {extract_to}")

if __name__ == "__main__":
    ensure_dataset_ready()
