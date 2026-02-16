#!/bin/bash

# Download this drive link: https://drive.google.com/drive/folders/1nyINi64mXJ-kiK79yQ4UPz7NAuX9MuhT?usp=drive_link
# and unzip the data into this folder

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRIVE_FOLDER_ID="1nyINi64mXJ-kiK79yQ4UPz7NAuX9MuhT"
OUTPUT_FILE="$SCRIPT_DIR/data.zip"

echo "Downloading data from Google Drive..."

# Using gdown to download from Google Drive
if ! command -v gdown &> /dev/null; then
    echo "gdown is required but not installed. Installing..."
    pip install gdown
fi

gdown --folder "https://drive.google.com/drive/folders/$DRIVE_FOLDER_ID" -O "$SCRIPT_DIR"

if [ ! -d "$SCRIPT_DIR" ] || [ -z "$(find "$SCRIPT_DIR" -type f -not -name 'download_data.sh' | head -1)" ]; then
    echo "Error: Download failed or no files were downloaded."
    exit 1
fi

echo "Download complete!"