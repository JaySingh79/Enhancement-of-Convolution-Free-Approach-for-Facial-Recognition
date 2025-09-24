# This script will help you download the Kaggle Kinship dataset using the Kaggle API.
# Make sure you have your Kaggle API credentials set up (kaggle.json in ~/.kaggle/ or %USERPROFILE%\.kaggle\ on Windows).

import os
import subprocess

def download_kinship_dataset():
    os.makedirs('../data', exist_ok=True)
    # Download the competition data
    subprocess.run(['kaggle', 'competitions', 'download', '-c', 'recognizing-faces-in-the-wild', '-p', '../data'])
    # TODO: Unzip the files and organize images and CSVs as needed

if __name__ == '__main__':
    download_kinship_dataset()
    print('Download complete. Please unzip and organize the files in the data/ directory.')
