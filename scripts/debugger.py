import pickle
from pathlib import Path

# Path to the specific .pkl file
file_path = Path("/dtu-compute/02456-p4-e24/data/data_fft-512_tscropwidth-150-200_vrcropwidth-60-15/train/136293_timeseries.pkl")

# Load the .pkl file and inspect its structure
with open(file_path, 'rb') as f:
    data = pickle.load(f)
    print(f"Content of the file {file_path}:")
    print(data)
    if isinstance(data, dict):
        print(f"Keys in the file: {list(data.keys())}")
    else:
        print(f"The file content is not a dictionary. Type: {type(data)}")
