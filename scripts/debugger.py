import pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Path to your data directory
DATA_DIR = Path("/dtu-compute/02456-p4-e24/data/data_fft-512_tscropwidth-150-200_vrcropwidth-60-15/train")

def get_sample_shapes(data_dir):
    """
    Reads all .pkl files in the directory and logs the shapes of the 'samples' arrays.
    """
    shapes = {}
    for file in data_dir.glob("*_timeseries.pkl"):
        try:
            with open(file, 'rb') as f:
                data = pickle.load(f)
                if isinstance(data, dict) and 'samples' in data:
                    samples = data['samples']
                    
                    # Convert list to numpy array if necessary
                    if isinstance(samples, list):
                        samples = np.array(samples)
                    
                    # Ensure the data is in the correct format
                    if np.iscomplexobj(samples):
                        samples = np.stack([samples.real, samples.imag], axis=-1)
                    elif samples.ndim != 2:  # If not a 2D array
                        raise ValueError(f"Unexpected shape {samples.shape} in file {file.name}")
                    
                    shapes[file.name] = samples.shape
                else:
                    print(f"File {file.name} does not contain a 'samples' key or is not a valid dictionary.")
        except Exception as e:
            print(f"Error reading file {file.name}: {e}")
    return shapes

def visualize_shapes(shapes):
    """
    Visualizes the shapes of the 'samples' arrays.
    """
    file_names = list(shapes.keys())
    lengths = [shape[0] for shape in shapes.values()]
    feature_dims = [shape[1] if len(shape) > 1 else 0 for shape in shapes.values()]

    # Plot sequence lengths
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(file_names)), lengths, color='blue', alpha=0.7)
    plt.xticks(range(len(file_names)), file_names, rotation=90, fontsize=8)
    plt.xlabel("File Names")
    plt.ylabel("Sequence Length")
    plt.title("Sequence Lengths of Samples in .pkl Files")
    plt.tight_layout()
    plt.show()

    # Plot feature dimensions
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(file_names)), feature_dims, color='orange', alpha=0.7)
    plt.xticks(range(len(file_names)), file_names, rotation=90, fontsize=8)
    plt.xlabel("File Names")
    plt.ylabel("Feature Dimensions")
    plt.title("Feature Dimensions of Samples in .pkl Files")
    plt.tight_layout()
    plt.show()

# Main Script
if __name__ == "__main__":
    shapes = get_sample_shapes(DATA_DIR)
    for file_name, shape in shapes.items():
        print(f"{file_name}: {shape}")
    visualize_shapes(shapes)
