import cv2
import numpy as np
from scipy.signal import find_peaks
from pathlib import Path
import cv2
import numpy as np
import torch
from torchvision.transforms.functional import resize



class LoadSpectrogram(object):
    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir

    def __call__(self, stmf_row) -> np.ndarray:
        obs_no = stmf_row.name
        spectrogram_filepath = self.root_dir / f"{obs_no}_stacked_spectrograms.npy"

        return np.load(spectrogram_filepath)

def preprocess_spectrogram(spectrogram_np):
    # Step 1: Normalize the spectrogram
    spectrogram_normalized = cv2.normalize(spectrogram_np, None, 0, 1, cv2.NORM_MINMAX)

    # Step 2: Compute the mean and standard deviation
    mean_value = np.mean(spectrogram_normalized)
    std_value = np.std(spectrogram_normalized)

    # Step 3: Define the target range based on 60% of the maximum and one standard deviation
    max_value = np.max(spectrogram_normalized)
    target_value = max_value * 0.70  # 60% of the maximum value
    lower_bound = target_value + std_value  # Lower limit
    # Step 4: Replace values outside the range with zero
    spectrogram_thresholded = np.where(
        (spectrogram_normalized >= lower_bound),
        spectrogram_normalized,
        0
    )

    # Step 5: Apply Gaussian Blur for noise reduction
    smoothed_spectrogram = cv2.GaussianBlur(spectrogram_thresholded, (3, 3), sigmaX=1)

    return smoothed_spectrogram

def extract_spectrogram_features(spectrogram_np):
    """
    Extract features from a spectrogram.

    Args:
        spectrogram_np (np.ndarray): Input spectrogram as a NumPy array must be normalized.

    Returns:
        list: Feature vector for the spectrogram.
    """


    # Step 1: Threshold the spectrogram to create a binary image
    _, thresholded = cv2.threshold((spectrogram_np * 255).astype(np.uint8), 20, 255, cv2.THRESH_BINARY)

    # Step 2: Find contours in the binary image
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

   

    # Step 2: Largest contour features
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        largest_contour_area = cv2.contourArea(largest_contour)
        x, y, w, h = cv2.boundingRect(largest_contour)
        elongation = max(w, h) / min(w, h) if min(w, h) != 0 else 0  # Avoid division by zero
        moment = cv2.moments(largest_contour)
        if moment["m00"] != 0:  # Avoid division by zero
            largest_cx = moment["m10"] / moment["m00"]
            largest_cy = moment["m01"] / moment["m00"]
            angle = 0.5 * np.arctan2(2 * moment["mu11"], moment["mu20"] - moment["mu02"])
        else:
            largest_cx, largest_cy, angle = 0, 0, 0
    else:
        largest_contour_area, largest_cx, largest_cy, elongation, angle = 0, 0, 0, 0, 0


    # Step 3: Contour-based features
    area_threshold = 0.10 * largest_contour_area
    contours = [c for c in contours if cv2.contourArea(c) >= area_threshold]
    areas = [cv2.contourArea(contour) for contour in contours]
    mean_area = np.mean(areas) if areas else 0
    std_area = np.std(areas) if areas else 0
    num_contours = len(contours)

    # Step 4: Peak-related features
    flat_spectrogram = spectrogram_np.flatten()
    peaks, _ = find_peaks(flat_spectrogram, height=0.95)  # Adjust height threshold as needed
    num_peaks = len(peaks)
    max_peak_idx = np.argmax(flat_spectrogram)  # Index of the largest peak
    peak_x = max_peak_idx % spectrogram_np.shape[1]  # Convert 1D index to 2D coordinates
    peak_y = max_peak_idx // spectrogram_np.shape[1]

    # Step 5: Intensity-based features
    mean_intensity = np.mean(spectrogram_np)
    std_intensity = np.std(spectrogram_np)

    # Combine all features into a single vector
    feature_vector = [
        mean_area, std_area,            # Contour statistics
        num_contours,                   # Number of contours
        largest_contour_area, largest_cx, largest_cy,  # Largest contour
        peak_x, peak_y,                 # Largest peak location
        elongation, angle,              # Shape features
        num_peaks, mean_intensity, std_intensity  # Intensity features
    ]
    feature_vector = [float(x) for x in feature_vector]
    return feature_vector

class ProcessSpectrogram:
    """
    Wrapper for process_spectrogram to use in a transformation pipeline.
    """
    def __call__(self, spectrogram_tensor):
        # Call the existing function and return its output
        return process_spectrogram(spectrogram_tensor)
        
def process_spectrogram(spectrogram_tensor):
    """
    Process a spectrogram with multiple channels (H x W x C) and extract features from
    the 4 power channels.

    Args:
        spectrogram (np.ndarray): Input spectrogram with shape (H, W, C).

    Returns:
        list: Combined feature vector for the 4 power channels.
    """

    # Convert the tensor to a NumPy array for preprocessing
    spectrogram = spectrogram_tensor.numpy()

    # Check that the spectrogram has 6 channels
    assert spectrogram.shape[-1] == 6, "Spectrogram must have 6 channels (4 power, 2 phase)."

    # Initialize the combined feature vector
    combined_feature_vector = []

    # Process each of the 4 power channels
    for channel_idx in range(4):  # Channels 0 to 3 are power channels
        # Extract the current power channel
        power_channel = spectrogram[:, :, channel_idx]

        # Preprocess the spectrogram
        preprocessed_channel = preprocess_spectrogram(power_channel)

        # Extract features from the preprocessed channel
        feature_vector = extract_spectrogram_features(preprocessed_channel)

        # Append the features to the combined vector
        combined_feature_vector.extend(feature_vector)

    # Convert the feature vector to a PyTorch tensor
    return torch.tensor(feature_vector, dtype=torch.float32)

class InterpolateSpectrogram(object):
    def __init__(self, size: tuple[int, int] = (74, 918)) -> None:
        self.size = size

    def __call__(self, spectrogram: torch.Tensor) -> torch.Tensor:
        return resize(img=spectrogram, size=self.size)

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, spectrogram):
        # swap channel axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        spectrogram = spectrogram.transpose((2, 0, 1))
        return torch.from_numpy(spectrogram.astype(np.float32))