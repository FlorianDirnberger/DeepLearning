#%%
# import datetime
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
    
class Prepro(object): # first 4 channels are power and last two are phase
    def __init__(self):
        # Define power and phase channels
        self.power_channels = [0, 1, 2, 3]  # Assuming channels 0-3 are power channels
        

    def __call__(self, spectrogram: np.ndarray) -> np.ndarray:
        # Copy the spectrogram to avoid modifying the original data
        spectrogram_processed = spectrogram.copy()
       
        # Apply Sobel filter with ksize=31 on x-axis only to power channels
        for ch in self.power_channels:
            # Get the channel data
            channel_data = spectrogram_processed[:, :, ch].astype(np.float64)

            # Apply Sobel filter in the x-direction
            sobelx = cv2.Sobel(channel_data, cv2.CV_64F, dx=1, dy=0, ksize=31)

            # Compute the gradient magnitude
            sobelx = np.sqrt(sobelx**2)

            # Replace the channel data with the Sobel filtered data
            spectrogram_processed[:, :, ch] = cv2.normalize(sobelx, None, 0, 1, cv2.NORM_MINMAX)

        # Return the processed spectrogram
        return spectrogram_processed



class NormalizeSpectrogram(object):
    phase_spectrogram_limits = (-np.pi, np.pi)

    def __call__(self, spectrogram: np.ndarray) -> np.ndarray:

        power_spectrogram_limits = np.min(spectrogram[:,:,:4]), np.max(spectrogram[:,:,:4])
        spectrogram[:,:,:4] -= power_spectrogram_limits[0]
        spectrogram[:,:,:4] /= power_spectrogram_limits[1] - power_spectrogram_limits[0]

        spectrogram[:,:,4:] -= self.phase_spectrogram_limits[0]
        spectrogram[:,:,4:] /= self.phase_spectrogram_limits[1] - self.phase_spectrogram_limits[0]

        return spectrogram

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
    


