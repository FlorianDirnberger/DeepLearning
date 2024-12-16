import numpy as np
import matplotlib.pyplot as plt

# Function to add salt and pepper noise
def add_salt_and_pepper_noise(data, salt_prob, pepper_prob):
    noisy_data = data.copy()
    num_elements = data.size

    # Number of salt and pepper pixels
    num_salt = int(num_elements * salt_prob)
    num_pepper = int(num_elements * pepper_prob)

    # Add salt noise
    salt_coords = (
        np.random.randint(0, data.shape[0], num_salt),
        np.random.randint(0, data.shape[1], num_salt),
        np.random.randint(0, data.shape[2], num_salt),
    )
    noisy_data[salt_coords] = 1

    # Add pepper noise
    pepper_coords = (
        np.random.randint(0, data.shape[0], num_pepper),
        np.random.randint(0, data.shape[1], num_pepper),
        np.random.randint(0, data.shape[2], num_pepper),
    )
    noisy_data[pepper_coords] = 0

    return noisy_data

# Function to generate synthetic data with specified parameters
def generate_synthetic_data():
    # Parameters for the synthetic data
    channels, height, width = 6, 74, 918
    center_height, center_width = height // 2, width // 2

    # Parameters for the slanted line
    line_length = 45  # Length of the line
    line_angle_deg = 210  # Angle of the line in degrees
    line_angle_rad = np.deg2rad(line_angle_deg)
    dy = int(line_length * np.sin(line_angle_rad))
    dx = int(line_length * np.cos(line_angle_rad))
    line_start_h = center_height - 5  # Starting height of the line
    line_start_w = center_width  # Starting width of the line

    # Calculate the coordinates of the slanted line
    line_coords_h = np.clip(
        np.arange(line_start_h, line_start_h + dy, np.sign(dy)), 0, height - 1
    ).astype(int)
    line_coords_w = np.clip(
        np.arange(line_start_w, line_start_w + dx, np.sign(dx)), 0, width - 1
    ).astype(int)

    # Parameters for the horizontal line
    horizontal_line_height = 37  # The height index where the line will be drawn
    horizontal_line_length = width  # Full width of the line

    # Peak size for thickness
    peak_size_height = 5  # Number of rows for the peak
    peak_size_width = 20  # Number of columns for the peak

    # Parameters for salt and pepper noise
    salt_prob = 0.1  # Probability of salt noise
    pepper_prob = 0.1  # Probability of pepper noise

    # Initialize synthetic data
    synthetic_data = np.zeros((channels, height, width))

    # Draw the slanted line
    for h, w in zip(line_coords_h, line_coords_w):
        start_h = max(h - peak_size_height // 2, 0)
        end_h = min(h + peak_size_height // 2 + 1, height)
        start_w = max(w - peak_size_width // 2, 0)
        end_w = min(w + peak_size_width // 2 + 1, width)
        synthetic_data[:, start_h:end_h, start_w:end_w] = 1

    # Draw the horizontal line
    start_h = max(horizontal_line_height - peak_size_height // 2, 0)
    end_h = min(horizontal_line_height + peak_size_height // 2 -1, height)
    start_w = 0
    end_w = width
    synthetic_data[:, start_h:end_h, start_w:end_w] = 1

    # Add salt and pepper noise
    noisy_synthetic_data = add_salt_and_pepper_noise(synthetic_data, salt_prob, pepper_prob)

    return noisy_synthetic_data

# Function to generate a tensor of synthetic data
def generate_synthetic_tensor(x):
    return np.stack([generate_synthetic_data() for _ in range(x)], axis=0)

# Main call to generate 10 synthetic tensors
if __name__ == "__main__":
    train_synthetic_tensor = generate_synthetic_tensor(64)
    print(f"Generated synthetic tensor with shape: {train_synthetic_tensor.shape}")
    val_synthetic_tensor = generate_synthetic_tensor(64)
    print(f"Generated synthetic tensor with shape: {val_synthetic_tensor.shape}")
   

    # TEST: Print synthetic data:
    # Remove batch dimension
    # tensor = synthetic_tensor[3,:,:,:]   # Resulting size: (6, 74, 918)

    # Plot each channel
    # for i in range(tensor.shape[0]):  # Iterate over channels
    #     plt.figure(figsize=(12, 6))  # Set figure size
    #     plt.imshow(tensor[i], cmap="viridis", aspect="auto")  # Plot channel data
    #     plt.colorbar(label="Values")  # Add a colorbar
    #     plt.title(f"Channel {i + 1}")
        
    #     # Save the plot
    #     filename = f"synthetic_data/channel_{i + 1}.png"
    #     plt.savefig(filename)
    #     print(f"Saved: {filename}")
    #     plt.show()

    # Save synthetic data in a .npy
    for i in range(train_synthetic_tensor.shape[0]):
        filename = f"synthetic_data/train/{i}_stacked_spectrograms.npy"
        print(filename)
        # print(train_synthetic_tensor[i].shape)
        array = train_synthetic_tensor[i]
        reshaped_array = np.transpose(array, (1, 2, 0))
        np.save(filename, reshaped_array)

    for i in range(val_synthetic_tensor.shape[0]):
        filename = f"synthetic_data/validation/{64+i}_stacked_spectrograms.npy"
        print(filename)
        # print(train_synthetic_tensor[i].shape)
        array = val_synthetic_tensor[i]
        reshaped_array = np.transpose(array, (1, 2, 0))
        np.save(filename, reshaped_array)

