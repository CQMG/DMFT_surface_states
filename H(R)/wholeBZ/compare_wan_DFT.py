import numpy as np
import matplotlib
#matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
# Function to parse the .dat file with multiple blocks
def parse_dat_file(filename):
    """Reads and parses data from the .dat file containing multiple blocks of bands separated by blank lines."""
    with open(filename, 'r') as file:
        lines = file.readlines()

    # Split the file into blocks by blank lines
    blocks = []
    temp_block = []

    for line in lines:
        # Skip empty lines
        if line.strip() == "":
            if temp_block:  # If a block is collected, add it to blocks
                blocks.append(np.array(temp_block))
                temp_block = []  # Reset for the next block
        else:
            temp_block.append([float(value) for value in line.split()])

    # Append the last block if the file does not end with a blank line
    if temp_block:
        blocks.append(np.array(temp_block))

    return blocks

# Parse the `.dat` file for multiple blocks of data
file3 = 'wan_band.dat'
blocks = parse_dat_file(file3)

# Read and parse the FeSe.spaghetti_ene file
with open('FeSe.spaghetti_ene', 'r') as file:
    lines = file.readlines()

# Initialize lists to store data
band_indices = []
data = []
temp_band = []

# Parse the data
for line in lines:
    if line.startswith('  bandindex:'):
        if temp_band:
            data.append(temp_band)
            temp_band = []
        band_indices.append(int(line.split()[1]))
    elif not line.startswith('band index'):
        temp_band.append([float(value) for value in line.split()])

# Add the last collected band data
if temp_band:
    data.append(temp_band)

# Convert data to NumPy arrays
data = [np.array(band) for band in data]

# Define plot parameters
plt.figure(figsize=(12, 8))

# Plot the bands from the FeSe.spaghetti_ene file
lower_bound = -6.0
upper_bound = 2.3

for band, band_index in zip(data, band_indices):
    band_data = np.array(band)
    # Filter data within the range for the 5th column
    filtered_data = band_data[(band_data[:, 4] >= lower_bound) & (band_data[:, 4] <= upper_bound)]

    if len(filtered_data) > 0:
        # Plot the raw data without smoothing or scaling
        scaling=2.85/1.5
        plt.plot(filtered_data[:, 3]*scaling, filtered_data[:, 4], label=f"FeSe Band {band_index}")

# Plot the bands from the .dat file (test_wann_band.dat)
for i, block in enumerate(blocks, 1):
    x = block[:, 0]  # k-points or another parameter
    y = block[:, 1]  # Energy or band values
    plt.plot(x, y, 'o', label=f"Band {i} (test_wann_band)", color='black', markersize=1, alpha=0.7)

# Set y-axis range from -6.0 to 2.3
plt.ylim(-6.0, 2.3)

# Add labels and title
plt.xlabel("Scaled $k_x$")
plt.ylabel("Energy (eV)")
plt.title("Comparison of Bands: FeSe vs. test_wann_band")
#plt.legend()
plt.grid(True)


# Display the plot
plt.savefig("plot.pdf", format="pdf", dpi=300, bbox_inches='tight')
plt.show()