import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def parse_hr_dat(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    num_wann = int(lines[1].strip())
    num_R = int(lines[2].strip())
    degeneracies = []
    line_index = 3
    while len(degeneracies) < num_R:
        degeneracies.extend(map(int, lines[line_index].split()))
        line_index += 1

    H_R_dict = defaultdict(lambda: np.zeros((num_wann, num_wann), dtype=complex))
    
    for line in lines[line_index:]:
        parts = line.split()
        if len(parts) == 7:
            R = tuple(map(int, parts[0:3]))
            m = int(parts[3]) - 1
            n = int(parts[4]) - 1
            real = float(parts[5])
            imag = float(parts[6])
            H_R_dict[R][m, n] += real + 1j * imag
        



    return num_wann, H_R_dict

def parse_kpt(filename, tol=1e-5):
    with open(filename, 'r') as f:
        lines = f.readlines()
        
    num_kpts = int(lines[0].strip())
    kpoints = np.array([[float(x) for x in line.split()[:3]] for line in lines[1:num_kpts+1]])
    
    # Detect symmetry points based on k-path direction change
    sym_points_idx = [0]
    for i in range(1, len(kpoints) - 1):
        v1 = kpoints[i] - kpoints[i - 1]
        v2 = kpoints[i + 1] - kpoints[i]
        if np.linalg.norm(v1 - v2) > tol:
            sym_points_idx.append(i)
    sym_points_idx.append(len(kpoints) - 1)
    
    # Default symmetry labels (modify as needed)
    #default_labels = [r'$\Gamma$', 'X', 'M', r'$\Gamma$']
    #default_labels = [r'$\Gamma$', 'M', r'$\Gamma$' ]
    default_labels = ['M', r'$\Gamma$', 'X','M' ]
    sym_labels = default_labels[:len(sym_points_idx)]

    return kpoints, sym_points_idx, sym_labels

def compute_kpath_distance(kpoints):
    distances = [0.0]
    for i in range(1, len(kpoints)):
        dk = np.linalg.norm(kpoints[i] - kpoints[i-1])
        distances.append(distances[-1] + dk)
    return np.array(distances)

def calculate_H_k(num_wann, H_R_dict, kpoints):
    print(len(kpoints))
    R_list = list(H_R_dict.keys())
    H_R_list = [H_R_dict[R] for R in R_list]
    R_vectors = np.array(R_list)
    DFT_spectrum = []
    H_k_dict = defaultdict(lambda: np.zeros((num_wann, num_wann), dtype=complex))
    for k in kpoints:
        k_tuple=tuple(map(float, k[0:3]))
        if k_tuple in H_k_dict:
            print(f"Warning: Duplicate k-point {k_tuple} — overwriting.")
        H_k = np.zeros((num_wann, num_wann), dtype=complex)
        for R, H_R in zip(R_vectors, H_R_list):
            phase = np.exp(2j * np.pi * np.dot(k[:3], R))
            H_k += (H_R * phase)
        H_k_dict[k_tuple]=H_k
        
        eigvals = np.linalg.eigvalsh((H_k))
        DFT_spectrum.append(np.sort(eigvals.real))
    return  H_k_dict, DFT_spectrum

def obtain_dmft_spectrum(distances):

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
    #plt.figure(figsize=(12, 8))

    # Plot the bands from the FeSe.spaghetti_ene file
    lower_bound = -3.0
    upper_bound = 2.3
    DMFT_spectrum = [[] for _ in range(500)]
    for band, band_index in zip(data, band_indices):
        band_data = np.array(band)
        #print(band_index)
        # Filter data within the range for the 5th column
        filtered_data = band_data[(band_data[:, 4] >= lower_bound) & (band_data[:, 4] <= upper_bound)]
        index=[]
        if len(filtered_data[:, 3]) == 500:
            DMFT_distances = filtered_data[:, 3]* 1.70710678/1.50461
            for distance in distances:
                # Find the index of the closest value
                closest_index = np.argmin(np.abs(DMFT_distances - distance))
                index.append(closest_index)
            for k in index:
                DMFT_spectrum[k].append(filtered_data[:, 4][k])
            #print(index)
    """
    plt.figure(figsize=(10, 6))
    for i,inner_array in enumerate(DMFT_spectrum):
        plt.plot(np.linspace(0,1.70710678,500),DMFT_spectrum, alpha=0.2)  # alpha makes lines more transparent

    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.title("Plot of 500 Inner Arrays")
    plt.grid(True)
    """
    # Remove empty arrays
    filtered = [arr for arr in DMFT_spectrum if len(arr) > 0]
    
    return filtered


def plot_bands(bands, kpoints, sym_points_idx, sym_labels,clr):
    x_vals = compute_kpath_distance(kpoints)
    num_bands = bands.shape[1]

    for i in range(num_bands):
        #plt.plot(x_vals, bands[:, i], color='b', linewidth=1)
        if i == 0:
            plt.plot(x_vals, bands[:, i], color=clr, linewidth=1)
        else: 
            plt.plot(x_vals, bands[:, i], color=clr, linewidth=1)
    for idx in sym_points_idx:
        plt.axvline(x=x_vals[idx], color='g', linestyle='--', linewidth=0.5)

    plt.xticks([x_vals[i] for i in sym_points_idx], sym_labels, fontsize=12)
    #plt.xlabel("Wave vector", fontsize=14)
    plt.ylabel("Energy (eV)", fontsize=14)
    plt.ylim(-3,2.3)
    plt.axhline(0, linestyle=(0, (5, 5)), linewidth=1, color='r', alpha=0.5)
    #plt.title(title, fontsize=15)
    plt.grid(True, linestyle=':', linewidth=0.5)
    plt.savefig("ayer.png",dpi = 500, bbox_inches='tight')
    plt.tight_layout()
    #plt.show()

def clean_matrix(matrix):
    threshold = 1e-6
    # Zero out small real and imaginary parts
    real_part = np.real(matrix)
    imag_part = np.imag(matrix)
    real_part[np.abs(real_part) < threshold] = 0.0
    imag_part[np.abs(imag_part) < threshold] = 0.0
    return real_part + 1j * imag_part


def H_k_to_H_R(H_R_dict, H_K_dict, num_wann):
    R_list=list(H_R_dict.keys())
    print(len(R_list))
    H_R_dict_new = defaultdict(lambda: np.zeros((num_wann, num_wann), dtype=complex))
    for R in R_list:
        H_R = np.zeros((num_wann, num_wann), dtype=complex)
        for K, H_K in H_K_dict.items():
            phase = np.exp(-1j * 2 * np.pi * np.dot(K, R))  # be sure k and R are in reciprocal units
            H_R += H_K * phase
        H_R_dict_new[R]=H_R
    return H_R_dict_new


def calculate_band_structure(num_wann, H_R_dict, kpoints):
    R_list = list(H_R_dict.keys())
    H_R_list = [H_R_dict[R] for R in R_list]
    R_vectors = np.array(R_list)
    bands = []
    for k in kpoints:
        H_k = np.zeros((num_wann, num_wann), dtype=complex)
        for R, H_R in zip(R_vectors, H_R_list):
            phase = np.exp(2j * np.pi * np.dot(k[:3], R))
            H_k += (H_R * phase)
        eigvals = np.linalg.eigvalsh((H_k))
        bands.append(np.sort(eigvals.real))
    return np.array(bands)
hr_file = "wan_hr.dat"
kpt_file = "wan_band.kpt"

num_wann, H_R_dict = parse_hr_dat(hr_file)

kpoints, sym_points_idx, sym_labels = parse_kpt(kpt_file)


H_k_dict, DFT_spectrum = calculate_H_k(num_wann, H_R_dict, kpoints)
print(len(DFT_spectrum))
distances = compute_kpath_distance(kpoints)


DMFT_spectrum = obtain_dmft_spectrum(distances)
print(len(DMFT_spectrum))




# Assuming H_k_dict is a dictionary with keys = k-points, values = matrices

DMFT_H_k_dict = defaultdict(lambda: np.zeros((num_wann, num_wann), dtype=complex))

counter = 0
for k, Hk in H_k_dict.items():
    counter+=1
    Hk_cleaned = clean_matrix(Hk)

    E0, U = np.linalg.eigh(Hk_cleaned)  # eigenvalues and eigenvectors
    E_target = np.array(DMFT_spectrum[counter])
    # Rebuild matrix
    H_target = U @ np.diag(E_target) @ U.conj().T
    DMFT_H_k_dict[tuple(map(float, k[0:3]))]=H_target
    
print(counter)

#print(counter)
bands = []

for k,H_K in DMFT_H_k_dict.items():
    eigvals = np.linalg.eigvalsh(H_K)
    bands.append(np.sort(eigvals.real))
bands.append(bands[0])


plot_bands(np.array(bands), kpoints, sym_points_idx, sym_labels,clr='b')

H_R_new=H_k_to_H_R(H_R_dict, H_k_dict, num_wann)

bands = calculate_band_structure(num_wann, H_R_new, kpoints)
   
plot_bands(np.array(bands), kpoints, sym_points_idx, sym_labels,clr='r')

plt.show()







    
    


