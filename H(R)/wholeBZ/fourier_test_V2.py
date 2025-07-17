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
    default_labels = default_labels * (242 // len(default_labels)) + default_labels[:242 % len(default_labels)]
    sym_labels = default_labels[:len(sym_points_idx)]

    return kpoints, sym_points_idx, sym_labels

def calculate_band_structure(num_wann, H_R_dict, kpoints):
    R_list = list(H_R_dict.keys())
    H_R_list = [H_R_dict[R] for R in R_list]
    R_vectors = np.array(R_list)
    bands = []
    H_K_dict = defaultdict(lambda: np.zeros((num_wann, num_wann), dtype=complex))
    for k in kpoints:
        H_k = np.zeros((num_wann, num_wann), dtype=complex)
        for R, H_R in zip(R_vectors, H_R_list):
            phase = np.exp(2j * np.pi * np.dot(k[:3], R))
            H_k += H_R * phase
        eigvals = np.linalg.eigvalsh(H_k)
        bands.append(np.sort(eigvals.real))
        H_K_dict[tuple(map(float, k[:3]))]=H_k
    return np.array(bands), H_K_dict

def compute_kpath_distance(kpoints):
    distances = [0.0]
    for i in range(1, len(kpoints)):
        dk = np.linalg.norm(kpoints[i] - kpoints[i-1])
        distances.append(distances[-1] + dk)
    return np.array(distances)

def plot_bands(bands, kpoints, sym_points_idx, sym_labels, clr, line_style):
    x_vals = compute_kpath_distance(kpoints)
    num_bands = bands.shape[1]

    
    for i in range(num_bands):
        #plt.plot(x_vals, bands[:, i], color='b', linewidth=1)
        plt.plot(x_vals, bands[:, i], color=clr, linewidth=2, linestyle=line_style)
    for idx in sym_points_idx:
        plt.axvline(x=x_vals[idx], color='g', linestyle='--', linewidth=0.5)

    plt.xticks([x_vals[i] for i in sym_points_idx], sym_labels, fontsize=12)
    #plt.xlabel("Wave vector", fontsize=14)
    plt.ylabel("Energy (eV)", fontsize=14)
    plt.ylim(-2,3.3)
    plt.axhline(0, linestyle=(0, (5, 5)), linewidth=1, color='r', alpha=0.5)
    #plt.title(title, fontsize=15)
    plt.grid(True, linestyle=':', linewidth=0.5)
    plt.savefig("Tight_Binding_Wannier90_FeSe_Monolayer.png",dpi = 500, bbox_inches='tight')
    plt.tight_layout()
    #plt.show()

def parse_kpt_MGXM(filename, tol=1e-5):
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
    #default_labels = ['M', r'$\Gamma$', 'X','M' ]
    default_labels = [r'$\Gamma$', 'Z' ]
    #default_labels = default_labels * (242 // len(default_labels)) + default_labels[:242 % len(default_labels)]
    sym_labels = default_labels[:len(sym_points_idx)]

    return kpoints, sym_points_idx, sym_labels

def plot_bands_MGXM(bands, kpoints, sym_points_idx, sym_labels, clr, line_style):
    x_vals = compute_kpath_distance(kpoints)
    num_bands = bands.shape[1]

    
    for i in range(num_bands):
        #plt.plot(x_vals, bands[:, i], color='b', linewidth=1)
        plt.plot(x_vals, bands[:, i], color=clr, linewidth=2, linestyle=line_style)
    for idx in sym_points_idx:
        plt.axvline(x=x_vals[idx], color='g', linestyle='--', linewidth=0.5)

    plt.xticks([x_vals[i] for i in sym_points_idx], sym_labels, fontsize=12)
    #plt.xlabel("Wave vector", fontsize=14)
    plt.ylabel("Energy (eV)", fontsize=14)
    plt.ylim(-0.25,.25)
    plt.axhline(0, linestyle=(0, (5, 5)), linewidth=1, color='r', alpha=0.5)
    #plt.title(title, fontsize=15)
    plt.grid(True, linestyle=':', linewidth=0.5)
    plt.savefig("Tight_Binding_Wannier90_FeSe_Monolayer.png",dpi = 500, bbox_inches='tight')
    plt.tight_layout()
    #plt.show()
def H_k_to_H_R(H_R_dict, H_K_dict, num_wann):

    H_R_dict_new = defaultdict(lambda: np.zeros((num_wann, num_wann), dtype=complex))
    for R, H_R in H_R_dict.items():
        #print(R)
        H_R = np.zeros((num_wann, num_wann), dtype=complex)
        k_counter=0
        for K, H_K in H_K_dict.items():
            
            K=np.array(list(K), dtype=float)
            
            phase = np.exp(-1j * 2 * np.pi * np.dot(K, list(R))) # be sure k and R are in reciprocal units
            H_R += H_K * phase
            k_counter+=1
        H_R_dict_new[R]=H_R/k_counter
    return H_R_dict_new

def obtain_dmft_spectrum(distances):

    # Read and parse the FeSe.spaghetti_ene file
    with open('FeSe.spaghetti_ene_DMFT', 'r') as file:
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

    #print(data)
    #print(band_indices)

    # Define plot parameters
    #plt.figure(figsize=(12, 8))

    # Plot the bands from the FeSe.spaghetti_ene file
    lower_bound = -2.3
    upper_bound = 3
    DMFT_spectrum = [[] for _ in range(847)]
    band_num=0
    for band, band_index in zip(data, band_indices):
        band_data = np.array(band)
        # Filter data within the range for the 5th column
        filtered_data = band_data[(band_data[:, 4] >= lower_bound) & (band_data[:, 4] <= upper_bound)]
        #print(filtered_data)
        
        if len(filtered_data[:, 3])==847 :
                
                DMFT_spectrum[band_num].append(filtered_data[:, 4])
                print(band_num)
                band_num+=1
    
    
    
       

    # Remove empty arrays
    filtered = [arr for arr in DMFT_spectrum if len(arr) > 0]
    flattened_data = [arr[0] for arr in filtered]  # shape: (10, 847)

    filtered =np.array(flattened_data).T
    return filtered

def clean_matrix(matrix):

    threshold = 1e-6
    # Zero out small real and imaginary parts
    real_part = np.real(matrix)
    imag_part = np.imag(matrix)
    real_part[np.abs(real_part) < threshold] = 0.0
    imag_part[np.abs(imag_part) < threshold] = 0.0
    return real_part + 1j * imag_part

def write_hr_dat(input_filename, output_filename, new_H_R_dict):
    """
    Modifies a wannier90_hr.dat-like file using a new H_R_dict and saves to a new file.

    Parameters:
    - input_filename: path to original HR file
    - output_filename: path to new HR file to save
    - new_H_R_dict: dictionary with keys as R-vectors (tuples of 3 ints)
                    and values as (num_wann x num_wann) complex numpy arrays
    """
    with open(input_filename, 'r') as f:
        lines = f.readlines()

    # Header lines
    header_0 = lines[0]           # comment line
    num_wann = int(lines[1].strip())
    num_R = int(lines[2].strip())

    # Degeneracy lines
    degeneracies = []
    degeneracy_lines = []
    line_index = 3
    while len(degeneracies) < num_R:
        degeneracy_lines.append(lines[line_index])
        degeneracies.extend(map(int, lines[line_index].split()))
        line_index += 1

    # Now write everything to the new file
    with open(output_filename, 'w') as f:
        f.write(header_0)
        f.write(f"{num_wann}\n")
        f.write(f"{num_R}\n")
        f.writelines(degeneracy_lines)

        # Write new matrix elements in the expected format
        for i, (R, H_mat) in enumerate(new_H_R_dict.items()):
            for m in range(num_wann):
                for n in range(num_wann):
                    val = H_mat[m, n]
                    f.write(f"{R[0]:3d} {R[1]:3d} {R[2]:3d} {m+1:3d} {n+1:3d} {val.real: .6f} {val.imag: .6f}\n")

def main():
    #hr_file = "wann_GF2_hr.dat"
    #kpt_file = "wann_GF2_band.kpt"
    
    hr_file = "wann_hr.dat"
    kpt_file = "wann_band_full.kpt"
    kpt_file_MGXM = "wann_band.kpt"
    
    
    #hr_file = "wannierHS1_hr.dat"
    #kpt_file = "wannierHS1_band.kpt"
    #hr_file = "wann_hr.dat"  
    #kpt_file = "wann_band.kpt"

    num_wann, H_R_dict = parse_hr_dat(hr_file)
    kpoints, sym_points_idx, sym_labels = parse_kpt(kpt_file)
    bands, H_K_dict = calculate_band_structure(num_wann, H_R_dict, kpoints)
    plt.figure(figsize=(8, 6))
    #plot_bands(bands, kpoints, sym_points_idx, sym_labels, clr='blue',line_style='-')

    #H_R_new=H_k_to_H_R(H_R_dict,H_K_dict,num_wann)

    #bands, H_K_dict = calculate_band_structure(num_wann, H_R_new, kpoints)
    #plot_bands(bands, kpoints, sym_points_idx, sym_labels, clr='red', line_style='dotted')
    
    distances=compute_kpath_distance(kpoints)
    #print(distance)

    DMFT_spectrum = obtain_dmft_spectrum(distances)
    #print(DMFT_spectrum)

    # Assuming H_k_dict is a dictionary with keys = k-points, values = matrices

    DMFT_H_k_dict = defaultdict(lambda: np.zeros((num_wann, num_wann), dtype=complex))

    counter = 0
    
    for k, Hk in H_K_dict.items():
        
        #Hk_cleaned = clean_matrix(Hk)
        Hk_cleaned = Hk

        E0, U = np.linalg.eigh(Hk_cleaned)  # eigenvalues and eigenvectors
        #print(U.shape, len(E0))
        E_target = DMFT_spectrum[counter]
        # Rebuild matrix
        
        H_target = U @ np.diag(E_target) @ U.conj().T
        DMFT_H_k_dict[tuple(map(float, k[0:3]))]=H_target
        counter+=1
    

    #print(counter)

    #print(counter)
    bands = []

    for k,H_K in DMFT_H_k_dict.items():
        eigvals = np.linalg.eigvalsh(H_K)
        bands.append(np.sort(eigvals.real))
    
    #bands.append(bands[0]) only needed for MGXM because last point is same as first.


    plot_bands(np.array(bands), kpoints, sym_points_idx, sym_labels,clr='g',line_style='-')

    H_R_new=H_k_to_H_R(H_R_dict, DMFT_H_k_dict, num_wann)

    plt.figure(figsize=(8, 6))
    kpoints2, sym_points_idx2, sym_labels2 = parse_kpt_MGXM("k_path.dat")

    bands, H_k_new = calculate_band_structure(num_wann, H_R_new, kpoints2)

    plot_bands_MGXM(np.array(bands), kpoints2, sym_points_idx2, sym_labels2,clr='b',line_style="-")


    
    
    #plot_bands(np.array(bands), kpoints, sym_points_idx, sym_labels,clr='p',line_style='-')
    # Read and parse the FeSe.spaghetti_ene file
    with open('FeSe.spaghetti_ene_DMFT_MGXM', 'r') as file:
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

    

    # Plot the bands from the FeSe.spaghetti_ene file
    lower_bound = -2
    upper_bound = 2.3
    
    for band, band_index in zip(data, band_indices):
       
        band_data = np.array(band)
        # Filter data within the range for the 5th column
        filtered_data = band_data[(band_data[:, 4] >= lower_bound) & (band_data[:, 4] <= upper_bound)]
        #print(filtered_data)
        if len(filtered_data) > 0 and len(band_data[:, 3])==500 and band_index != 35:
            #print ("Band indicies: ", band_index)
            distance_FBZ = 2.11056408e+02
            distance_MGXM = 1.70710678
            #plt.plot(np.linspace(0,distance_MGXM,len(filtered_data[:, 4])), filtered_data[:, 4], label=f"FeSe Band {band_index}",color="green")

    
    write_hr_dat(hr_file,"new.dat", H_R_new)
    
    #plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()




