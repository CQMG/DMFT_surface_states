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

def calculate_band_structure(num_wann, H_R_dict, kpoints,sigma_matrix,Z_matrix):
    R_list = list(H_R_dict.keys())
    H_R_list = [H_R_dict[R] for R in R_list]
    R_vectors = np.array(R_list)
    bands = []
    k_counter=0
    H_k1=np.zeros((num_wann, num_wann), dtype=complex)
    for k in kpoints:
        H_k = np.zeros((num_wann, num_wann), dtype=complex)
        for R, H_R in zip(R_vectors, H_R_list):
            phase = np.exp(2j * np.pi * np.dot(k[:3], R))
            H_k += (H_R * phase)
        eigvals = np.linalg.eigvalsh(Z_matrix@(H_k+sigma_matrix)@Z_matrix)
        bands.append(np.sort(eigvals.real))
        
        
        if k_counter == 50:
            np.set_printoptions(precision=4, suppress=True)
            #print(H_k)
            # Open file for writing
            tolerance = 1e-6
            num_approx_zero = np.sum(np.abs(H_k.real) < tolerance)
            with open('H_k_matrix.txt', 'w') as f:
                for row in H_k:
                    # Format each complex element as "(real, imag)"
                    row_str = '\t'.join(f'{elem.real:.6f}' for elem in row)
                    f.write(row_str + '\n')
                f.write(f'Number of elements with real part close to zero (|Re| < {tolerance}): {num_approx_zero}\n')
            H_k1=H_k
            k_counter+=1
    
    return np.array(bands), H_k1

def compute_kpath_distance(kpoints):
    distances = [0.0]
    for i in range(1, len(kpoints)):
        dk = np.linalg.norm(kpoints[i] - kpoints[i-1])
        distances.append(distances[-1] + dk)
    return np.array(distances)

def plot_bands(bands, kpoints, sym_points_idx, sym_labels,clr):
    x_vals = compute_kpath_distance(kpoints)
    print(x_vals)
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
    plt.savefig("Tight_Binding_Wannier90_FeSe_Monolayer.png",dpi = 500, bbox_inches='tight')
    plt.tight_layout()
    #plt.show()

def main():
    #hr_file = "wann_GF2_hr.dat"
    #kpt_file = "wann_GF2_band.kpt"
    
    hr_file = "wan_hr.dat"
    kpt_file = "wan_band.kpt"
    
    
    #hr_file = "wannierHS1_hr.dat"
    #kpt_file = "wannierHS1_band.kpt"
    #hr_file = "wann_hr.dat"  
    #kpt_file = "wann_band.kpt"

    num_wann, H_R_dict = parse_hr_dat(hr_file)
    #print(H_R_dict)

    

   

    Z_matrix = np.zeros((num_wann, num_wann), dtype=complex)

    # Apply Zxy = 3.5 to both Fe atoms' dxy orbitals
    xy_indices = [0, 5]  # index of dxy for Fe1 and Fe2
    xz_yz_indices = [1, 2, 6, 7]
    other_indices  = [3, 4, 8, 9]  # x^2-y^2 and z^2

    # Assign values

    x=.8
    
    for i in xy_indices:
        Z_matrix[i, i] = 3

    for i in xz_yz_indices:
        Z_matrix[i, i] = 3

    for i in other_indices:
        Z_matrix[i, i] =3

    
    sigma_matrix = np.zeros((num_wann, num_wann), dtype=complex)

    mu=0

    # Define chemical potentials per orbital type (in meV or eV)
    sigma_xy =  .9-mu    # for d_xy
    sigma_xz = .7-mu   # for d_xz, d_yz
    sigma_yz = .7-mu  # for d_x2-y2, d_z2
    sigma_x_sq = .6-mu
    sigma_z_sq = 1.14-mu

    xy_indices = [0, 5]
    xz_indices = [1, 6]
    yz_indices = [2, 7]
    x_sq_indices =[3,8]
    z_sq_indices = [4,9]


    # Assign diagonal chemical potential
    for i in xy_indices:
        sigma_matrix[i, i] = sigma_xy

    for i in xz_indices:
        sigma_matrix[i, i] = sigma_xz

    for i in yz_indices:
        sigma_matrix[i, i] = sigma_yz
    
    for i in x_sq_indices:
        sigma_matrix[i, i] = sigma_x_sq

    for i in z_sq_indices:
        sigma_matrix[i, i] = sigma_z_sq

    H_R_dict_modified = defaultdict(lambda: np.zeros((num_wann, num_wann), dtype=complex))
    
    for R, H in H_R_dict.items():
        if R == (0,0,0):
            H_R_dict_modified[R]= np.sqrt(Z_matrix) @ (H+sigma_matrix) @ np.sqrt(Z_matrix)
        else:
            H_R_dict_modified[R]=Z_matrix @ (H)

    

    plt.figure(figsize=(8, 6))
    kpoints, sym_points_idx, sym_labels = parse_kpt(kpt_file)
    #bands = calculate_band_structure(num_wann, H_R_dict, kpoints,sigma_matrix)
   
    #plot_bands(bands, kpoints, sym_points_idx, sym_labels,clr='r')

    bands, H_k1 = calculate_band_structure(num_wann, H_R_dict, kpoints, sigma_matrix, Z_matrix)
    bands, H_k1 = calculate_band_structure(num_wann, H_R_dict_modified, kpoints, sigma_matrix*0, np.eye(Z_matrix.shape[0]))
   
    plot_bands(bands, kpoints, sym_points_idx, sym_labels,clr='b')

    

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
    dmft_spectrum = []
    for band, band_index in zip(data, band_indices):
        band_data = np.array(band)
        #print(band_index)
        # Filter data within the range for the 5th column
        filtered_data = band_data[(band_data[:, 4] >= lower_bound) & (band_data[:, 4] <= upper_bound)]

        if len(filtered_data) > 0 and band_index != 23 and band_index != 24 and band_index != 35 :
            # Plot the raw data without smoothing or scaling
            
            x_range=np.linspace(0,1.70710678,len(filtered_data[:, 3]))
            plt.plot(x_range, filtered_data[:, 4], label=f"FeSe Band {band_index}")
            #print(filtered_data[:, 3]*scaling)
        if len(filtered_data[:, 3]) == 500:
            dmft_eigenvalue = filtered_data[:, 4][209]
            dmft_spectrum.append(dmft_eigenvalue)

        

        
    #print(dmft_spectrum, bands[40])

    #print( np.linalg.eigvalsh((H_k1)))

    E0, U = np.linalg.eigh(H_k1)  # eigenvalues and eigenvectors
    E_target = np.array(dmft_spectrum)

    # Number of interpolation steps
    num_steps = 10
    interpolated_spectra = []

    for t in np.linspace(0, 1, num_steps):
        # Linear interpolation of eigenvalues
        E_t = (1 - t) * E0 + t * E_target
        # Rebuild interpolated matrix
        H_t = U @ np.diag(E_t) @ U.conj().T
        # Compute spectrum of the perturbed matrix
        E_t_check = np.linalg.eigvalsh(H_t)
        interpolated_spectra.append(E_t_check)

    # Plot spectrum evolution
    interpolated_spectra = np.array(interpolated_spectra)
    for band_idx in range(H_k1.shape[0]):
        plt.scatter(.7, interpolated_spectra[:, band_idx][-1], label=f'Band {band_idx+1}')
    plt.xlabel('Interpolation (t)')
    plt.ylabel('Energy (eV)')
    plt.title('Spectrum interpolation from H_k1 to DMFT spectrum')
    plt.grid(True)



    #plt.legend()
    plt.show()
    

if __name__ == "__main__":
    main()

