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
    #labels_list = default_labels * (242 // len(default_labels)) + default_labels[:242 % len(default_labels)]
    sym_labels = default_labels[:len(sym_points_idx)]

    return kpoints, sym_points_idx, sym_labels

def calculate_band_structure(num_wann, H_R_dict, kpoints):
    R_list = list(H_R_dict.keys())
    H_R_list = [H_R_dict[R] for R in R_list]
    R_vectors = np.array(R_list)
    bands = []
    for k in kpoints:
        H_k = np.zeros((num_wann, num_wann), dtype=complex)
        for R, H_R in zip(R_vectors, H_R_list):
            phase = np.exp(2j * np.pi * np.dot(k[:3], R))
            H_k += H_R * phase
        eigvals = np.linalg.eigvalsh(H_k)
        bands.append(np.sort(eigvals.real))
    return np.array(bands)

def compute_kpath_distance(kpoints):
    distances = [0.0]
    for i in range(1, len(kpoints)):
        dk = np.linalg.norm(kpoints[i] - kpoints[i-1])
        distances.append(distances[-1] + dk)
    return np.array(distances)

def plot_bands(bands, kpoints, sym_points_idx, sym_labels, title="Wannier90 Tight Binding Band Strucutre"):
    x_vals = compute_kpath_distance(kpoints)
    num_bands = bands.shape[1]

    plt.figure(figsize=(8, 6))
    for i in range(num_bands):
        #plt.plot(x_vals, bands[:, i], color='b', linewidth=1)
        plt.plot(x_vals, bands[:, i] + 1.680, color='b', linewidth=1)
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
    plt.show()

def main():
    #hr_file = "wann_GF2_hr.dat"
    #kpt_file = "wann_GF2_band.kpt"
    
    hr_file = "wann_hr.dat"
    kpt_file = "wann_band.kpt"
    
    
    #hr_file = "wannierHS1_hr.dat"
    #kpt_file = "wannierHS1_band.kpt"
    #hr_file = "wann_hr.dat"  
    #kpt_file = "wann_band.kpt"

    num_wann, H_R_dict = parse_hr_dat(hr_file)
    kpoints, sym_points_idx, sym_labels = parse_kpt(kpt_file)
    bands = calculate_band_structure(num_wann, H_R_dict, kpoints)
    np.save("Tight_binding_bands.npy", bands)
    plot_bands(bands, kpoints, sym_points_idx, sym_labels)

if __name__ == "__main__":
    main()




