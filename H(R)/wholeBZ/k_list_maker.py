import numpy as np

def generate_k_path(high_sym_points, points_per_segment=50):
    """
    Generate a list of k-points interpolated between high-symmetry points.
    """
    k_path = []
    for i in range(len(high_sym_points) - 1):
        start = np.array(high_sym_points[i])
        end = np.array(high_sym_points[i + 1])
        segment = np.linspace(start, end, points_per_segment, endpoint=False)
        k_path.extend(segment)
    k_path.append(high_sym_points[-1])
    return k_path

def write_k_path_to_file(k_path, filename="k_path.dat"):
    """
    Write the k-point path to a file in the desired format.
    First line: number of points.
    Remaining lines: kx ky kz 1.0
    """
    with open(filename, "w") as f:
        f.write(f"{len(k_path)}\n")
        for k in k_path:
            f.write(f"{k[0]:.6f}    {k[1]:.6f}    {k[2]:.6f}   1.0\n")
    print(f"K-path written to {filename}")

# === Example usage ===
if __name__ == "__main__":
    high_sym_points = [
        (0.0, 0.0, 0.0),   # Γ
        (0.0, 0.0, 0.5),   # Z
        
    ]

    points_per_segment = 40
    k_path = generate_k_path(high_sym_points, points_per_segment)
    write_k_path_to_file(k_path, "k_path.dat")
