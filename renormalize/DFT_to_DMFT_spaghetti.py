import pandas as pd
import numpy as np
import re
import glob

def process_file_to_dataframe(file_path):
    # Initialize an empty list to store the band energy values for each k-point
    band_data = []
    band_columns = None
    
    with open(file_path, 'r') as f:
        current_band_energies = []  # List to store the energies of bands for the current k-point
        current_band_count = None  # To store the band count (from the 5th number in the matched line)
        
        for line in f:
            line = line.strip()  # Remove leading/trailing whitespace
            
            # Split the line by spaces and filter out any empty strings
            parts = [part for part in line.split() if part]
            
            # Check if the line has exactly 6 numbers, we look for this structure
            first_band_count = 0
            if len(parts) == 6 or len(parts) == 7:
                if len(parts) == 6:
                    band_count_position=4
                if len(parts)==7:
                    band_count_position=5
                
                

                try:
                    # Extract the band count from the 5th number
                    band_count = int(parts[band_count_position]) # The 5th number is the band count (need to make dynamic)
                    current_band_count = band_count
                    current_band_energies = []  # Reset the current band energies for a new k-point
                    
                except ValueError:
                    continue  # Skip lines that don't have valid numerical values for the band count
            
            # If the line has 2 values (band number and energy), we append the energy to the current k-point's data
            elif len(parts) == 2 and current_band_count is not None:
                try:
                    band_number = int(parts[0])  # Band number (first column)
                    energy = float(parts[1])  # Energy (second column)
                    
                    # Append the energy to the current k-point's band data
                    current_band_energies.append(energy)
                    
                    # If we've collected all the bands for this k-point, add the data to band_data
                    if len(current_band_energies) == current_band_count:
                        band_data.append(current_band_energies)
                        current_band_energies = []  # Reset for the next k-point
                
                except ValueError:
                    continue  # Skip lines that don't have valid numerical values for band number or energy
    max_band_count=max(len(bands) for bands in band_data)
    print("Maximum number of bands in .energy file is:", max(len(bands) for bands in band_data))
    #print(band_data)
     # If we haven't set the band_columns yet, set them dynamically based on the band count
    if band_columns is None:
        band_columns = [f'Band_{i+1}' for i in range(max_band_count)]  # Dynamic column names
                        
    
    df = pd.DataFrame(band_data, columns=band_columns)
     # Create a pandas DataFrame with dynamic columns
    
    # Add k_index column as 0, 1, 2, ..., len(df) - 1
    df.insert(0, 'k_index', range(1,len(df)+1))
    
    return df


def parse_self_energy_file(filepath):

    # Read the first line of EF.dat
    with open('EF.dat', 'r') as f:
        first_line = f.readline().strip()

    # Optionally convert to float (if it's a number)
    try:
        EF_value = float(first_line)
    except ValueError:
        EF_value = first_line  # Keep as string if it's not a number

    print("EF =", EF_value, type(EF_value))


    data = []
    conversion_factor = 0.0734985857  # Conversion factor to Ry
    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Extract number of bands from the first # line
    first_line = lines[0].strip()
    parts = first_line.split()
    nbands = int(parts[3])  # The number of bands is the 4th value in the header line
    nemin = int(parts[4])   # nemin value
    nomega = int(parts[5])  # nomega value

    k_index = 1  # Start from 1
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("#"):
            block = []
            for j in range(nomega):
                i += 1
                block.append([float(x) for x in lines[i].split()])
            block = np.array(block)

            omega = block[:, 0] + EF_value
            block_dict = {
                "omega": omega
            }

            for b in range(nbands):
                re_col = 2 * b + 1
                im_col = 2 * b + 2
                band = nemin + b
                block_dict[f"ReSigma_b{band}"] = block[:, re_col]
                #block_dict[f"ImSigma_b{band}"] = block[:, im_col]

            df_block = pd.DataFrame(block_dict)
            df_block.insert(0, "k_index", k_index)  # First column is k_index

            # Multiply all columns except "k_index" by the conversion factor (to Ry)
            for column in df_block.columns:
                if column != "k_index":
                    df_block[column] = df_block[column] * conversion_factor

            # Formatting numbers to ensure proper significant figures
            for column in df_block.columns:
                if column != "k_index":
                    df_block[column] = df_block[column].apply(lambda x: f"{x:.16f}")

            data.append(df_block)

            k_index += 1
        i += 1

    return pd.concat(data, ignore_index=True), nbands, nemin

df_DMFT, num_of_DMFT_bands, min_DMFT_band = parse_self_energy_file("eigvals.dat")
#print(df_DMFT.columns)
#print(df_DMFT.head(2), num_of_DMFT_bands, min_DMFT_band)
#print(df.tail())

file_path = glob.glob("*.energy")[0]  # Replace with the actual file path

print("Found .energy file", file_path)
df_DFT = process_file_to_dataframe(file_path)
#print(df_DFT)
df_DFT_relates_to_DMFT = df_DFT[['k_index'] + [f'Band_{i}' for i in range(min_DMFT_band, min_DMFT_band+num_of_DMFT_bands)]]
#print(df_DFT_relates_to_DMFT)

# Print the DataFrame
#print(df_DFT_relates_to_DMFT)

# Loop through all k_index values in df2
for k in df_DFT_relates_to_DMFT['k_index']:
    # Find matching rows in df1
    matches = df_DMFT[df_DMFT['k_index'] == k]
    
    # Print the k_index and corresponding matching rows
    #print(f"\nMatches for k_index = {k}:")
    #print(matches)



# Ensure omega column is numeric
df_DMFT['omega'] = pd.to_numeric(df_DMFT['omega'], errors='coerce')

output_path = file_path+'_modified'

# Assume df_DFT_relates_to_DMFT and df_DMFT are preloaded

k_points = 0
bands = 0
in_between = False
current_k_index = 0
band_index = 0

output_lines = []

with open(file_path, 'r') as f:
    for line in f:
        parts = line.strip().split()
        n_parts = len(parts)

        if n_parts == 6 or n_parts==7:
            current_k_index += 1
            k_points += 1
            bands = 0
            band_index = 0
            in_between = True
            output_lines.append(line)

        elif in_between and n_parts == 2:
            band_index += 1
            bands += 1

            try:
                dft_val = float(parts[0])  # Band value
                current_val = float(parts[1])  # Second column value (modified)
            except ValueError:
                output_lines.append(line)
                continue

            matching_rows = df_DMFT[df_DMFT['k_index'] == current_k_index]
            band_min=matching_rows.columns[2]
            band_max=matching_rows.columns[-1]

            # Use regex to find the first number in the string
            match = re.search(r'\d+', band_min)  # \d+ matches one or more digits

            match_max = re.search(r'\d+', band_max)

            if match:
                band_min = int(match.group(0))  # Extract the number and convert to int
                #print(f"Extracted band number: {band_min}")
                band_max = int(match_max.group(0))
            else:
                print("No number found in the string")
        


            if not matching_rows.empty:
                omega_min = matching_rows['omega'].min()
                omega_max = matching_rows['omega'].max()

                # Check if second value lies within DMFT omega range
                if omega_min <= current_val <= omega_max and band_index>=band_min and band_index<=band_max:
                    
                    # Find closest omega row
                    closest_idx = np.abs(matching_rows['omega'] - pd.to_numeric(matching_rows[f'ReSigma_b{band_index}'], errors='coerce')).argmin() 
                    #enforces dirac delta of A(w,k) when imaginary part of self energy goes to zero
                    closest_row = matching_rows.iloc[closest_idx]
                    re_sigma = float(closest_row[f'ReSigma_b{band_index}'])
                    modified_val = float(re_sigma)
                    if modified_val < 0:
                        formatted_modified_val = f"{modified_val:.14f}"
                    else:
                        formatted_modified_val = f"{modified_val:.14f}"
                    new_line = f"          {int(dft_val)}   {formatted_modified_val}\n"
                    #print(new_line)
                    output_lines.append(new_line)
                    continue  # Skip default append

            # If no modification, append the original line
            output_lines.append(line)

        elif in_between and n_parts not in (2, 6, 7):
            in_between = False
            output_lines.append(line)
        else:
            output_lines.append(line)

# Write to new file
with open(output_path, 'w') as f_out:
    f_out.writelines(output_lines)

print(f"Number of k-points: {k_points}")
#print(f"Number of bands per k-point (based on last block): {bands}")
print(f"Modified .energy file saved as '{output_path}'")


# Example usage
file_path_mod = glob.glob("*.energy_modified")[0]  # Replace with the actual file path

df_DFT = process_file_to_dataframe(file_path_mod)

print("modified energy file found", file_path_mod)
#print(df_DFT.head())
# Assume df_DFT is your DataFrame loaded already

count = 0
output_lines = []

file_path_output = glob.glob("*.output1")[0]

print("output1 file found", file_path_output)
with open(file_path_output, "r") as file:
    lines = file.readlines()

i = 0
while i < len(lines):
    line = lines[i]
    output_lines.append(line)

    if "EIGENVALUES ARE:" in line:
        count += 1

        # Find the matching row in df_DFT
        matched_row = None
        for index, row in df_DFT.iterrows():
            k_value = row.iloc[0]
            if k_value == count:
                matched_row = row
                break
        
        if matched_row is not None:
            values = matched_row.iloc[1:].values


            # Format the block and count how many lines it will take
            formatted_block = ""
            print_counter = 0
            for start in range(0, len(values), 5):
                chunk = values[start:start+5]
                formatted_line = ' '.join(f"{val:12.7f}" for val in chunk)

                if np.all(pd.isna(chunk)):
                    continue
                
                if "nan" in formatted_line:
                    # Only format values before first NaN
                    valid_chunk = chunk[~pd.isna(chunk)]
                   
                    if len(valid_chunk) > 0:
                        formatted = ''.join(f"{val:12.7f} " for val in valid_chunk)
                        formatted_block +=  "   " + formatted + "\n"
                        i+=1
                    break
                        
                else:
                    formatted_block += "   " + formatted_line + '\n'
                    if print_counter%10==7:
                        formatted_block += '\n'
                    print_counter += 1

            # Skip the next print_counter lines from original file (replace with formatted block)
            i += 1  # move past "EIGENVALUES ARE:" line
            for _ in range(print_counter):
                i += 1  # skip these lines (the original block)
            
            # Insert the new formatted block lines into output
            
            output_lines.append(formatted_block)

            
            i+=1
            continue  # continue the while loop without incrementing i again

    i += 1

# Write to a new file
with open(file_path_output+"_modified", "w") as f_out:
    f_out.writelines(output_lines)
