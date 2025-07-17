import pandas as pd
import numpy as np
import re

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
            if len(parts) >= 6:
                try:
                    # Extract the band count from the 5th number
                    band_count = int(parts[4])  # The 5th number is the band count (need to make dynamic)
                    
                    # If we haven't set the band_columns yet, set them dynamically based on the band count
                    if band_columns is None:
                        band_columns = [f'Band_{i+1}' for i in range(band_count)]  # Dynamic column names

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
    
    # Create a pandas DataFrame with dynamic columns
    df = pd.DataFrame(band_data, columns=band_columns)
    # Add k_index column as 0, 1, 2, ..., len(df) - 1
    df.insert(0, 'k_index', range(1,len(df)+1))
    
    return df

file_path = 'FeSe.energy_modified'  # Replace with the actual file path

df_DFT = process_file_to_dataframe(file_path)


print(df_DFT.head())
# Assume df_DFT is your DataFrame loaded already

count = 0
output_lines = []

with open("FeSe.output1", "r") as file:
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
                        formatted = ' '.join(f"{val:12.7f}" for val in valid_chunk)
                        formatted_block += formatted
                    break  # Stop replacing further lines for this k-point

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
with open("FeSe.output1_modified", "w") as f_out:
    f_out.writelines(output_lines)



    
    

