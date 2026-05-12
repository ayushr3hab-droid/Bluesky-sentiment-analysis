import pandas as pd
import os


folder_path = 'cleaned_final.csv'  
output_file = 'Combined_Data.xlsx'

# This looks for files ending in .csv in that folder
all_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

if not all_files:
    print("No CSV files found in this folder!")
else:
    combined_data = []
    for file in all_files:
        print(f"Processing: {file}")
        df = pd.read_csv(file)
        
        df['Source_File'] = file
        combined_data.append(df)

    # Combine all data frames into one
    final_df = pd.concat(combined_data, axis=0, ignore_index=True)
    
    # Save it
    final_df.to_excel(output_file, index=False)
    print(f"Success! {output_file} created with {len(all_files)} files.")
