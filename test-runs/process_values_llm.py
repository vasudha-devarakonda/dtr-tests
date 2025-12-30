import pandas as pd
import glob
import os

def process_csv_files(input_folder, output_file="llm-results/summary-llm.csv"):
    summary_rows = []

    # Get all CSV files in the input folder
    csv_files = glob.glob(os.path.join(input_folder, "*.csv"))

    # Loop through each CSV file
    for file in csv_files:
        df = pd.read_csv(file)

        # Extract general information (from first row of the DataFrame)
        model_name = df["model_name"].iloc[0]
        batch_size = df["batch_size"].iloc[0]
        memory_budget = df["memory_budget"].iloc[0] / (1024 * 1024)  # Convert memory budget to MB

        # Loop through each row to process data and calculate metrics
        for j, data in df.iterrows():
            # Create an entry for each row (rep, epoch, etc.)
            entry = {
                'model_name': model_name,
                'batch_size': batch_size,
                'rep': j,  # This is the index, or if you have another column for 'rep', you can replace it
                'epoch': data['epoch'],
                'time': data['time'] * 1e3,  # Convert time from seconds to milliseconds
                'time_with_model': data['time_with_model'] * 1e3,  # Convert time from seconds to milliseconds
                'total_mem': data['total_mem'] / (1024 * 1024),  # Convert total_mem from bytes to MB
                'memory_budget': memory_budget,  # Memory budget already in MB
                'base_compute_time': data['base_compute_time'] * 1e-6,  # Convert microseconds to seconds
                'remat_compute_time': data['remat_compute_time'] * 1e-6,  # Convert microseconds to seconds
                'search_time': data['search_time'] * 1e-6,  # Convert microseconds to seconds
                'cost_time': data['cost_time'] * 1e-6,  # Convert microseconds to seconds
                'remat_count': data['remat_count'],  # Remat count (integer)
                'remat_size': data['remat_size'] / (1024 * 1024),  # Convert remat_size from bytes to MB
                'model_mem': data['model_mem'] / (1024 * 1024),  # Convert model_mem from bytes to MB
                'input_mem': data['input_mem'] / (1024 * 1024)  # Convert input_mem from bytes to MB
            }

            # Append the entry to the summary rows
            summary_rows.append(entry)

    # Create a DataFrame from the summary rows
    summary_df = pd.DataFrame(summary_rows)

    # Round all numeric columns to 3 decimal places for better readability
    numeric_cols = summary_df.select_dtypes(include=["float", "int"]).columns
    summary_df[numeric_cols] = summary_df[numeric_cols].round(3)

    # Write the summary DataFrame to a CSV file
    summary_df.to_csv(output_file, index=False)
    print(f"Summary written to {output_file}")

# Call the function with the input folder containing your CSV files
process_csv_files("llm-results")
