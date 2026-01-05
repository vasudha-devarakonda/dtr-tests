import pandas as pd
import glob
import os

def process_csv_files(input_folder, output_file="cnn-results-2/summary-cnn.csv"):
    summary_rows = []

    # Get all CSV files in the input folder
    csv_files = glob.glob(os.path.join(input_folder, "*.csv"))

    # Loop through each CSV file
    for file in csv_files:
        df = pd.read_csv(file)

        # Extract general information (from first row of the DataFrame)
        model_name = df["model_name"].iloc[0]
        batch_size = df["batch_size"].iloc[0]
        memory_budget = df["memory_budget"].iloc[0] / (1024 * 1024)  # Convert memory budget to MB if needed

        # Calculate total time in seconds (assuming time is in milliseconds)
        total_time_sec = df["time"].sum() / 1000.0
        total_time_with_model_sec = df["time_with_model"].sum() / 1000.0
        total_sync_time_sec = df["sync_time"].sum() / 1000.0
        total_gpu_time_sec = df["gpu_time"].sum() / 1000.0
        total_base_compute_time_sec = df["base_compute_time"].sum() / 1000.0
        total_remat_compute_time_sec = df["remat_compute_time"].sum() / 1000.0
        total_search_time_sec = df["search_time"].sum() / 1000.0
        total_cost_time_sec = df["cost_time"].sum() / 1000.0

        # Use memory values directly (no conversion to MB needed)
        total_mem_mb = df["total_mem"].mean()  # Already in MB
        input_mem_mb = df["input_mem"].mean()  # Already in MB
        model_mem_mb = df["model_mem"].mean()  # Already in MB

        # Count the number of unique repetitions and epochs
        num_reps = df["rep"].nunique()
        num_epochs = df["epoch"].nunique()
        num_iterations = num_reps / num_epochs if num_epochs != 0 else None

        # Calculate the total number of remat counts and remat size in MB (remat_size already in MB)
        total_remat_count = df["remat_count"].sum()
        total_remat_size_mb = df["remat_size"].sum()  # Already in MB

        # Append a summary row with the calculated metrics
        summary_rows.append({
            "model_name": model_name,
            "batch_size": batch_size,
            "num_iterations": num_iterations,
            "total_time_sec": total_time_sec,
            "total_time_with_model_sec": total_time_with_model_sec,
            "total_sync_time_sec": total_sync_time_sec,
            "total_gpu_time_sec": total_gpu_time_sec,
            "total_base_compute_time_sec": total_base_compute_time_sec,
            "total_remat_compute_time_sec": total_remat_compute_time_sec,
            "total_search_time_sec": total_search_time_sec,
            "total_cost_time_sec": total_cost_time_sec,
            "memory_budget_MB": memory_budget,
            "avg_total_mem_MB": total_mem_mb,
            "avg_input_mem_MB": input_mem_mb,
            "avg_model_mem_MB": model_mem_mb,
            "total_remat_count": total_remat_count,
            "total_remat_size_MB": total_remat_size_mb,
        })

    # Create a DataFrame from the summary rows
    summary_df = pd.DataFrame(summary_rows)

    # Round all numeric columns to 3 decimal places for better readability
    numeric_cols = summary_df.select_dtypes(include=["float", "int"]).columns
    summary_df[numeric_cols] = summary_df[numeric_cols].round(3)

    # Write the summary DataFrame to a CSV file
    summary_df.to_csv(output_file, index=False)
    print(f"Summary written to {output_file}")

# Call the function with the input folder containing your CSV files
process_csv_files("cnn-results-2")
