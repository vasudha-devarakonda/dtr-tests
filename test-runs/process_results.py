import pandas as pd
import glob
import os

def process_csv_files(input_folder, output_file="summary.csv"):
    summary_rows = []

    csv_files = glob.glob(os.path.join(input_folder, "*.csv"))

    for file in csv_files:
        df = pd.read_csv(file)


        model_name = df["model_name"].iloc[0]
        batch_size = df["batch_size"].iloc[0]
        memory_budget = df["memory_budget"].iloc[0] / (1024 * 1024) 

        total_time_sec = df["time"].sum() / 1000.0


        total_search_time_sec = df["search_time"].sum() / 1000.0
        total_cost_time_sec = df["cost_time"].sum() / 1000.0
        total_remat_compute_time_sec = df["remat_compute_time"].sum() / 1000.0


        total_mem_mb = (df["total_mem"].mean() * 1e6) / (1024 * 1024)
        input_mem_mb = (df["input_mem"].mean() * 1e6) / (1024 * 1024)
        model_mem_mb = (df["model_mem"].mean() * 1e6) / (1024 * 1024)

        num_reps = df["rep"].nunique()
        num_epochs = df["epoch"].nunique()
        num_iterations = num_reps / num_epochs if num_epochs != 0 else None

        summary_rows.append({
            "model_name": model_name,
            "batch_size": batch_size,
            "num_iterations": num_iterations,
            "total_time_sec": total_time_sec,
            "total_search_time_sec": total_search_time_sec,
            "total_cost_time_sec": total_cost_time_sec,
            "total_remat_compute_time_sec": total_remat_compute_time_sec,
            "memory_budget_MB": memory_budget,
            "avg_total_mem_MB": total_mem_mb,
            "avg_input_mem_MB": input_mem_mb,
            "avg_model_mem_MB": model_mem_mb,
        })

    summary_df = pd.DataFrame(summary_rows)
    numeric_cols = summary_df.select_dtypes(include=["float", "int"]).columns
    summary_df[numeric_cols] = summary_df[numeric_cols].round(3)

    summary_df.to_csv(output_file, index=False)
    print(f"Summary written to {output_file}")


process_csv_files("results2")
