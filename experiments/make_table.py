import json

# Load the JSON data
with open('experiments/benchmark_results/benchmark_checkpoint.json', 'r') as f:
    data = json.load(f)

metrics_summary = []

# Process each model
for model_name, tasks in data.items():
    fitness_scores = []
    auc_scores = []
    validity_rates = []
    duplicate_rates = []
    
    # Iterate through all tasks and their respective runs
    for task_name, runs in tasks.items():
        for run in runs:
            fitness_scores.append(run['fitness'])
            auc_scores.append(run['auc'])
            validity_rates.append(run['validity'])
            duplicate_rates.append(run['duplicate_rate'])
            
    # Calculate means
    mean_fitness = sum(fitness_scores) / len(fitness_scores)
    mean_auc = sum(auc_scores) / len(auc_scores)
    mean_validity = sum(validity_rates) / len(validity_rates)
    mean_duplicate = sum(duplicate_rates) / len(duplicate_rates)
    
    # Calculate uniqueness rate
    mean_uniqueness = 100.0 - mean_duplicate
    
    # Store the results
    metrics_summary.append({
        'Model': model_name,
        'Mean Max Fitness': mean_fitness,
        'Mean AUC': mean_auc,
        'Mean Validity Rate (%)': mean_validity,
        'Mean Duplicate Rate (%)': mean_duplicate,
        'Mean Uniqueness Rate (%)': mean_uniqueness
    })

# Print the results as a formatted markdown-style table
header = f"| {'Model':<10} | {'Mean Max Fitness':<16} | {'Mean AUC':<10} | {'Mean Validity Rate (%)':<22} | {'Mean Duplicate Rate (%)':<23} | {'Mean Uniqueness Rate (%)':<24} |"
print(header)
print("|" + "-"*12 + "|" + "-"*18 + "|" + "-"*12 + "|" + "-"*24 + "|" + "-"*25 + "|" + "-"*26 + "|")

for row in metrics_summary:
    print(f"| {row['Model']:<10} | {row['Mean Max Fitness']:<16.4f} | {row['Mean AUC']:<10.4f} | {row['Mean Validity Rate (%)']:<22.2f} | {row['Mean Duplicate Rate (%)']:<23.2f} | {row['Mean Uniqueness Rate (%)']:<24.2f} |")