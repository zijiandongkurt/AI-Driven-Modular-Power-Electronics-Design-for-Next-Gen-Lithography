import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set styling for IEEE academic look
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.5)

# ---------------------------------------------------------
# Dynamic Path Resolution for New Folder Structure
# ---------------------------------------------------------
script_dir = Path(__file__).resolve().parent
datasets_dir = script_dir.parent / "data" / "datasets"

# Load the JSON files using the robust paths (Removed "Phase" nomenclature)
files = {
    "Easy": datasets_dir / "constraints_easy.json",
    "Medium": datasets_dir / "constraints_medium.json",
    "Hard": datasets_dir / "constraints_hard.json"
}

data = []

for tier, filepath in files.items():
    if not filepath.exists():
        print(f"Warning: Could not find {filepath}")
        continue
        
    with open(filepath, 'r') as f:
        constraints = json.load(f)
        for c in constraints:
            # Calculate the average input voltage for the ratio
            v_in_avg = (c['vin_min'] + c['vin_max']) / 2.0
            v_out = c['vout_target']
            
            # Calculate conversion ratio (always >= 1 for scale)
            ratio = v_out / v_in_avg if v_out > v_in_avg else v_in_avg / v_out
            
            data.append({
                "Difficulty Tier": tier,
                "Power In (W)": c['power_in'],
                "Target Efficiency (%)": c['efficiency_target'] * 100,
                "Conversion Ratio": ratio
            })

df = pd.DataFrame(data)

# Create the plot figure
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
palette = ["#2ecc71", "#f1c40f", "#e74c3c"] # Green, Yellow, Red

# Plot 1: Input Power
sns.boxplot(x="Difficulty Tier", y="Power In (W)", data=df, ax=axes[0], palette=palette)
axes[0].set_title("Power Requirements")
axes[0].set_yscale("log") 
axes[0].set_ylabel("Power Input (Watts) [Log Scale]")

# Plot 2: Conversion Ratio
sns.boxplot(x="Difficulty Tier", y="Conversion Ratio", data=df, ax=axes[1], palette=palette)
axes[1].set_title("Voltage Conversion Ratio ($V_{high} / V_{low}$)")
axes[1].set_yscale("log")
axes[1].set_ylabel("Ratio magnitude [Log Scale]")

# Plot 3: Target Efficiency
sns.boxplot(x="Difficulty Tier", y="Target Efficiency (%)", data=df, ax=axes[2], palette=palette)
axes[2].set_title("Target Efficiency")
axes[2].set_ylabel("Efficiency (%)")

# Formatting
for ax in axes:
    ax.tick_params(axis='x', rotation=0) # Removed rotation since names are short now
    ax.set_xlabel("")

plt.tight_layout()

# Save the output image in the same directory as this script
output_path = script_dir / "constraint_spread.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Successfully generated the figure at: {output_path}")