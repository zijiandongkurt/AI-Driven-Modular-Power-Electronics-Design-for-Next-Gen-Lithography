import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_all_datasets(base_dir="constraint_datasets"):
    """
    Crawls the dataset directory structure and loads all CSVs into a single 
    DataFrame, adding tags for topology and ratio category.
    """
    all_data = []
    
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"Directory '{base_dir}' not found. Run the generator script first.")

    for topology in os.listdir(base_dir):
        topo_path = os.path.join(base_dir, topology)
        if not os.path.isdir(topo_path):
            continue
            
        for ratio_cat in os.listdir(topo_path):
            ratio_path = os.path.join(topo_path, ratio_cat)
            if not os.path.isdir(ratio_path):
                continue
                
            for file in os.listdir(ratio_path):
                if file.endswith(".csv"):
                    df = pd.read_csv(os.path.join(ratio_path, file))
                    # Tag the data so we can color-code the plots
                    df['topology'] = topology
                    df['ratio_category'] = ratio_cat
                    all_data.append(df)
                    
    return pd.concat(all_data, ignore_index=True)


def plot_vin_vs_vout(df):
    """
    Creates a grid of scatter plots showing Vout vs Vin_min across topologies.
    Uses log scales due to the massive 1000x ratio ranges.
    """
    # Set the visual style
    sns.set_theme(style="whitegrid")
    
    # Create a faceted plot (one subplot per topology)
    g = sns.FacetGrid(df, col="topology", hue="ratio_category", 
                      col_wrap=3, height=5, aspect=1.2, 
                      palette="viridis", sharex=False, sharey=False)
    
    # Map the scatter plot (Target Vout vs Minimum required Vin)
    g.map(sns.scatterplot, "vout_target", "vin_min", alpha=0.6, edgecolor=None)
    
    # Formatting
    g.add_legend(title="Conversion Ratio")
    g.set_axis_labels("Target V_out (Volts)", "Minimum V_in (Volts)")
    g.set_titles(col_template="{col_name} Topology")
    
    # Apply log scales to handle the 1000x ratios gracefully
    for ax in g.axes.flat:
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        # Draw a diagonal line (Vin = Vout) as a physical reference boundary
        lims = [
            min(ax.get_xlim()[0], ax.get_ylim()[0]),  
            max(ax.get_xlim()[1], ax.get_ylim()[1])
        ]
        ax.plot(lims, lims, 'r--', alpha=0.5, zorder=0)
        ax.text(lims[0]*1.2, lims[0]*1.5, 'V_in = V_out', color='red', alpha=0.7)
   # Add the main title
    g.fig.suptitle("Voltage Boundaries: V_out vs V_in (Log Scale)", fontsize=16)
    
    # 1. Tell the plot to calculate tight margins to fit all labels
    plt.tight_layout()
    
    # 2. Push the top down slightly so the tight_layout doesn't overlap the main title
    plt.subplots_adjust(top=0.85)
    
    # 3. Render
    plt.show()

def plot_all_vin_vs_vout_overlay(df):
    """
    Creates a single overlay scatter plot of Vout vs Vin_min for ALL topologies 
    and ratios to identify gaps in the training data space.
    """
    plt.figure(figsize=(12, 8))
    sns.set_theme(style="whitegrid")
    
    # Create the scatter plot overlaying everything
    # Hue (Color) = Topology
    # Style (Marker Shape) = Conversion Ratio Category
    ax = sns.scatterplot(
        data=df, 
        x="vout_target", 
        y="vin_min", 
        hue="topology", 
        style="ratio_category", 
        alpha=0.6,          # Transparency to see overlaps
        palette="bright",   # High contrast colors
        s=40                # Marker size
    )
    
    # Set logarithmic scales
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # Draw the V_in = V_out boundary line
    lims = [
        min(ax.get_xlim()[0], ax.get_ylim()[0]),  
        max(ax.get_xlim()[1], ax.get_ylim()[1])
    ]
    plt.plot(lims, lims, 'k--', alpha=0.5, zorder=0, label="V_in = V_out")
    plt.text(lims[0]*1.2, lims[0]*1.5, 'V_in = V_out', color='black', alpha=0.7)
    
    # Formatting the titles and labels
    plt.title("Total Voltage Coverage Space (Identifying Training Gaps)", fontsize=16)
    plt.xlabel("Target V_out (Volts)", fontsize=12)
    plt.ylabel("Minimum V_in (Volts)", fontsize=12)
    
    # Move the legend outside the plot so it doesn't cover data points
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    
    plt.tight_layout()
    plt.show()


def plot_efficiency_vs_power(df):
    """
    Creates a scatter plot showing the distribution of Power vs Efficiency.
    """
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="darkgrid")
    
    # Scatter plot
    sns.scatterplot(data=df, x="power_in", y="efficiency_target", 
                    hue="topology", style="ratio_category", 
                    alpha=0.5, palette="Set1")
    
    plt.title("Constraint Distribution: Input Power vs Target Efficiency", fontsize=14)
    plt.xlabel("Input Power (Watts)")
    plt.ylabel("Target Efficiency")
    
    # Put legend outside the plot
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

def plot_input_tolerance(df):
    """
    Plots Vin_max vs Vin_min to visualize the input voltage operating bands.
    Points further from the diagonal line represent circuits that must survive
    wild voltage swings.
    """
    plt.figure(figsize=(8, 8))
    sns.set_theme(style="whitegrid")
    
    ax = sns.scatterplot(
        data=df, 
        x="vin_min", 
        y="vin_max", 
        hue="topology", 
        alpha=0.6,
        palette="viridis",
        s=30
    )
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # Draw the boundary line (Vin_max = Vin_min, which is a 0% tolerance circuit)
    lims = [
        min(ax.get_xlim()[0], ax.get_ylim()[0]),  
        max(ax.get_xlim()[1], ax.get_ylim()[1])
    ]
    plt.plot(lims, lims, 'k--', alpha=0.5, label="Zero Tolerance (Vin_max = Vin_min)")
    
    plt.title("Input Tolerance Range: Vin_max vs Vin_min", fontsize=14)
    plt.xlabel("Minimum Required V_in (Volts)")
    plt.ylabel("Maximum Required V_in (Volts)")
    plt.legend(loc='upper left')
    
    plt.tight_layout()
    plt.show()


def plot_current_stress(df):
    """
    Estimates output current (Pin / Vout) to check if the dataset asks for 
    physically impossible high-current designs.
    """
    # Create a temporary column for estimated current
    # Assuming rough 100% efficiency for the sake of plotting the stress requirement
    df_plot = df.copy()
    df_plot['estimated_i_out'] = df_plot['power_in'] / df_plot['vout_target']
    
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="ticks")
    
    # Using a log scale for both because current will swing from 0.1A to 1000A
    ax = sns.scatterplot(
        data=df_plot, 
        x="vout_target", 
        y="estimated_i_out", 
        hue="topology", 
        style="ratio_category",
        alpha=0.6,
        palette="rocket"
    )
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # Draw a "Danger Zone" line at 50 Amps (arbitrary threshold for single-phase difficulty)
    plt.axhline(y=50, color='r', linestyle=':', label='High Current Stress (>50A)')
    
    plt.title("Physical Stress: V_out vs Estimated Output Current", fontsize=14)
    plt.xlabel("Target V_out (Volts)")
    plt.ylabel("Estimated Output Current (Amps)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
# Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # --- CHANGED: Point directly to the Capped folder ---
    main_dataset_dir = os.path.join(script_dir, "constraint_datasets", "Phase1_Capped")
    
    print(f"Loading datasets from: {main_dataset_dir}")
    try:
        combined_df = load_all_datasets(main_dataset_dir)
        print(f"Successfully loaded {len(combined_df)} total constraints.")
        
        print("Generating Voltage plot...")
        plot_vin_vs_vout(combined_df)

        print("Generating Overlay plot...")
        plot_all_vin_vs_vout_overlay(combined_df)
        
        print("Generating Efficiency/Power plot...")
        plot_efficiency_vs_power(combined_df)

        print("Generating Tolerance plot...")
        plot_input_tolerance(combined_df)
        
        print("Generating Current Stress plot...")
        plot_current_stress(combined_df)
    
        
    except Exception as e:
        print(f"Error: {e}")