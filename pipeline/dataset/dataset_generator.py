import os
import random
import pandas as pd

def generate_datasets(
    num_samples_per_dataset=1000, 
    base_dir="datasets",
    enforce_current_limit=True,   # <-- The curriculum learning toggle
    max_allowed_current=40.0,     # <-- The current stress cap
    max_tolerance_pct=20.0        # <-- ALWAYS active: Tolerance boundary in %
):
    """
    Generates 9 distinct datasets for power converter constraints, categorized by 
    topology (Step-up, Step-down, Buck-Boost) and Conversion Ratio.
    """
    
    topologies = ["step_up", "step_down", "buck_boost"]
    
    ratio_categories = [
        ("ratio_0_to_10", 1.1, 10.0),
        ("ratio_10_to_100", 10.0, 100.0),
        ("ratio_100_to_1000", 100.0, 1000.0)
    ]

    for topology in topologies:
        for ratio_name, r_min, r_max in ratio_categories:
            data = []
            
            for _ in range(num_samples_per_dataset):
                valid_point_found = False
                
                while not valid_point_found:
                    
                    R = random.uniform(r_min, r_max)
                    base_v = random.uniform(3.3, 100.0) 
                    
                    # --- ALWAYS Active: Calculate Tolerance Multipliers ---
                    # Convert percentage to a decimal (e.g., 20.0 -> 0.20)
                    max_dev = max_tolerance_pct / 100.0
                    
                    # Randomly select the severity of the drop and spike within the limit
                    drop_multiplier = random.uniform(1.0 - max_dev, 1.0)
                    spike_multiplier = random.uniform(1.0, 1.0 + max_dev)
                    
                    # --- Topology Logic ---
                    if topology == "step_up":
                        vout_target = base_v * R
                        vin_nom = base_v
                        
                        vin_min = vin_nom * drop_multiplier
                        # Safety clamp: Ensure vin_max never exceeds vout_target
                        vin_max = min(vin_nom * spike_multiplier, vout_target * 0.95)

                    elif topology == "step_down":
                        vout_target = base_v
                        vin_nom = base_v * R
                        
                        # Safety clamp: Ensure vin_min never drops below vout_target
                        vin_min = max(vin_nom * drop_multiplier, vout_target * 1.05)
                        vin_max = vin_nom * spike_multiplier

                    elif topology == "buck_boost":
                        vout_target = base_v
                        if random.choice([True, False]):
                            vin_min = vout_target / R
                            vin_max = vout_target * random.uniform(1.2, 3.0) 
                        else:
                            vin_max = vout_target * R
                            vin_min = vout_target / random.uniform(1.2, 3.0)

                    # Generate Power
                    p_in = random.uniform(10.0, 500.0)
                    
                    # --- The Curriculum Toggle: Check Current Limit ---
                    if enforce_current_limit:
                        estimated_current = p_in / vout_target
                        if estimated_current <= max_allowed_current:
                            valid_point_found = True
                    else:
                        # If curriculum mode requires high current, accept anything
                        valid_point_found = True
                        
                # --- We only reach here once a valid point is accepted ---
                eff = random.uniform(0.75, 0.98)

                data.append({
                    "vin_min": round(vin_min, 2),
                    "vin_max": round(vin_max, 2),
                    "vout_target": round(vout_target, 2),
                    "efficiency_target": round(eff, 2),
                    "power_in": round(p_in, 2)
                })

            target_dir = os.path.join(base_dir, topology, ratio_name)
            os.makedirs(target_dir, exist_ok=True)
            
            df = pd.DataFrame(data)
            filename = f"{topology}_{ratio_name}.csv"
            filepath = os.path.join(target_dir, filename)
            
            df.to_csv(filepath, index=False)
            print(f"Generated {filepath} ({num_samples_per_dataset} rows)")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    main_dataset_dir = os.path.join(script_dir, "constraint_datasets")
    
    print(f"Starting generation in: {main_dataset_dir}\n")
    
    # Generate Phase 1: Easy circuits (Current strictly capped at 40A)
    generate_datasets(
        num_samples_per_dataset=1000, 
        base_dir=os.path.join(main_dataset_dir, "Phase1_Capped"),
        enforce_current_limit=True,  
        max_allowed_current=40.0,
        max_tolerance_pct=20.0
    )
    
    # Generate Phase 2: Unrestricted circuits for advanced training
    generate_datasets(
        num_samples_per_dataset=1000, 
        base_dir=os.path.join(main_dataset_dir, "Phase2_Unrestricted"),
        enforce_current_limit=False,  # Unleash the high amps!
        max_tolerance_pct=20.0
    )
    
    print("\nGeneration Complete!")