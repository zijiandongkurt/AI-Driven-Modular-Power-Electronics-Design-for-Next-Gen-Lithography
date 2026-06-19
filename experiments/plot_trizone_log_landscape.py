import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Tailored font settings for readability
plt.rcParams.update({
    'axes.titlesize': 16,    
    'axes.labelsize': 13,    
    'xtick.labelsize': 11,   
    'ytick.labelsize': 11,   
    'legend.fontsize': 10,
    'figure.titlesize': 20   
})

def loss_trizone_hybrid(v_outs, v_target, v_in):
    """
    The Tri-Zone C1-Continuous Hybrid.
    Combines the symmetry of Log-Ratio near the target with linear tangent 
    extensions at both extremes to permanently cure Vanishing Gradients.
    """
    safe_target = v_target if v_target != 0 else 1e-6
    safe_in = v_in if v_in != 0 else 1e-6
    
    baseline_log_dist = abs(np.log(safe_in / safe_target))
    if baseline_log_dist == 0: baseline_log_dist = 1e-6
    
    # --- Define Handoff Points ---
    v_low = safe_target * 0.1  
    v_high = max(safe_in * 1.5, safe_target * 10.0) 
    
    # --- Calculate R and exact slopes at Handoff Points ---
    R_low = np.abs(np.log(v_low / safe_target)) / baseline_log_dist
    slope_low = 1.0 / (v_low * baseline_log_dist)
    
    R_high = np.abs(np.log(v_high / safe_target)) / baseline_log_dist
    slope_high = 1.0 / (v_high * baseline_log_dist)
    
    R = np.zeros_like(v_outs, dtype=float)
    
    # --- Zone 2: The Logarithmic Goldilocks Zone ---
    log_mask = (v_outs >= v_low) & (v_outs <= v_high)
    v_log = v_outs[log_mask]
    R[log_mask] = np.abs(np.log(v_log / safe_target)) / baseline_log_dist
    
    # --- Zone 1: Negative / Undershoot (Linear Tangent) ---
    low_mask = v_outs < v_low
    v_under = v_outs[low_mask]
    R[low_mask] = R_low + slope_low * (v_low - v_under)
    
    # --- Zone 3: Extreme Overshoot (Linear Tangent) ---
    high_mask = v_outs > v_high
    v_over = v_outs[high_mask]
    R[high_mask] = R_high + slope_high * (v_over - v_high)
    
    # --- Rational Compression [0, 1) ---
    c = 0.25
    return R / (R + c)

def get_scenarios():
    return [
        {"title": "Hard Buck (380V in → 5V out)", "v_in": 380, "v_target": 5, "v_min": 0.1, "v_max": 10000},
        {"title": "Hard Boost (3V in → 480V out)", "v_in": 3, "v_target": 480, "v_min": 0.1, "v_max": 10000},
        {"title": "Easy Buck (12V in → 5V out)", "v_in": 12, "v_target": 5, "v_min": 0.1, "v_max": 1000},
        {"title": "Easy Boost (3V in → 9V out)", "v_in": 3, "v_target": 9, "v_min": 0.1, "v_max": 1000}
    ]

def plot_loss_landscape_logx(loss_function, main_title, filename, data_dir):
    scenarios = get_scenarios()
    fig, axs = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(main_title, fontweight='bold', y=0.98)

    for idx, ax in enumerate(axs.flat):
        s = scenarios[idx]
        v_outs = np.logspace(np.log10(s["v_min"]), np.log10(s["v_max"]), 1000)
        losses = loss_function(v_outs, s["v_target"], s["v_in"])
        
        v_low = s["v_target"] * 0.1
        v_high = max(s["v_in"] * 1.5, s["v_target"] * 10.0)
        
        ax.axvspan(s["v_min"], v_low, color='salmon', alpha=0.1, label='Zone 1 (Linear Recovery)')
        ax.axvspan(v_low, v_high, color='mediumseagreen', alpha=0.08, label='Zone 2 (Log Goldilocks)')
        ax.axvspan(v_high, s["v_max"], color='orange', alpha=0.1, label='Zone 3 (Linear Escape)')

        ax.plot(v_outs, losses, color='indigo', linewidth=2.5, zorder=3, label="Loss Curve")
        
        ax.axvline(x=s["v_target"], color='seagreen', linestyle='--', linewidth=2, zorder=2)
        ax.plot(s["v_target"], 0, marker='*', color='seagreen', markersize=15, zorder=4, label=f'Target ({s["v_target"]}V)')
        
        loss_at_vin = loss_function(np.array([s["v_in"]]), s["v_target"], s["v_in"])[0]
        ax.axvline(x=s["v_in"], color='crimson', linestyle=':', linewidth=2, zorder=2)
        ax.plot(s["v_in"], loss_at_vin, marker='o', color='crimson', markersize=8, zorder=4, label=f'V_in Failure ({s["v_in"]}V)')
        
        if round(loss_at_vin, 3) != 1.000:
            ax.text(s["v_in"] * 1.15, loss_at_vin + 0.05, f' L={loss_at_vin:.3f}', color='crimson', fontweight='bold')
             
        ax.axhline(y=1.0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        ax.axhline(y=0.0, color='black', linestyle='-', linewidth=1, alpha=0.5)

        ax.set_title(s["title"], fontweight='bold')
        ax.set_xlabel("Generated Output Voltage ($V_{out}$) [Log Scale]")
        ax.set_ylabel("Calculated Loss")
        ax.set_xscale('log')
        ax.set_ylim(-0.05, 1.1)
        ax.set_xlim(s["v_min"], s["v_max"])
        ax.grid(True, which="both", linestyle='--', alpha=0.4, zorder=1)
        ax.legend(loc='lower right' if s["v_in"] < s["v_target"] else 'lower left')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_path = os.path.join(data_dir, filename)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Loss Plot saved to: {out_path}")

def plot_gradient_landscape_logx(loss_function, main_title, filename, data_dir):
    scenarios = get_scenarios()
    fig, axs = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(main_title, fontweight='bold', y=0.98)

    for idx, ax in enumerate(axs.flat):
        s = scenarios[idx]
        
        # High resolution for accurate numerical gradients
        v_outs = np.logspace(np.log10(s["v_min"]), np.log10(s["v_max"]), 5000)
        losses = loss_function(v_outs, s["v_target"], s["v_in"])
        
        # Calculate numerical gradient (derivative) wrt V_out
        gradients = np.gradient(losses, v_outs)
        abs_grads = np.abs(gradients)
        
        v_low = s["v_target"] * 0.1
        v_high = max(s["v_in"] * 1.5, s["v_target"] * 10.0)
        
        # Shade the 3 Zones
        ax.axvspan(s["v_min"], v_low, color='salmon', alpha=0.1, label='Zone 1 (Linear Recovery)')
        ax.axvspan(v_low, v_high, color='mediumseagreen', alpha=0.08, label='Zone 2 (Log Goldilocks)')
        ax.axvspan(v_high, s["v_max"], color='orange', alpha=0.1, label='Zone 3 (Linear Escape)')

        # Plot the Gradient Magnitude
        ax.plot(v_outs, abs_grads, color='dodgerblue', linewidth=2.5, zorder=3, label="|Gradient Magnitude|")
        
        # Mark Target and Vin
        ax.axvline(x=s["v_target"], color='seagreen', linestyle='--', linewidth=2, zorder=2, label=f'Target ({s["v_target"]}V)')
        ax.axvline(x=s["v_in"], color='crimson', linestyle=':', linewidth=2, zorder=2, label=f'V_in Failure ({s["v_in"]}V)')
        
        ax.set_title(s["title"] + " Gradients", fontweight='bold')
        ax.set_xlabel("Generated Output Voltage ($V_{out}$) [Log Scale]")
        ax.set_ylabel("Absolute Gradient |dL / dV| [Log Scale]")
        
        # Both axes must be Log to see the 1/x slope and constant floors clearly
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        ax.set_xlim(s["v_min"], s["v_max"])
        ax.grid(True, which="both", linestyle='--', alpha=0.4, zorder=1)
        ax.legend(loc='upper right')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_path = os.path.join(data_dir, filename)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Gradient Plot saved to: {out_path}")

if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
    DATA_DIR = os.path.join(PROJECT_ROOT, 'pipeline', 'data')
    os.makedirs(DATA_DIR, exist_ok=True)

    print("📊 Generating Tri-Zone Landscapes...")
    
    # 1. The Standard Loss Plot
    plot_loss_landscape_logx(
        loss_function=loss_trizone_hybrid, 
        main_title='Tri-Zone Hybrid Loss: Evaluated on a Logarithmic Voltage Scale', 
        filename='loss_11_trizone_logx.png',
        data_dir=DATA_DIR
    )
    
    # 2. The New Gradient Magnitude Plot
    plot_gradient_landscape_logx(
        loss_function=loss_trizone_hybrid, 
        main_title='Learning Signal Strength: Gradient Magnitude (|dL/dV|) of Tri-Zone Loss', 
        filename='loss_11_trizone_gradients.png',
        data_dir=DATA_DIR
    )
    
    print("🎉 All plots generated successfully!")