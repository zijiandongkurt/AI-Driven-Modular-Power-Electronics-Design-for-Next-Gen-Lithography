import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'axes.titlesize': 16,    
    'axes.labelsize': 13,    
    'xtick.labelsize': 11,   
    'ytick.labelsize': 11,   
    'legend.fontsize': 11,
    'figure.titlesize': 20   
})

# ==============================================================================
# PART 1: THE VISUALIZATION ENGINE
# ==============================================================================

def plot_loss_landscape(loss_function, main_title, filename):
    """
    Takes any loss function and generates the 4-scenario 2x2 grid.
    """
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
    DATA_DIR = os.path.join(PROJECT_ROOT, 'pipeline', 'data')
    os.makedirs(DATA_DIR, exist_ok=True)

    # The four standardized environments
    scenarios = [
        {"title": "Hard Buck (380V in → 5V out)", "v_in": 380, "v_target": 5, "v_min": -50, "v_max": 450},
        {"title": "Hard Boost (3V in → 480V out)", "v_in": 3, "v_target": 480, "v_min": -10, "v_max": 1000},
        {"title": "Easy Buck (12V in → 5V out)", "v_in": 12, "v_target": 5, "v_min": -10, "v_max": 25},
        {"title": "Easy Boost (3V in → 9V out)", "v_in": 3, "v_target": 9, "v_min": -5, "v_max": 20}
    ]

    fig, axs = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(main_title, fontweight='bold', y=0.98)

    for idx, ax in enumerate(axs.flat):
        s = scenarios[idx]
        
        # 1. Generate Data using the injected loss function
        v_outs = np.linspace(s["v_min"], s["v_max"], 1000)
        losses = loss_function(v_outs, s["v_target"], s["v_in"])
        
        # 2. Plot main curve
        ax.plot(v_outs, losses, color='indigo', linewidth=2.5, zorder=3, label="Loss Curve")
        
        # 3. Highlight Target (0 Loss)
        ax.axvline(x=s["v_target"], color='seagreen', linestyle='--', linewidth=2, zorder=2)
        ax.plot(s["v_target"], 0, marker='*', color='seagreen', markersize=15, zorder=4, label=f'Target ({s["v_target"]}V)')
        
        # 4. Highlight Failure State (V_out == V_in)
        loss_at_vin = loss_function(np.array([s["v_in"]]), s["v_target"], s["v_in"])[0]
        ax.axvline(x=s["v_in"], color='crimson', linestyle=':', linewidth=2, zorder=2)
        ax.plot(s["v_in"], loss_at_vin, marker='o', color='crimson', markersize=8, zorder=4, label=f'V_in Failure ({s["v_in"]}V)')
        
        # Add a text annotation for the V_in loss if it's not strictly 1.0
        if round(loss_at_vin, 3) != 1.000:
            ax.text(s["v_in"], loss_at_vin + 0.05, f' L={loss_at_vin:.3f}', color='crimson', fontweight='bold')
             
        # 5. Boundaries and Shading
        ax.axhline(y=1.0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        ax.axhline(y=0.0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        ax.axvspan(s["v_min"], 0, color='salmon', alpha=0.1, label='Wrong Polarity')
        ax.axvspan(0, s["v_max"], color='mediumseagreen', alpha=0.05)

        # 6. Formatting
        ax.set_title(s["title"], fontweight='bold')
        ax.set_xlabel("Generated Output Voltage ($V_{out}$)")
        ax.set_ylabel("Calculated Loss")
        
        # Dynamically scale Y-axis to contain the V_in point but limit extreme explosions
        max_y = max(1.5, min(loss_at_vin * 1.2, 3.0)) 
        ax.set_ylim(-0.1, max_y)
        ax.set_xlim(s["v_min"], s["v_max"])
        ax.grid(True, linestyle='--', alpha=0.6, zorder=1)
        ax.legend(loc='upper right')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    out_path = os.path.join(DATA_DIR, filename)
    try:
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f"✅ Generated: {filename}")
    except PermissionError:
        print(f"❌ PERMISSION ERROR: Cannot overwrite {filename}! Close it and try again.")
    plt.close()

# ==============================================================================
# PART 2: THE MODULAR LOSS FUNCTIONS
# Rule: Every function must accept (v_outs, v_target, v_in) and return an array.
# ==============================================================================

def loss_naive_mse(v_outs, v_target, v_in):
    """
    Naive Mean Squared Error, normalized by the V_in squared error.
    (Added just to prove how plug-and-play this script is!)
    """
    baseline_error = (v_in - v_target) ** 2
    if baseline_error == 0: baseline_error = 1e-6
    
    current_error = (v_outs - v_target) ** 2
    return np.clip(current_error / baseline_error, 0.0, 1.5)

def loss_piecewise_ratio(v_outs, v_target, v_in):
    """
    Iteration 3: The Piecewise Ratio Loss (Asymptotic).
    (Note: v_in is in the signature for API consistency, but unused here)
    """
    safe_target = v_target if v_target != 0 else 1e-6
    x = v_outs / safe_target
    loss = np.zeros_like(x)
    
    pos_mask = x >= 0
    x_pos = x[pos_mask]
    loss[pos_mask] = np.abs(x_pos - 1) / np.maximum(x_pos, 1)
    
    neg_mask = x < 0
    x_neg = x[neg_mask]
    loss[neg_mask] = 1.0 + (np.abs(x_neg) / (1.0 + np.abs(x_neg)))
    
    return loss

def loss_dynamic_baseline(v_outs, v_target, v_in):
    """
    Iteration 4: The Dynamic Baseline Ratio Loss (Linear Gradients).
    Flawlessly anchors V_in failure to exactly 1.0.
    """
    safe_target = v_target if v_target != 0 else 1e-6
    safe_in = v_in if v_in != 0 else 1e-6
    
    baseline_ratio = max(safe_in / safe_target, safe_target / safe_in)
    e_baseline = baseline_ratio - 1.0
    if e_baseline == 0: e_baseline = 1e-6
    
    loss = np.zeros_like(v_outs)
    x = v_outs / safe_target
    
    pos_mask = x > 0
    x_pos = x[pos_mask]
    r_pos = np.maximum(x_pos, 1.0 / x_pos) - 1.0
    loss[pos_mask] = np.clip(r_pos / e_baseline, 0.0, 1.0)
    
    neg_mask = x <= 0
    v_neg = v_outs[neg_mask]
    loss[neg_mask] = 1.0 + (np.abs(v_neg) / safe_target)
    
    return loss

def loss_normalized_mse(v_outs, v_target, v_in):
    """
    Vanilla Normalized MSE.
    Creates a smooth parabola, but suffers from extreme gradient flattening 
    near the target for tasks with large Vin-to-Vtarget gaps.
    """
    baseline_error = (v_in - v_target) ** 2
    if baseline_error == 0: baseline_error = 1e-6 # Failsafe
    
    current_error = (v_outs - v_target) ** 2
    loss = current_error / baseline_error
    
    # Cap at 2.0 to prevent exploding gradients if the model outputs 10,000V
    return np.clip(loss, 0.0, 2.0)

def loss_normalized_mae(v_outs, v_target, v_in):
    """
    Normalized Mean Absolute Error (Root NMSE).
    The Ultimate Solution: Creates a perfect 'V' shape anchored at 1.0 at V_in.
    Provides a constant, unyielding gradient all the way to 0.0.
    """
    baseline_dist = abs(v_in - v_target)
    if baseline_dist == 0: baseline_dist = 1e-6 # Failsafe
    
    current_dist = np.abs(v_outs - v_target)
    loss = current_dist / baseline_dist
    
    # Cap at 2.0 to prevent exploding gradients, but keep it linear everywhere else
    return np.clip(loss, 0.0, 2.0)

def loss_bounded_rational_mae(v_outs, v_target, v_in):
    """
    Strictly bounded [0, 1) Normalized Loss using Rational Compression.
    Anchors the V_in failure at exactly 0.8, leaving the top 20%
    for extreme overshoots to maintain a constant, non-zero gradient to infinity.
    """
    safe_target = v_target if v_target != 0 else 1e-6
    baseline_dist = abs(v_in - safe_target)
    if baseline_dist == 0: baseline_dist = 1e-6 # Failsafe
    
    current_dist = np.abs(v_outs - safe_target)
    
    # 1. Heavy penalty for wrong polarity 
    # This forces negative voltages instantly into the > 0.8 bracket
    polarity_penalty = np.where(
        np.sign(v_outs) != np.sign(safe_target), 
        baseline_dist + np.abs(v_outs), 
        0
    )
    
    # 2. Determine Effective Distance
    e_eff = np.where(polarity_penalty > 0, polarity_penalty, current_dist)
    
    # 3. Normalize against the baseline V_in distance
    R = e_eff / baseline_dist
    
    # 4. Rational Compression: L = R / (R + c)
    # Setting c = 0.25 guarantees that when R = 1 (V_out == V_in), L = 0.8.
    c = 0.25
    
    return R / (R + c)

def loss_continuous_bounded_mae(v_outs, v_target, v_in):
    """
    Iteration 5: Continuous Bounded Rational MAE.
    Solves the discontinuity at 0V by changing the slope instead of jumping.
    Guarantees C0 continuity while keeping the 0.8 Vin anchor and [0,1) bounds.
    """
    safe_target = v_target if v_target != 0 else 1e-6
    baseline_dist = abs(v_in - safe_target)
    if baseline_dist == 0: baseline_dist = 1e-6 # Failsafe
    
    # Calculate the steepness scalar for wrong polarity
    K = max(1.0, baseline_dist / abs(safe_target))
    
    d_eff = np.zeros_like(v_outs)
    
    # Valid side: Same polarity or exactly 0
    valid_mask = (np.sign(v_outs) == np.sign(safe_target)) | (v_outs == 0)
    d_eff[valid_mask] = np.abs(v_outs[valid_mask] - safe_target)
    
    # Wrong polarity side: Distance to 0 + (K * Distance past 0)
    wrong_mask = ~valid_mask
    d_eff[wrong_mask] = np.abs(safe_target) + (K * np.abs(v_outs[wrong_mask]))
    
    # Normalize to ratio R
    R = d_eff / baseline_dist
    
    # Rational Compression (c=0.25 locks R=1 to exactly 0.8 loss)
    c = 0.25
    return R / (R + c)

def loss_log_ratio_compressed(v_outs, v_target, v_in):
    """
    Iteration 6: The Log-Ratio (Bode) Loss.
    Treats voltage errors logarithmically (like decibels), ensuring that
    a 10x overshoot is penalized exactly the same as a 10x undershoot.
    """
    safe_target = v_target if v_target != 0 else 1e-6
    safe_in = v_in if v_in != 0 else 1e-6
    
    # Baseline distance in log space
    baseline_log_dist = abs(np.log(safe_in / safe_target))
    if baseline_log_dist == 0: baseline_log_dist = 1e-6
    
    loss = np.zeros_like(v_outs)
    
    # We must handle positive and negative/zero voltages differently because log(<=0) is NaN
    eps = 1e-3 # Smallest allowed positive voltage for log calculation
    
    # Condition A: Strictly positive voltages
    pos_mask = v_outs > eps
    v_pos = v_outs[pos_mask]
    
    current_log_dist = np.abs(np.log(v_pos / safe_target))
    R_pos = current_log_dist / baseline_log_dist
    
    # Rational compression to bound [0, 1) and anchor V_in at 0.8 (c=0.25)
    c = 0.25
    loss[pos_mask] = R_pos / (R_pos + c)
    
    # Condition B: Negative or zero voltages (Total Polarity Failure)
    # We calculate the max log penalty right at the epsilon boundary...
    max_log_dist = np.abs(np.log(eps / safe_target))
    R_boundary = max_log_dist / baseline_log_dist
    
    neg_mask = ~pos_mask
    v_neg = v_outs[neg_mask]
    
    # ...and add a strict linear penalty for plunging deeper into negative territory.
    # We pass this through the same compression so the curve connects flawlessly.
    R_neg = R_boundary + (np.abs(v_neg - eps) / safe_target)
    loss[neg_mask] = R_neg / (R_neg + c)
    
    return loss

def loss_smooth_log_ratio(v_outs, v_target, v_in):
    """
    Iteration 7: Smooth C1-Continuous Log-Ratio Loss.
    Solves the "kink" at the polarity border by calculating the exact derivative 
    of the log curve at a proportional handoff point (10% of target), 
    and extending that exact slope linearly into the negative domain.
    """
    safe_target = v_target if v_target != 0 else 1e-6
    safe_in = v_in if v_in != 0 else 1e-6
    
    baseline_log_dist = abs(np.log(safe_in / safe_target))
    if baseline_log_dist == 0: baseline_log_dist = 1e-6
    
    # Define the Handoff Point (10% of the target voltage)
    # This is where we switch from Logarithmic to Linear
    v_h = safe_target * 0.1 
    
    # Calculate the R-value exactly at the handoff point
    R_h = np.abs(np.log(v_h / safe_target)) / baseline_log_dist
    
    # Calculate the exact derivative (slope) of the R curve at v_h
    # The derivative of log(x) is 1/x
    slope_h = 1.0 / (v_h * baseline_log_dist)
    
    # Force dtype=float to prevent the plotting bug where integer inputs truncate 0.8 to 0!
    R = np.zeros_like(v_outs, dtype=float)
    
    # Condition A: Above handoff point (Pure Log-Ratio)
    log_mask = v_outs >= v_h
    v_log = v_outs[log_mask]
    R[log_mask] = np.abs(np.log(v_log / safe_target)) / baseline_log_dist
    
    # Condition B: Below handoff point, extending into negative polarity
    # Linear extension using the EXACT matched slope from the handoff point
    lin_mask = ~log_mask
    v_lin = v_outs[lin_mask]
    R[lin_mask] = R_h + slope_h * (v_h - v_lin)
    
    # Rational compression to bound strictly [0, 1) and anchor V_in at 0.8
    c = 0.25
    return R / (R + c)

def loss_log_absolute_error(v_outs, v_target, v_in):
    """
    Iteration 9: Normalized Log of Absolute Error Loss.
    Demonstrates the 'Vanishing Gradient Trap' at extreme overshoots.
    """
    safe_target = v_target if v_target != 0 else 1e-6
    
    # 1. Baseline Error (for V_in)
    baseline_error = np.abs(v_in - safe_target)
    # Use log(1 + error) so that when error is 0, log is 0.
    baseline_log = np.log(1 + baseline_error)
    if baseline_log == 0: baseline_log = 1e-6
    
    # 2. Current Error
    current_error = np.abs(v_outs - safe_target)
    
    # Severe penalty for wrong polarity to keep the left side consistent
    polarity_penalty = np.where(
        np.sign(v_outs) != np.sign(safe_target),
        baseline_error + np.abs(v_outs),
        0
    )
    e_eff = np.where(polarity_penalty > 0, polarity_penalty, current_error)
    
    current_log = np.log(1 + e_eff)
    
    # 3. Normalize against baseline
    R = current_log / baseline_log
    
    # 4. Rational Compression (Anchor V_in to exactly 0.8)
    c = 0.25
    return R / (R + c)

def loss_trizone_hybrid(v_outs, v_target, v_in):
    """
    Iteration 8: The Tri-Zone C1-Continuous Hybrid.
    Combines the symmetry of Log-Ratio near the target with linear tangent 
    extensions at both extremes to permanently cure Vanishing Gradients.
    """
    safe_target = v_target if v_target != 0 else 1e-6
    safe_in = v_in if v_in != 0 else 1e-6
    
    baseline_log_dist = abs(np.log(safe_in / safe_target))
    if baseline_log_dist == 0: baseline_log_dist = 1e-6
    
    # --- Define Handoff Points ---
    v_low = safe_target * 0.1  # 10% of target
    v_high = max(safe_in * 1.5, safe_target * 10.0) # 150% of Vin or 10x target
    
    # --- Calculate R and exact slopes at Handoff Points ---
    # Lower Handoff
    R_low = np.abs(np.log(v_low / safe_target)) / baseline_log_dist
    slope_low = 1.0 / (v_low * baseline_log_dist)
    
    # Upper Handoff
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

if __name__ == "__main__":
    print("📊 Running modular loss plotting suite...")
    """
    # Run 1: MSE Test
    plot_loss_landscape(
        loss_function=loss_naive_mse, 
        main_title='Naive MSE Loss: Terrible Asymmetry (Notice the Boost Curves)', 
        filename='loss_01_naive_mse.png'
    )
    
    # Run 2: The Piecewise Asymptote
    plot_loss_landscape(
        loss_function=loss_piecewise_ratio, 
        main_title='Piecewise Ratio Loss: Flat Gradients at Extremes', 
        filename='loss_02_piecewise_ratio.png'
    )
    
    # Run 3: The Winning Dynamic Baseline
    plot_loss_landscape(
        loss_function=loss_dynamic_baseline, 
        main_title='Dynamic Ratio Loss: Task-Anchored Linear Gradients', 
        filename='loss_03_dynamic_baseline.png'
    )

    # Run 4: The Winning Dynamic Baseline
    plot_loss_landscape(
        loss_function=loss_normalized_mse, 
        main_title='Normalized MSE Loss: Smooth Parabola but Extreme Flattening Near Target', 
        filename='loss_04_normalized_mse.png'
    )

    # Run 5: The Winning Dynamic Baseline
    plot_loss_landscape(
        loss_function=loss_normalized_mae, 
        main_title='Normalized MAE Loss: Robust to Outliers with Steeper Gradients', 
        filename='loss_05_normalized_mae.png'
    )

    # Run 6: The 0 to 1 Bounded Rational Compression
    plot_loss_landscape(
        loss_function=loss_bounded_rational_mae, 
        main_title='Bounded Rational MAE: Strictly [0,1) with Infinite Gradients', 
        filename='loss_06_bounded_rational.png'
    )

    # Run 7: The Continuous Bounded Rational MAE (Final Iteration)
    plot_loss_landscape(
        loss_function=loss_continuous_bounded_mae, 
        main_title='Continuous Bounded Rational MAE: Solves the 0V Discontinuity', 
        filename='loss_07_continuous_bounded.png'
    )

    # Run 8: The Log-Ratio (Bode) Loss
    plot_loss_landscape(
        loss_function=loss_log_ratio_compressed, 
        main_title='Log-Ratio Loss: Logarithmic Treatment of Voltage Errors', 
        filename='loss_08_log_ratio.png'
    )

    # Run 9: The Smooth Log-Ratio Loss
    plot_loss_landscape(
        loss_function=loss_smooth_log_ratio, 
        main_title='Smooth Log-Ratio Loss: C1-Continuous Logarithmic Loss', 
        filename='loss_09_smooth_log_ratio.png'
    )

    # Run 10: The Log of Absolute Error Loss
    plot_loss_landscape(
        loss_function=loss_log_absolute_error, 
        main_title='Log Absolute Error Loss: Watch the Vanishing Gradient Trap at Extreme Overshoots', 
        filename='loss_10_log_absolute_error.png'
    )"""

    # Run 11: The Tri-Zone Hybrid
    plot_loss_landscape(
        loss_function=loss_trizone_hybrid, 
        main_title='Tri-Zone Hybrid Loss: Log-Ratio Core with Linear Extensions to Cure Vanishing Gradients', 
        filename='loss_11_trizone_hybrid.png'
    )
    
    print("🎉 All plots generated successfully!")