# find_best_cutoff.py
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind

# ── PARAMETERS ────────────────────────────────────────────────────────────────

# 1) Path to the CSV that has one row per subject, including columns:
#    'subject', 'accuracy_pooled', 'accuracy_delta'
CSV_PATH = "tl_subject_level_comparison.csv"

# 2) Which cutoffs to test? (e.g. from 0.50 to 0.90 in steps of 0.05)
CUT_OFFS = np.arange(0.50, 0.91, 0.05)  # [0.50, 0.55, 0.60, …, 0.90]

# 3) Significance level (for reporting only)
ALPHA = 0.05

# ── LOAD YOUR DATA ───────────────────────────────────────────────────────────

df = pd.read_csv(CSV_PATH)

# Check that required columns exist:
for c in ["accuracy_pooled", "accuracy_delta"]:
    if c not in df.columns:
        raise RuntimeError(f"Column '{c}' not found in {CSV_PATH}.")

# ── FUNCTION TO TEST A SINGLE CUTOFF ──────────────────────────────────────────

def test_cutoff(df, cutoff):
    """
    Split df into:
      low_baseline  = df[ accuracy_pooled < cutoff ]
      high_baseline = df[ accuracy_pooled >= cutoff ]
    Compute:
      - N_low, N_high
      - mean Δ_low, mean Δ_high
      - Welch’s t-test (one-tailed) testing Δ_low > Δ_high
    Returns a dict with all relevant values.
    """
    low_df  = df[df["accuracy_pooled"] < cutoff]
    high_df = df[df["accuracy_pooled"] >= cutoff]

    N_low  = len(low_df)
    N_high = len(high_df)

    mean_low  = low_df["accuracy_delta"].mean()  if N_low  > 0 else np.nan
    mean_high = high_df["accuracy_delta"].mean() if N_high > 0 else np.nan

    # If one group is too small, skip t-test:
    if (N_low < 2) or (N_high < 2):
        return {
            "cutoff": cutoff,
            "N_low": N_low,
            "N_high": N_high,
            "mean_low": mean_low,
            "mean_high": mean_high,
            "t_stat": np.nan,
            "p_one_tailed": np.nan
        }

    # Perform Welch’s t-test (two-sided) then convert to one-tailed p-value.
    low_vals  = low_df ["accuracy_delta"].values
    high_vals = high_df["accuracy_delta"].values
    t_two, p_two = ttest_ind(low_vals, high_vals, equal_var=False)

    if np.isnan(t_two) or np.isnan(p_two):
        p_one = np.nan
    else:
        # One‐tailed p for “Δ_low > Δ_high”
        if t_two > 0:
            p_one = p_two / 2.0
        else:
            p_one = 1.0 - (p_two / 2.0)

    return {
        "cutoff": cutoff,
        "N_low": N_low,
        "N_high": N_high,
        "mean_low": mean_low,
        "mean_high": mean_high,
        "t_stat": t_two,
        "p_one_tailed": p_one
    }

# ── RUN THROUGH ALL CUT-OFFS ──────────────────────────────────────────────────

results = []
for c in CUT_OFFS:
    info = test_cutoff(df, c)
    results.append(info)

results_df = pd.DataFrame(results)

# ── PRINT A SUMMARY TABLE ─────────────────────────────────────────────────────

print("\nCutoff |  N_low | N_high | mean Δ_low | mean Δ_high |   t-stat  |  p (1-tailed)")
print("----------------------------------------------------------------------------")
for _, row in results_df.iterrows():
    cutoff    = row["cutoff"]
    N_low     = int(row["N_low"])
    N_high    = int(row["N_high"])
    mean_low  = row["mean_low"]
    mean_high = row["mean_high"]
    t_stat    = row["t_stat"]
    p_one     = row["p_one_tailed"]

    # Corrected f-string formatting: no extra spaces inside format specs
    print(f"{cutoff: .2f}   | {N_low:4d}  | {N_high:4d}   |"
          f"  {mean_low:+.4f}   |   {mean_high:+.4f}   |"
          f"  {t_stat: .3f}  |  {p_one:.4f}")

# ── SHOW WHICH CUT-OFFS GIVE p_one_tailed < ALPHA ─────────────────────────────

sig_df = results_df[results_df["p_one_tailed"] < ALPHA]
if len(sig_df) == 0:
    print(f"\nNo cut-off in {CUT_OFFS.tolist()} produced p_one_tailed < {ALPHA:.3f}.")
else:
    print(f"\nCut-offs that yield p_one_tailed < {ALPHA:.3f}:")
    for _, r in sig_df.iterrows():
        print(f"  • cutoff = {r['cutoff']:.2f},   t = {r['t_stat']:.3f},   p_one_tailed = {r['p_one_tailed']:.4f}")

# ── SAVE RESULTS TO CSV FOR LATER INSPECTION ─────────────────────────────────

results_df.to_csv("baseline_cutoff_sweep_results.csv", index=False)
print("\nFull results saved to 'baseline_cutoff_sweep_results.csv'.\n")
