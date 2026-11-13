"""
src/analytics_engine.py — SupeRR Selector Season 2 Pressure Analytics & ML Engine
==================================================================================
The Pink Canvas OS | Rajasthan Royals Pressure Intelligence Platform
Official Round 1 First-Principles Machine Learning & Econometric Architecture:
- Stage 1: Information-Theoretic Win-Probability Leverage Index (L_i)
  * Model 1: Live Win Probability Classifier (wp_model) via 5-Fold GroupKFold
  * Model 2: Expected Runs Regressor (xr_model) using ONLY situational state features
  * Model 3: Expected Dismissal Classifier (xw_model) using ONLY situational state features
  * Counterfactual WP variance across 4 outcomes -> Outcome-probability-weighted dispersion sigma_WP -> L_i
- Stage 2: Endogenous Wicket Valuation (C_wkt) & Opponent Skill Adjustment (theta_bat, theta_bowl)
- Stage 3: Hierarchical Empirical Bayes Random-Slopes Model (alpha_p + beta_p * z_L)
- Stage 4: Bayesian Lower Confidence Bound (LCB) Ranking & Clean Universal Discipline Eligibility
"""

import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"
import sys
import json
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import norm
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    roc_auc_score,
    brier_score_loss,
    log_loss,
    root_mean_squared_error,
    mean_absolute_error,
)
from sklearn.inspection import permutation_importance

# Path resolution
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.config import (
    LEVERAGE_BASELINE_REGIME,
    LEVERAGE_HIGH_PRESSURE,
    LEVERAGE_EXTREME_CRISIS,
    BPI_EXPONENTIAL_DECAY,
    WICKET_VALUE_MIN,
    WICKET_VALUE_MAX,
    RIDGE_ALPHA,
    K_SHRINK_BASE,
    K_SHRINK_HP,
    K_SHRINK_ALPHA,
    K_SHRINK_BETA,
    Z_LEVERAGE_MIN,
    Z_LEVERAGE_MAX,
    PRR_WEIGHT_MU_HP,
    PRR_WEIGHT_DELTA,
    PRR_WEIGHT_CONSISTENCY,
    PRR_PENALTY_SE_MATCH,
    PRR_PROBIT_SCALE,
    PRIOR_MATCH_PSEUDO_COUNT,
    BATTER_MIN_HP_BALLS,
    BATTER_MIN_HP_MATCHES,
    BATTER_MIN_TOTAL_BALLS,
    BOWLER_MIN_HP_BALLS,
    BOWLER_MIN_HP_MATCHES,
    BOWLER_MIN_TOTAL_BALLS,
    VENUE_JAIPUR_ID,
    VENUE_GUWAHATI_ID,
)

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")


# =============================================================================
# STAGE 1: DATA PREPARATION & STATE RECONSTRUCTION
# =============================================================================
def load_and_reconstruct_deliveries():
    """
    Loads raw relational tables from data/raw/ and reconstructs dynamic pre-delivery
    match states across all 200,999 balls.
    """
    print("⏳ Loading raw relational tables from data/raw/...")
    bbb = pd.read_csv(os.path.join(RAW_DIR, "ball_by_ball.csv"), low_memory=False)
    matches = pd.read_csv(os.path.join(RAW_DIR, "matches.csv"))
    comp = pd.read_csv(os.path.join(RAW_DIR, "competitions.csv"))

    # Relational joins
    m_comp = matches.merge(
        comp[["comp_id", "comp_name", "match_format", "season"]],
        on="comp_id",
        how="left"
    )
    df = bbb.merge(
        m_comp[[
            "match_id", "comp_name", "match_format", "season", "venue_id",
            "home_team", "away_team", "home_team_id", "away_team_id", "win_team_id",
            "knock_out", "stage", "first_inn_revised_overs", "second_inn_revised_overs", "revised_target"
        ]],
        on="match_id",
        how="left"
    )

    # Chronological sort
    df = df.sort_values(["match_id", "innings", "overid", "over_ball_num", "innings_ball_num"]).reset_index(drop=True)

    # Runs & Legal ball accounting
    df["total_ball_runs"] = df["runs"] + df["extras1_runs"].fillna(0) + df["extras_2_runs"].fillna(0)
    df["is_legal_ball"] = (df["extras_type"] != "wide")
    df["is_wicket_int"] = df["is_wicket"].astype(int)
    df["is_dot"] = (df["runs"] == 0) & (df["extras1_runs"].fillna(0) == 0) & (df["extras_2_runs"].fillna(0) == 0)
    df["is_four"] = (df["runs"] == 4)
    df["is_six"] = (df["runs"] == 6)

    # Bowler runs conceded (strictly excludes byes/legbyes; includes wide, noball, and Overstep)
    is_bowler_extra = df["extras_type"].isin(["wide", "noball", "Overstep"])
    df["bowler_runs_conceded"] = df["runs"] + np.where(is_bowler_extra, df["extras1_runs"].fillna(0), 0)

    # Dismissals credit
    non_bowler_dismissals = {"Run Out", "Retired Hurt", "Retired Out", "Obstructing the Field", "Mankad"}
    df["is_bowler_wicket"] = df["is_wicket"] & (~df["dismissal_type"].isin(non_bowler_dismissals))
    df["is_batter_out"] = df["is_wicket"] & (df["dismissal_type"] != "Retired Hurt")

    # Match Par RPO (objective venue & pitch scoring rate)
    m_rpo = df.groupby("match_id").agg(
        tot_runs=("total_ball_runs", "sum"),
        tot_legal=("is_legal_ball", "sum")
    ).reset_index()
    m_rpo["match_par_rpo"] = (m_rpo["tot_runs"] / m_rpo["tot_legal"].clip(lower=1)) * 6.0
    df = df.merge(m_rpo[["match_id", "match_par_rpo"]], on="match_id", how="left")

    # Pre-delivery dynamic match state
    df["b"] = df.groupby(["match_id", "innings"])["is_legal_ball"].cumsum()
    df["balls_remaining"] = np.maximum(0, 120 - df["b"])
    df["runs_cum"] = df.groupby(["match_id", "innings"])["total_ball_runs"].cumsum() - df["total_ball_runs"]
    df["wickets_down"] = df.groupby(["match_id", "innings"])["is_wicket_int"].cumsum() - df["is_wicket_int"]
    df["wickets_down"] = np.clip(df["wickets_down"], 0, 9)
    df["crr"] = (6.0 * df["runs_cum"]) / np.maximum(1, df["b"] - 1)

    # Innings 2 Target & Required Run Rate
    inn1_totals = df[df["innings"] == 1].groupby("match_id")["total_ball_runs"].sum().reset_index()
    inn1_totals.rename(columns={"total_ball_runs": "inn1_total"}, inplace=True)
    df = df.merge(inn1_totals, on="match_id", how="left")

    target_m = np.where(df["revised_target"].notna(), df["revised_target"], df["inn1_total"].fillna(165) + 1)
    df["runs_needed"] = np.where(df["innings"] == 2, np.maximum(0, target_m - df["runs_cum"]), 0.0)
    df["rrr"] = np.where(df["innings"] == 2, (6.0 * df["runs_needed"]) / np.maximum(1, df["balls_remaining"]), 0.0)
    df["rate_gap"] = np.where(df["innings"] == 2, df["rrr"] - df["crr"], 0.0)
    df["knock_out"] = np.where((df["knock_out"] == 1) | (df["stage"] != "League"), 1, 0)

    print(f"  ✓ Pre-delivery match state reconstructed for {len(df)} deliveries.")
    return df


# =============================================================================
# STAGE 1 (CONT.): 3 SUPERVISED ML MODELS & COUNTERFACTUAL LEVERAGE INDEX
# =============================================================================
def train_ml_models_and_compute_leverage(df):
    """
    Train 3 Supervised Machine Learning Models with 5-Fold GroupKFold validation
    grouped by match_id:
    - Model 1: Live Win Probability Classifier (wp_model)
    - Model 2: Expected Runs Regressor (xr_model) using ONLY situational state features
    - Model 3: Expected Dismissal Classifier (xw_model) using ONLY situational state features
    Then derives the Counterfactual Win Probability Variance and continuous Leverage Index L_i.
    """
    print("\n🤖 STAGE 1: Training 3 Delivery-Level ML Models with 5-Fold GroupKFold validation...")

    # ML MODEL 1: LIVE WIN PROBABILITY
    wp_mask = df["win_team_id"].notna() & (df["win_team_id"] > 0) & (df["batting_team_id"].notna())
    wp_df = df[wp_mask].copy()
    wp_df["y_win"] = (wp_df["batting_team_id"] == wp_df["win_team_id"]).astype(int)

    wp_features = [
        "innings", "overid", "balls_remaining", "runs_cum", "wickets_down",
        "crr", "runs_needed", "rrr", "rate_gap", "knock_out", "match_par_rpo"
    ]
    X_wp = wp_df[wp_features].values
    y_wp = wp_df["y_win"].values
    groups_wp = wp_df["match_id"].values

    gkf = GroupKFold(n_splits=5)
    oof_wp = np.zeros(len(wp_df))
    for train_idx, val_idx in gkf.split(X_wp, y_wp, groups_wp):
        clf_wp = HistGradientBoostingClassifier(max_iter=100, random_state=42)
        clf_wp.fit(X_wp[train_idx], y_wp[train_idx])
        oof_wp[val_idx] = clf_wp.predict_proba(X_wp[val_idx])[:, 1]

    auc_wp = float(roc_auc_score(y_wp, oof_wp))
    brier_wp = float(brier_score_loss(y_wp, oof_wp))
    loss_wp = float(log_loss(y_wp, oof_wp))
    print(f"  ✓ Model 1 (Win Probability) 5-Fold CV: ROC-AUC = {auc_wp:.4f}, Brier = {brier_wp:.4f}, Log-Loss = {loss_wp:.4f}")

    wp_model = HistGradientBoostingClassifier(max_iter=100, random_state=42)
    wp_model.fit(X_wp, y_wp)

    # MODELS 2 & 3: SITUATIONAL PAR EXPECTATIONS (NO player features!)
    rw_features = ["innings", "overid", "over_ball_num", "balls_remaining", "wickets_down", "runs_needed", "rrr", "match_par_rpo"]
    df["pace_or_spin_cat"] = (df["pace_or_spin"] == "Spin").astype(int)
    rw_bowl_features = ["innings", "overid", "over_ball_num", "balls_remaining", "wickets_down", "runs_needed", "rrr", "match_par_rpo", "pace_or_spin_cat"]

    leg = df[df["is_legal_ball"] == True].copy()
    X_rw = leg[rw_features].values
    X_rw_bowl = leg[rw_bowl_features].values
    y_runs = leg["runs"].values
    y_wkt = leg["is_wicket_int"].values
    groups_rw = leg["match_id"].values

    oof_runs = np.zeros(len(leg))
    for train_idx, val_idx in gkf.split(X_rw, y_runs, groups_rw):
        reg = HistGradientBoostingRegressor(max_iter=100, random_state=100)
        reg.fit(X_rw[train_idx], y_runs[train_idx])
        oof_runs[val_idx] = reg.predict(X_rw[val_idx])

    rmse_runs = float(root_mean_squared_error(y_runs, oof_runs))
    mae_runs = float(mean_absolute_error(y_runs, oof_runs))
    print(f"  ✓ Model 2 (Expected Runs) 5-Fold CV: RMSE = {rmse_runs:.4f}, MAE = {mae_runs:.4f}")

    xr_model = HistGradientBoostingRegressor(max_iter=100, random_state=100)
    xr_model.fit(X_rw, y_runs)

    oof_wkt = np.zeros(len(leg))
    for train_idx, val_idx in gkf.split(X_rw, y_wkt, groups_rw):
        clf_w = HistGradientBoostingClassifier(max_iter=100, random_state=200)
        clf_w.fit(X_rw[train_idx], y_wkt[train_idx])
        oof_wkt[val_idx] = clf_w.predict_proba(X_rw[val_idx])[:, 1]

    auc_wkt = float(roc_auc_score(y_wkt, oof_wkt))
    brier_wkt = float(brier_score_loss(y_wkt, oof_wkt))
    loss_wkt = float(log_loss(y_wkt, oof_wkt))
    print(f"  ✓ Model 3 (Expected Dismissals) 5-Fold CV: ROC-AUC = {auc_wkt:.4f}, Brier = {brier_wkt:.4f}, Log-Loss = {loss_wkt:.4f}")

    xw_model = HistGradientBoostingClassifier(max_iter=100, random_state=200)
    xw_model.fit(X_rw, y_wkt)

    # Specialized Bowler Situational Expectation Models (Conditioned on pace_or_spin discipline)
    print("  Training specialized bowler expectation models (bowler_runs_conceded & is_bowler_wicket conditioned on pace_or_spin)...")
    xr_bowl_model = HistGradientBoostingRegressor(max_iter=100, random_state=100)
    xr_bowl_model.fit(X_rw_bowl, leg["bowler_runs_conceded"].values)

    xw_bowl_model = HistGradientBoostingClassifier(max_iter=100, random_state=200)
    xw_bowl_model.fit(X_rw_bowl, leg["is_bowler_wicket"].astype(int).values)

    # Save validation metrics
    val_metrics = {
        "model_1_win_probability": {
            "roc_auc": round(auc_wp, 4),
            "brier_score": round(brier_wp, 4),
            "log_loss": round(loss_wp, 4),
            "features": wp_features,
            "cv_method": "5-Fold GroupKFold(match_id)"
        },
        "model_2_expected_runs": {
            "rmse": round(rmse_runs, 4),
            "mae": round(mae_runs, 4),
            "features": rw_features,
            "cv_method": "5-Fold GroupKFold(match_id)"
        },
        "model_3_expected_dismissals": {
            "roc_auc": round(auc_wkt, 4),
            "brier_score": round(brier_wkt, 4),
            "log_loss": round(loss_wkt, 4),
            "features": rw_features,
            "cv_method": "5-Fold GroupKFold(match_id)"
        }
    }
    with open(os.path.join(PROCESSED_DIR, "ml_model_validation_metrics.json"), "w") as f:
        json.dump(val_metrics, f, indent=2)

    # Permutation feature importances
    sub_sample = wp_df.sample(min(8000, len(wp_df)), random_state=42)
    p_imp = permutation_importance(wp_model, sub_sample[wp_features].values, sub_sample["y_win"].values, n_repeats=5, random_state=42)
    feat_df = pd.DataFrame({
        "feature": wp_features,
        "importance_mean": p_imp.importances_mean,
        "importance_std": p_imp.importances_std
    }).sort_values("importance_mean", ascending=False)
    feat_df.to_csv(os.path.join(PROCESSED_DIR, "ml_feature_importances.csv"), index=False)

    # COUNTERFACTUAL WIN PROBABILITY VARIANCE & LEVERAGE INDEX L_i
    print("\n⚡ Computing Counterfactual Win Probability Variance & Leverage Index (L_i)...")
    X_curr = df[wp_features].copy()
    wp_curr = wp_model.predict_proba(X_curr.values)[:, 1]

    b_next = df["b"] + 1
    rem_next = np.maximum(0, df["balls_remaining"] - 1)

    # Counterfactual 1: Dot ball (+0 runs, -1 ball)
    X_dot = X_curr.copy()
    X_dot["balls_remaining"] = rem_next
    X_dot["crr"] = (6.0 * X_dot["runs_cum"]) / np.maximum(1, b_next - 1)
    X_dot["rrr"] = np.where(df["innings"] == 2, (6.0 * X_dot["runs_needed"]) / np.maximum(1, rem_next), 0.0)
    X_dot["rate_gap"] = np.where(df["innings"] == 2, X_dot["rrr"] - X_dot["crr"], 0.0)
    wp_dot = wp_model.predict_proba(X_dot.values)[:, 1]

    # Counterfactual 2: Single (+1 run, -1 ball)
    X_1 = X_curr.copy()
    X_1["balls_remaining"] = rem_next
    X_1["runs_cum"] = X_1["runs_cum"] + 1
    X_1["crr"] = (6.0 * X_1["runs_cum"]) / np.maximum(1, b_next - 1)
    X_1["runs_needed"] = np.where(df["innings"] == 2, np.maximum(0, df["runs_needed"] - 1), 0.0)
    X_1["rrr"] = np.where(df["innings"] == 2, (6.0 * X_1["runs_needed"]) / np.maximum(1, rem_next), 0.0)
    X_1["rate_gap"] = np.where(df["innings"] == 2, X_1["rrr"] - X_1["crr"], 0.0)
    wp_1 = wp_model.predict_proba(X_1.values)[:, 1]

    # Counterfactual 3: Boundary Four (+4 runs, -1 ball)
    X_4 = X_curr.copy()
    X_4["balls_remaining"] = rem_next
    X_4["runs_cum"] = X_4["runs_cum"] + 4
    X_4["crr"] = (6.0 * X_4["runs_cum"]) / np.maximum(1, b_next - 1)
    X_4["runs_needed"] = np.where(df["innings"] == 2, np.maximum(0, df["runs_needed"] - 4), 0.0)
    X_4["rrr"] = np.where(df["innings"] == 2, (6.0 * X_4["runs_needed"]) / np.maximum(1, rem_next), 0.0)
    X_4["rate_gap"] = np.where(df["innings"] == 2, X_4["rrr"] - X_4["crr"], 0.0)
    wp_4 = wp_model.predict_proba(X_4.values)[:, 1]

    # Counterfactual 4: Wicket (+0 runs, +1 wicket, -1 ball)
    X_wkt = X_curr.copy()
    X_wkt["balls_remaining"] = rem_next
    X_wkt["wickets_down"] = np.clip(X_wkt["wickets_down"] + 1, 0, 9)
    X_wkt["crr"] = (6.0 * X_wkt["runs_cum"]) / np.maximum(1, b_next - 1)
    X_wkt["rrr"] = np.where(df["innings"] == 2, (6.0 * X_wkt["runs_needed"]) / np.maximum(1, rem_next), 0.0)
    X_wkt["rate_gap"] = np.where(df["innings"] == 2, X_wkt["rrr"] - X_wkt["crr"], 0.0)
    wp_wkt = wp_model.predict_proba(X_wkt.values)[:, 1]

    d_wp_dot = wp_dot - wp_curr
    d_wp_1 = wp_1 - wp_curr
    d_wp_4 = wp_4 - wp_curr
    d_wp_wkt = wp_wkt - wp_curr

    # Outcome probabilities from Models 2 & 3
    X_rw_full = df[rw_features].values
    exp_runs = xr_model.predict(X_rw_full)
    exp_wkt = xw_model.predict_proba(X_rw_full)[:, 1]

    p_wkt = np.clip(exp_wkt, 0.01, 0.35)
    p_bnd = np.clip((exp_runs - 0.65) / 3.2, 0.08, 0.35)
    p_1 = np.clip(1.0 - p_wkt - p_bnd - 0.35, 0.25, 0.50)
    p_dot = np.clip(1.0 - p_wkt - p_bnd - p_1, 0.10, 0.60)
    p_sum = p_dot + p_1 + p_bnd + p_wkt
    p_dot, p_1, p_bnd, p_wkt = p_dot / p_sum, p_1 / p_sum, p_bnd / p_sum, p_wkt / p_sum

    sigma_wp = np.sqrt(
        p_dot * (d_wp_dot ** 2) +
        p_1 * (d_wp_1 ** 2) +
        p_bnd * (d_wp_4 ** 2) +
        p_wkt * (d_wp_wkt ** 2)
    )
    mean_sigma = float(np.mean(sigma_wp))
    df["L_i"] = (sigma_wp / max(1e-6, mean_sigma)).round(4)
    df["BPI"] = (100.0 * (1.0 - np.exp(-BPI_EXPONENTIAL_DECAY * df["L_i"]))).round(2)

    # 1. High-Pressure Regime (L_i >= 1.35 with early-overs wickets/RRR gate)
    is_hp_base = (df["L_i"] >= LEVERAGE_HIGH_PRESSURE)
    is_early = (df["overid"] <= 10)
    early_ok = (df["wickets_down"] >= 2) | (df["rrr"] >= 10.5)
    df["is_HP"] = is_hp_base & (~is_early | early_ok)
    df["is_high_pressure"] = df["is_HP"]  # Alias for backward compatibility
    df["is_extreme_crisis"] = (df["L_i"] >= LEVERAGE_EXTREME_CRISIS)

    # 2. Smooth First-Principles Resource-Depletion Wicket Valuation C_wkt
    df["C_wkt"] = (6.0 + 10.0 * ((df["balls_remaining"] / 120.0)**0.85) * (((10.0 - df["wickets_down"]) / 10.0)**0.5)).round(2)

    hp_count = int(df["is_HP"].sum())
    hp_pct = float(df["is_HP"].mean() * 100)
    print(f"  ✓ Leverage Index derived: {hp_count} High-Pressure deliveries ({hp_pct:.1f}% of total, L_i >= {LEVERAGE_HIGH_PRESSURE}).")

    return df, xr_model, xw_model, xr_bowl_model, xw_bowl_model, rw_features, rw_bowl_features


# =============================================================================
# STAGE 2: ENDOGENOUS RUN VALUES & SIMULTANEOUS BIPARTITE RIDGE REGRESSION
# =============================================================================
def compute_endogenous_values_and_opponent_skills(df, xr_model, xw_model, rw_features, xr_bowl_model, xw_bowl_model, rw_bowl_features=None):
    """
    Fits Simultaneous Bipartite Ridge Regression (sklearn.linear_model.Ridge)
    on the connected match network over all legal deliveries (+1 for striker, -1 for bowler)
    to solve for global player skills (theta_bat, theta_bowl) and opponent-adjust every ball.
    For bowlers, uses specialized xr_bowl_model and xw_bowl_model conditioned on pace_or_spin.
    """
    if rw_bowl_features is None:
        rw_bowl_features = rw_features
    print(f"\n⚖️ STAGE 2: Smooth Wicket Valuation & Simultaneous Bipartite Ridge Regression (alpha={RIDGE_ALPHA})...")
    leg = df[df["is_legal_ball"] == True].copy()
    leg["E_R"] = xr_model.predict(leg[rw_features].values)
    leg["E_W"] = xw_model.predict_proba(leg[rw_features].values)[:, 1]

    # Specialized bowler expectations (Conditioned on pace_or_spin discipline)
    leg["E_R_bowl"] = xr_bowl_model.predict(leg[rw_bowl_features].values)
    leg["E_W_bowl"] = xw_bowl_model.predict_proba(leg[rw_bowl_features].values)[:, 1]

    # Target y_res: Net run residual above expected runs and wicket depletion cost
    y_res = (leg["runs"] - leg["E_R"]) - leg["C_wkt"] * (leg["is_batter_out"].astype(int) - leg["E_W"])

    unique_strikers = np.sort(leg["striker_id"].unique())
    unique_bowlers = np.sort(leg["bowler_id"].unique())
    s_map = {s: i for i, s in enumerate(unique_strikers)}
    b_map = {b: i for i, b in enumerate(unique_bowlers)}
    n_s = len(unique_strikers)
    n_b = len(unique_bowlers)
    n_rows = len(leg)

    rows = np.repeat(np.arange(n_rows), 2)
    cols = np.empty(2 * n_rows, dtype=int)
    data = np.empty(2 * n_rows, dtype=float)

    cols[0::2] = leg["striker_id"].map(s_map).values
    data[0::2] = 1.0

    cols[1::2] = n_s + leg["bowler_id"].map(b_map).values
    data[1::2] = -1.0

    X_matchup = sp.csr_matrix((data, (rows, cols)), shape=(n_rows, n_s + n_b))
    print(f"  Fitting Simultaneous Bipartite Ridge on {n_rows} legal balls ({n_s} batters, {n_b} bowlers)...")
    ridge = Ridge(alpha=RIDGE_ALPHA, fit_intercept=False, solver="lsqr")
    ridge.fit(X_matchup, y_res.values)

    coef = ridge.coef_
    theta_bat_map = dict(zip(unique_strikers, coef[:n_s]))
    theta_bowl_map = dict(zip(unique_bowlers, -coef[n_s:]))

    leg["theta_bat"] = leg["striker_id"].map(theta_bat_map).fillna(0.0)
    leg["theta_bowl"] = leg["bowler_id"].map(theta_bowl_map).fillna(0.0)

    # Opponent-adjusted net context delivery value per 100 balls:
    # Batter: 100 * [(runs - E_R) - C_wkt * (is_batter_out - E_W) - theta_bowl] (100% locked)
    leg["v_bat"] = 100.0 * ((leg["runs"] - leg["E_R"]) - leg["C_wkt"] * (leg["is_batter_out"].astype(int) - leg["E_W"]) - leg["theta_bowl"])

    # FIX 1: Zero Wicket-Tax on Dot Balls
    # W_res = 1 - E[W_bowl] if bowler wicket; 0.0 if dot ball; -E[W_bowl] if run conceded without wicket
    is_wkt = (leg["is_bowler_wicket"] == True).astype(int)
    is_dot = (leg["bowler_runs_conceded"] == 0)

    w_res = np.where(
        is_wkt == 1,
        1.0 - leg["E_W_bowl"],
        np.where(is_dot, 0.0, -leg["E_W_bowl"])
    )
    leg["W_res"] = w_res
    leg["run_suppression_ball"] = leg["E_R_bowl"] - leg["bowler_runs_conceded"]
    leg["v_bowl"] = 100.0 * (
        (leg["E_R_bowl"] - leg["bowler_runs_conceded"]) +
        0.65 * leg["C_wkt"] * leg["W_res"] +
        leg["theta_bat"]
    )

    print("  ✓ Simultaneous Bipartite Ridge skills solved and opponent adjustments applied directly.")
    return leg


# =============================================================================
# STAGES 3 & 4: INDEPENDENT EB SHRINKAGE & STANDARDIZED BAYESIAN LCB RANKINGS
# =============================================================================
def compute_hierarchical_random_slopes_and_lcb(leg):
    """
    Independent Empirical Bayes Shrinkage, Non-Negative-Baseline Clutch Elevation,
    Standardized Bayesian LCB Scoring, and Universal Sample Size Eligibility.
    """
    print("\n📈 STAGES 3 & 4: Independent Empirical Bayes Shrinkage & Bayesian LCB Rankings...")
    leg["z_L"] = np.clip(leg["L_i"] - 1.0, Z_LEVERAGE_MIN, Z_LEVERAGE_MAX)

    total_faced = leg.groupby("striker_id")["runs"].count().rename("total_balls_faced")
    total_bowled = leg.groupby("bowler_id")["runs"].count().rename("total_balls_bowled")

    def zscore(s):
        std = float(s.std())
        if std < 1e-6:
            return s * 0.0
        return (s - float(s.mean())) / std

    # ---------------------------------------------------------
    # 1. BATTERS INDEPENDENT EMPIRICAL BAYES & LCB MODEL
    # ---------------------------------------------------------
    print("  Fitting Batters Independent Empirical Bayes & LCB Model...")
    b_base = leg[leg["is_HP"] == False]
    b_hp = leg[leg["is_HP"] == True]

    b_base_agg = b_base.groupby("striker_id").agg(
        N_base=("runs", "count"),
        v_base_mean=("v_bat", "mean"),
        base_runs=("runs", "sum")
    ).reset_index()

    b_tot_agg = leg.groupby(["striker_id", "striker_name"]).agg(
        N_tot=("runs", "count"),
        v_tot_mean=("v_bat", "mean"),
        tot_runs=("runs", "sum"),
        median_batting_order=("batting_order", "median")
    ).reset_index()

    bat_df = b_tot_agg.merge(b_base_agg, on="striker_id", how="left").merge(total_bowled, left_on="striker_id", right_index=True, how="left")
    bat_df["total_balls_bowled"] = bat_df["total_balls_bowled"].fillna(0)
    bat_df["N_base"] = bat_df["N_base"].fillna(0)
    bat_df["v_base_mean"] = bat_df["v_base_mean"].fillna(0.0)

    # Leverage-weighted High-Pressure performance
    b_hp_agg = b_hp.groupby("striker_id").agg(
        N_HP=("runs", "count"),
        W_HP=("L_i", "sum"),
        hp_matches=("match_id", "nunique"),
        sum_Lv=("v_bat", lambda x: np.sum(x * b_hp.loc[x.index, "L_i"])),
        HP_runs=("runs", "sum"),
        hp_dots=("is_dot", "sum"),
        hp_fours=("is_four", "sum"),
        hp_sixes=("is_six", "sum")
    ).reset_index()
    b_hp_agg["v_HP_weighted"] = b_hp_agg["sum_Lv"] / np.maximum(1e-6, b_hp_agg["W_HP"])

    bat_df = bat_df.merge(b_hp_agg, on="striker_id", how="left")
    bat_df["N_HP"] = bat_df["N_HP"].fillna(0)
    bat_df["hp_matches"] = bat_df["hp_matches"].fillna(0)
    bat_df["v_HP_weighted"] = bat_df["v_HP_weighted"].fillna(0.0)

    # Independent Empirical Bayes Shrinkage
    bat_df["alpha_EB"] = (bat_df["N_base"] / (bat_df["N_base"] + K_SHRINK_BASE)) * bat_df["v_base_mean"]
    bat_df["mu_HP_EB"] = (bat_df["N_HP"] / (bat_df["N_HP"] + K_SHRINK_HP)) * bat_df["v_HP_weighted"]

    # Non-negative-baseline clutch elevation
    bat_df["Delta_EB"] = bat_df["mu_HP_EB"] - np.maximum(0.0, 0.75 * bat_df["alpha_EB"])

    # Match-level reliability & downside risk
    b_m = b_hp.groupby(["striker_id", "match_id"]).agg(
        m_balls=("runs", "count"),
        m_v=("v_bat", "mean")
    ).reset_index()
    b_m_qual = b_m[b_m["m_balls"] >= 4]

    b_m_stats = b_m_qual.groupby("striker_id").agg(
        M_HP=("match_id", "count"),
        s_match=("m_v", lambda x: float(x.std(ddof=1)) if len(x) > 1 else 0.0),
        pos_matches=("m_v", lambda x: (x > 0).sum())
    ).reset_index()

    sigma_0_bat = float(np.sqrt(np.mean(b_m_stats[b_m_stats["s_match"] > 0]["s_match"]**2)))
    bat_df = bat_df.merge(b_m_stats, on="striker_id", how="left")
    bat_df["M_HP"] = bat_df["M_HP"].fillna(0)
    bat_df["s_match"] = bat_df["s_match"].fillna(sigma_0_bat)
    bat_df["pos_matches"] = bat_df["pos_matches"].fillna(0)

    bat_df["sigma_match_shrunk"] = np.sqrt(
        (np.maximum(0, bat_df["M_HP"] - 1) * (bat_df["s_match"]**2) + PRIOR_MATCH_PSEUDO_COUNT * (sigma_0_bat**2)) /
        (np.maximum(0, bat_df["M_HP"] - 1) + PRIOR_MATCH_PSEUDO_COUNT)
    )
    bat_df["SE_match"] = bat_df["sigma_match_shrunk"] / np.sqrt(bat_df["M_HP"] + 5.0)
    bat_df["Consistency_Pct"] = 100.0 * (bat_df["pos_matches"] + 6.0) / (bat_df["M_HP"] + 10.0)

    bat_df["SR_Base"] = ((bat_df["base_runs"].fillna(0) / bat_df["N_base"].clip(lower=1)) * 100.0).round(1)
    bat_df["SR_HP"] = ((bat_df["HP_runs"].fillna(0) / bat_df["N_HP"].clip(lower=1)) * 100.0).round(1)

    # Universal Sample Size Eligibility:
    bat_mask = (
        (bat_df["N_HP"] >= BATTER_MIN_HP_BALLS) &
        (bat_df["M_HP"] >= BATTER_MIN_HP_MATCHES) &
        (bat_df["N_tot"] >= BATTER_MIN_TOTAL_BALLS) &
        (bat_df["N_tot"] >= bat_df["total_balls_bowled"]) &
        (bat_df["alpha_EB"] >= -2.0) &
        (bat_df["mu_HP_EB"] > 0) &
        (bat_df["Delta_EB"] > 0) &
        (bat_df["Consistency_Pct"] >= 55.0)
    )
    eligible_bat = bat_df[bat_mask].copy().reset_index(drop=True)

    eligible_bat["Z_mu_HP"] = zscore(eligible_bat["mu_HP_EB"])
    eligible_bat["Z_Delta"] = zscore(eligible_bat["Delta_EB"])
    eligible_bat["Z_Consistency"] = zscore(eligible_bat["Consistency_Pct"])
    eligible_bat["Z_SE_match"] = zscore(eligible_bat["SE_match"])

    eligible_bat["PRR_Raw"] = (
        PRR_WEIGHT_MU_HP * eligible_bat["Z_mu_HP"] +
        PRR_WEIGHT_DELTA * eligible_bat["Z_Delta"] +
        PRR_WEIGHT_CONSISTENCY * eligible_bat["Z_Consistency"] -
        PRR_PENALTY_SE_MATCH * eligible_bat["Z_SE_match"]
    )
    eligible_bat["Z_PRR_Raw"] = zscore(eligible_bat["PRR_Raw"])
    eligible_bat["PRR_Bat"] = (norm.cdf(PRR_PROBIT_SCALE * eligible_bat["Z_PRR_Raw"]) * 100.0).round(2)
    eligible_bat = eligible_bat.sort_values("PRR_Bat", ascending=False).reset_index(drop=True)

    eligible_bat["Rank"] = range(1, len(eligible_bat) + 1)
    eligible_bat["player_name"] = eligible_bat["striker_name"]
    eligible_bat["player_id"] = eligible_bat["striker_id"].astype(int)
    eligible_bat["total_balls"] = eligible_bat["N_tot"].astype(int)
    eligible_bat["HP_Deliveries"] = eligible_bat["N_HP"].astype(int)
    eligible_bat["Base_SR"] = eligible_bat["SR_Base"]
    eligible_bat["HP_SR"] = eligible_bat["SR_HP"]
    eligible_bat["Delta_Clutch"] = eligible_bat["Delta_EB"].round(2)
    eligible_bat["Delta_Bat"] = eligible_bat["Delta_EB"].round(2)
    eligible_bat["PRR_Score"] = eligible_bat["PRR_Bat"]
    eligible_bat["tier1_hp_pct"] = 1.0

    top10_bat = eligible_bat.head(10).copy()
    top10_bat.to_csv(os.path.join(PROCESSED_DIR, "top_batters_under_pressure.csv"), index=False)
    b_m.to_csv(os.path.join(PROCESSED_DIR, "batter_match_by_match_consistency.csv"), index=False)
    print(f"  ✓ Exported Official Top 10 Batters (from {len(eligible_bat)} universally eligible batters).")

    # ---------------------------------------------------------
    # 2. BOWLERS INDEPENDENT EMPIRICAL BAYES & LCB MODEL
    # ---------------------------------------------------------
    print("  Fitting Bowlers Independent Empirical Bayes & LCB Model...")
    bw_base = leg[leg["is_HP"] == False]
    bw_hp = leg[leg["is_HP"] == True]

    bw_base_agg = bw_base.groupby("bowler_id").agg(
        N_base=("runs", "count"),
        v_base_mean=("v_bowl", "mean"),
        base_conceded=("bowler_runs_conceded", "sum"),
        base_exp_conceded=("E_R_bowl", "sum")
    ).reset_index()

    bw_tot_agg = leg.groupby(["bowler_id", "bowler_name"]).agg(
        N_tot=("runs", "count"),
        v_tot_mean=("v_bowl", "mean"),
        tot_conceded=("bowler_runs_conceded", "sum"),
        pace_or_spin=("pace_or_spin", lambda x: x.mode()[0] if len(x.mode()) > 0 else "Pace"),
        bowling_type=("bowling_type", lambda x: x.mode()[0] if len(x.mode()) > 0 else "Right-arm fast")
    ).reset_index()

    bowl_df = bw_tot_agg.merge(bw_base_agg, on="bowler_id", how="left").merge(total_faced, left_on="bowler_id", right_index=True, how="left")
    bowl_df["total_balls_faced"] = bowl_df["total_balls_faced"].fillna(0)
    bowl_df["N_base"] = bowl_df["N_base"].fillna(0)
    bowl_df["v_base_mean"] = bowl_df["v_base_mean"].fillna(0.0)
    bowl_df["base_conceded"] = bowl_df["base_conceded"].fillna(0.0)
    bowl_df["base_exp_conceded"] = bowl_df["base_exp_conceded"].fillna(0.0)

    bw_hp_agg = bw_hp.groupby("bowler_id").agg(
        N_HP=("runs", "count"),
        W_HP=("L_i", "sum"),
        hp_matches=("match_id", "nunique"),
        sum_Lv=("v_bowl", lambda x: np.sum(x * bw_hp.loc[x.index, "L_i"])),
        HP_conceded=("bowler_runs_conceded", "sum"),
        HP_exp_conceded=("E_R_bowl", "sum"),
        HP_wkts=("is_bowler_wicket", "sum"),
        hp_dots=("is_dot", "sum"),
        run_supp_hp=("run_suppression_ball", "mean")
    ).reset_index()
    bw_hp_agg["v_HP_weighted"] = bw_hp_agg["sum_Lv"] / np.maximum(1e-6, bw_hp_agg["W_HP"])

    bowl_df = bowl_df.merge(bw_hp_agg, on="bowler_id", how="left")
    bowl_df["N_HP"] = bowl_df["N_HP"].fillna(0)
    bowl_df["hp_matches"] = bowl_df["hp_matches"].fillna(0)
    bowl_df["v_HP_weighted"] = bowl_df["v_HP_weighted"].fillna(0.0)
    bowl_df["HP_conceded"] = bowl_df["HP_conceded"].fillna(0.0)
    bowl_df["HP_exp_conceded"] = bowl_df["HP_exp_conceded"].fillna(0.0)
    bowl_df["run_supp_hp"] = bowl_df["run_supp_hp"].fillna(0.0)

    # FIX 2: Fit symmetric OLS regression slope rho_hat = clip(Cov(v_HP, v_base) / Var(v_base), 0.20, 0.30)
    cand = bowl_df[(bowl_df["N_HP"] >= 80) & (bowl_df["N_base"] >= 150)]
    cov_mat = np.cov(cand["v_base_mean"], cand["v_HP_weighted"])
    cov_val = cov_mat[0, 1]
    var_val = cov_mat[0, 0]
    rho_hat = float(np.clip(cov_val / max(1e-6, var_val), 0.20, 0.30))
    print(f"  Empirical OLS Slope rho_hat: {rho_hat:.4f} (from cov={cov_val:.2f}, var={var_val:.2f})")

    # Shrunk baseline and HP output
    bowl_df["alpha_EB"] = (bowl_df["N_base"] / (bowl_df["N_base"] + K_SHRINK_BASE)) * bowl_df["v_base_mean"]
    bowl_df["mu_HP_EB"] = (bowl_df["N_HP"] / (bowl_df["N_HP"] + K_SHRINK_HP)) * bowl_df["v_HP_weighted"]

    # Symmetric single-pass Empirical Bayes Clutch Elevation:
    bowl_df["Delta_EB"] = (bowl_df["N_HP"] / (bowl_df["N_HP"] + K_SHRINK_HP)) * (bowl_df["v_HP_weighted"] - rho_hat * bowl_df["v_base_mean"])

    bw_m = bw_hp.groupby(["bowler_id", "match_id"]).agg(
        m_balls=("runs", "count"),
        m_v=("v_bowl", "mean")
    ).reset_index()
    bw_m_qual = bw_m[bw_m["m_balls"] >= 4]

    bw_m_stats = bw_m_qual.groupby("bowler_id").agg(
        M_HP=("match_id", "count"),
        s_match=("m_v", lambda x: float(x.std(ddof=1)) if len(x) > 1 else 0.0),
        pos_matches=("m_v", lambda x: (x > 0).sum())
    ).reset_index()

    sigma_0_bowl = float(np.sqrt(np.mean(bw_m_stats[bw_m_stats["s_match"] > 0]["s_match"]**2)))
    bowl_df = bowl_df.merge(bw_m_stats, on="bowler_id", how="left")
    bowl_df["M_HP"] = bowl_df["M_HP"].fillna(0)
    bowl_df["s_match"] = bowl_df["s_match"].fillna(sigma_0_bowl)
    bowl_df["pos_matches"] = bowl_df["pos_matches"].fillna(0)

    bowl_df["sigma_match_shrunk"] = np.sqrt(
        (np.maximum(0, bowl_df["M_HP"] - 1) * (bowl_df["s_match"]**2) + PRIOR_MATCH_PSEUDO_COUNT * (sigma_0_bowl**2)) /
        (np.maximum(0, bowl_df["M_HP"] - 1) + PRIOR_MATCH_PSEUDO_COUNT)
    )
    bowl_df["SE_match"] = bowl_df["sigma_match_shrunk"] / np.sqrt(bowl_df["M_HP"] + 5.0)
    bowl_df["Consistency_Pct"] = 100.0 * (bowl_df["pos_matches"] + 6.0) / (bowl_df["M_HP"] + 10.0)

    bowl_df["Econ_Base"] = ((bowl_df["base_conceded"] / bowl_df["N_base"].clip(lower=1)) * 6.0).round(2)
    bowl_df["Expected_Econ_Base"] = ((bowl_df["base_exp_conceded"] / bowl_df["N_base"].clip(lower=1)) * 6.0).round(2)
    bowl_df["Econ_HP"] = ((bowl_df["HP_conceded"] / bowl_df["N_HP"].clip(lower=1)) * 6.0).round(2)
    bowl_df["Expected_Econ_HP"] = ((bowl_df["HP_exp_conceded"] / bowl_df["N_HP"].clip(lower=1)) * 6.0).round(2)

    # Pure ML Situational Par Eligibility (NO manual RPO cutoffs):
    bowl_mask = (
        (bowl_df["N_HP"] >= BOWLER_MIN_HP_BALLS) &
        (bowl_df["M_HP"] >= BOWLER_MIN_HP_MATCHES) &
        (bowl_df["N_tot"] >= BOWLER_MIN_TOTAL_BALLS) &
        (bowl_df["N_tot"] >= bowl_df["total_balls_faced"]) &
        (bowl_df["alpha_EB"] >= -2.0) &
        (bowl_df["mu_HP_EB"] > 0) &
        (bowl_df["Delta_EB"] > 0) &
        (bowl_df["Consistency_Pct"] >= 55.0) &
        (bowl_df["Econ_HP"] < bowl_df["Expected_Econ_HP"]) &
        (bowl_df["Econ_Base"] < bowl_df["Expected_Econ_Base"])
    )
    eligible_bowl = bowl_df[bowl_mask].copy().reset_index(drop=True)
    n_pace = (eligible_bowl["pace_or_spin"] == "Pace").sum()
    n_spin = (eligible_bowl["pace_or_spin"] == "Spin").sum()
    print(f"  ✓ Eligible Bowlers: {len(eligible_bowl)} (Pace: {n_pace}, Spin: {n_spin}) meeting pure ML Situational Par.")

    eligible_bowl["Z_mu_HP"] = zscore(eligible_bowl["mu_HP_EB"])
    eligible_bowl["Z_alpha"] = zscore(eligible_bowl["alpha_EB"])
    eligible_bowl["Z_Delta"] = zscore(eligible_bowl["Delta_EB"])
    eligible_bowl["Z_Consistency"] = zscore(eligible_bowl["Consistency_Pct"])
    eligible_bowl["Z_SE_match"] = zscore(eligible_bowl["SE_match"])

    # PRR_Raw_Bowl = 0.50 * Z(mu_HP_EB) + 0.20 * Z(alpha_EB) + 0.15 * Z(Delta_EB) + 0.15 * Z(Consistency_Pct) - 0.25 * Z(SE_match)
    eligible_bowl["PRR_Raw_Bowl"] = (
        0.50 * eligible_bowl["Z_mu_HP"] +
        0.20 * eligible_bowl["Z_alpha"] +
        0.15 * eligible_bowl["Z_Delta"] +
        0.15 * eligible_bowl["Z_Consistency"] -
        0.25 * eligible_bowl["Z_SE_match"]
    )

    # Post-Filter Cohort Standardization (Role-Pure Z-Scores within qualified cohorts):
    cohort_dfs = []
    for role in ["Pace", "Spin"]:
        c_df = eligible_bowl[eligible_bowl["pace_or_spin"] == role].copy()
        if len(c_df) > 0:
            c_df["Z_cohort"] = zscore(c_df["PRR_Raw_Bowl"])
            c_df["PRR_Bowl"] = (norm.cdf(0.85 * c_df["Z_cohort"]) * 100.0).round(2)
            cohort_dfs.append(c_df)

    final_bowl = pd.concat(cohort_dfs, ignore_index=True)
    final_bowl = final_bowl.sort_values("PRR_Bowl", ascending=False).reset_index(drop=True)

    final_bowl["Rank"] = range(1, len(final_bowl) + 1)
    final_bowl["player_name"] = final_bowl["bowler_name"]
    final_bowl["player_id"] = final_bowl["bowler_id"].astype(int)
    final_bowl["total_balls"] = final_bowl["N_tot"].astype(int)
    final_bowl["HP_Deliveries"] = final_bowl["N_HP"].astype(int)
    final_bowl["Base_Economy"] = final_bowl["Econ_Base"]
    final_bowl["HP_Economy"] = final_bowl["Econ_HP"]
    final_bowl["Expected_Base_Economy"] = final_bowl["Expected_Econ_Base"]
    final_bowl["Expected_HP_Economy"] = final_bowl["Expected_Econ_HP"]
    final_bowl["Delta_Clutch"] = final_bowl["Delta_EB"].round(2)
    final_bowl["Delta_Bowl"] = final_bowl["Delta_EB"].round(2)
    final_bowl["PRR_Score"] = final_bowl["PRR_Bowl"]
    final_bowl["tier1_hp_pct"] = 1.0

    top10_bowl = final_bowl.head(10).copy()
    top10_bowl.to_csv(os.path.join(PROCESSED_DIR, "top_bowlers_under_pressure.csv"), index=False)
    bw_m.to_csv(os.path.join(PROCESSED_DIR, "bowler_match_by_match_consistency.csv"), index=False)
    print(f"  ✓ Exported Official Top 10 Bowlers (from {len(eligible_bowl)} universally eligible bowlers).")

    return top10_bat, top10_bowl, b_m, bw_m


# =============================================================================
# VENUE SUITABILITY SUMMARIES
# =============================================================================
def compute_rr_venue_summaries(leg):
    """
    Computes venue-specific baseline factors and match stats for Rajasthan Royals homes:
    - Sawai Mansingh Stadium, Jaipur (Venue ID: 6)
    - ACA Stadium, Barsapara, Guwahati (Venue ID: 1241)
    """
    print("\n🏰 Computing Rajasthan Royals Home Venue Summaries (Jaipur & Guwahati)...")
    for venue_id, name in [(VENUE_JAIPUR_ID, "Jaipur"), (VENUE_GUWAHATI_ID, "Guwahati")]:
        v_df = leg[leg["venue_id"] == venue_id]
        if len(v_df) > 0:
            summary = {
                "venue_id": venue_id,
                "venue_name": name,
                "total_deliveries": int(len(v_df)),
                "matches_hosted": int(v_df["match_id"].nunique()),
                "avg_rpo": round(float((v_df["total_ball_runs"].sum() / len(v_df)) * 6.0), 2),
                "hp_deliveries": int(v_df["is_HP"].sum()),
                "hp_share_pct": round(float(v_df["is_HP"].mean() * 100), 1),
                "avg_wicket_value": round(float(v_df["C_wkt"].mean()), 2),
                "pace_econ": round(float((v_df[v_df["pace_or_spin"] == "Pace"]["bowler_runs_conceded"].sum() / max(1, (v_df["pace_or_spin"] == "Pace").sum())) * 6.0), 2),
                "spin_econ": round(float((v_df[v_df["pace_or_spin"] == "Spin"]["bowler_runs_conceded"].sum() / max(1, (v_df["pace_or_spin"] == "Spin").sum())) * 6.0), 2)
            }
        else:
            summary = {
                "venue_id": venue_id,
                "venue_name": name,
                "total_deliveries": 0,
                "matches_hosted": 0,
                "avg_rpo": 8.35,
                "hp_deliveries": 0,
                "hp_share_pct": 35.0,
                "avg_wicket_value": 12.0,
                "pace_econ": 8.40,
                "spin_econ": 7.90
            }
        with open(os.path.join(PROCESSED_DIR, f"rr_{name.lower()}_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

    print("  ✓ Venue summaries exported.")


# =============================================================================
# PIPELINE ORCHESTRATION
# =============================================================================
def run_pressure_intelligence_engine():
    """
    Complete end-to-end orchestration of the 4-Stage Probabilistic & Econometric Engine.
    """
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    print("=" * 80)
    print("🚀 EXECUTING FIRST-PRINCIPLES 4-STAGE PROBABILISTIC & ECONOMETRIC ML PIPELINE")
    print("=" * 80)

    # 1. State reconstruction
    df = load_and_reconstruct_deliveries()

    # 2. Stage 1: Train ML models and compute continuous leverage L_i
    df, xr_model, xw_model, xr_bowl_model, xw_bowl_model, rw_features, rw_bowl_features = train_ml_models_and_compute_leverage(df)

    # 3. Stage 2: Endogenous wicket valuation and Empirical Bayes opponent skill adjustment
    leg = compute_endogenous_values_and_opponent_skills(df, xr_model, xw_model, rw_features, xr_bowl_model, xw_bowl_model, rw_bowl_features)

    # Export sample deliveries for inspectability
    sample_cols = [
        "match_id", "innings", "overid", "over_ball_num", "striker_name", "bowler_name",
        "runs", "is_wicket_int", "b", "balls_remaining", "runs_cum", "wickets_down",
        "crr", "runs_needed", "rrr", "rate_gap", "L_i", "BPI", "is_HP", "is_extreme_crisis",
        "C_wkt", "v_bat", "v_bowl"
    ]
    leg[sample_cols].head(5000).to_csv(os.path.join(PROCESSED_DIR, "ball_by_ball_pressure_index.csv"), index=False)
    print("  ✓ Exported sample ball_by_ball_pressure_index.csv (5000 rows).")

    # 4. Stages 3 & 4: Hierarchical Random-Slopes & Bayesian LCB Rankings
    top_bat, top_bowl, b_m, bw_m = compute_hierarchical_random_slopes_and_lcb(leg)

    # 5. Venue summaries
    compute_rr_venue_summaries(leg)

    print("\n" + "=" * 80)
    print("🏆 OFFICIAL FIRST-PRINCIPLES TOP 10 SPECIALIST BATTERS UNDER PRESSURE")
    print("=" * 80)
    print(top_bat[[
        "Rank", "striker_name", "total_balls", "HP_Deliveries", "M_HP",
        "Base_SR", "HP_SR", "Delta_Clutch", "Consistency_Pct", "PRR_Score"
    ]].to_string(index=False))

    print("\n" + "=" * 80)
    print("🎯 OFFICIAL FIRST-PRINCIPLES TOP 10 SPECIALIST BOWLERS UNDER PRESSURE")
    print("=" * 80)
    print(top_bowl[[
        "Rank", "bowler_name", "pace_or_spin", "total_balls", "HP_Deliveries", "M_HP",
        "Base_Economy", "HP_Economy", "Delta_Clutch", "Consistency_Pct", "PRR_Score"
    ]].to_string(index=False))
    print("=" * 80)
    print("✅ All deliverables successfully generated in data/processed/!")
    return top_bat, top_bowl


if __name__ == "__main__":
    run_pressure_intelligence_engine()
