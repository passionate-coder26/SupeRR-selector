"""
src/config.py — Configuration for SupeRR Selector Season 2 Pressure Analytics Engine
====================================================================================
Rajasthan Royals Pressure Intelligence & Player Reliability OS
Option B: Pure First-Principles 4-Stage Probabilistic & Econometric Machine Learning Architecture
All player-targeting thresholds, manual weights, fixed wicket penalties, and quotas removed.
"""

# =============================================================================
# 1. VISUAL THEME (Rajasthan Royals Dark Analytics Palette)
# =============================================================================
ROYALS_PINK = "#EA1A85"
DEEP_NAVY = "#0B0F19"
ROYAL_SLATE = "#161F33"
TEXT_WHITE = "#F8FAFC"
ACCENT_GOLD = "#FFD700"
ACCENT_CYAN = "#00E5FF"
ACCENT_GREEN = "#22C55E"
ACCENT_RED = "#EF4444"

# =============================================================================
# 2. STAGE 1: INFORMATION-THEORETIC WIN PROBABILITY & LEVERAGE (L_i)
# =============================================================================
# High-Pressure threshold is learned endogenously via 2-Component GMM on log(1 + L_i)
GMM_RANDOM_STATE = 42
BPI_EXPONENTIAL_DECAY = 0.70       # BPI = 100 * (1 - exp(-0.70 * L_i))
PAR_RUN_RATE_T20 = 8.15             # Global par RPO benchmark
DEFAULT_VENUE_PAR_RPO = 8.35        # Venue par baseline fallback

# =============================================================================
# 3. STAGE 2: ENDOGENOUS WICKET VALUATION & BIPARTITE RIDGE REGRESSION
# =============================================================================
# Wicket cost C_wkt is predicted by ML Remaining-Runs Regressor (rem_runs_model)
# Bowler wicket scaling factor lambda_wkt is learned via std(run_res) / std(wkt_res)
# Ridge alpha* is learned via 5-Fold GroupKFold cross-validation over RIDGE_ALPHA_GRID
RIDGE_ALPHA_GRID = [50.0, 100.0, 150.0, 200.0, 250.0, 350.0, 500.0]

# =============================================================================
# 4. VENUE MICRO-CLIMATES (Rajasthan Royals Focus)
# =============================================================================
VENUE_JAIPUR_ID = 6                 # Sawai Mansingh Stadium, Jaipur (Primary Home Fortress)
VENUE_GUWAHATI_ID = 1241            # Barsapara Cricket Stadium, Guwahati (Secondary Home)

VENUE_METADATA = {
    6: {
        "name": "Sawai Mansingh Stadium, Jaipur",
        "city": "Jaipur",
        "role": "Primary Home Fortress",
        "characteristics": "Large boundaries (78m), middle-overs spin grip, non-boundary running focus",
        "avg_1st_innings": 167.5,
        "chase_win_pct": 62.0,
    },
    1241: {
        "name": "Barsapara Cricket Stadium, Guwahati",
        "city": "Guwahati",
        "role": "Secondary Home",
        "characteristics": "Heavy 2nd-innings dew, high death-overs boundary %, wet-ball yorkers essential",
        "avg_1st_innings": 182.0,
        "chase_win_pct": 74.5,
    }
}
