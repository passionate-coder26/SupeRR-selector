"""
src/config.py — Configuration for SupeRR Selector Season 2 Pressure Analytics Engine
====================================================================================
Rajasthan Royals Pressure Intelligence & Player Reliability OS
4-Stage Probabilistic & Econometric Machine Learning Architecture
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
# Option B: High-Pressure threshold is learned endogenously via 2-Component GMM on log(1 + L_i)
GMM_RANDOM_STATE = 42
BPI_EXPONENTIAL_DECAY = 0.70       # BPI = 100 * (1 - exp(-0.70 * L_i))
PAR_RUN_RATE_T20 = 8.15             # Global par RPO
DEFAULT_VENUE_PAR_RPO = 8.35        # Venue par baseline

# Backward compatibility / Display fallbacks
LEVERAGE_BASELINE_REGIME = 1.0
LEVERAGE_HIGH_PRESSURE = 1.0        # Dynamically replaced by L*_GMM in analytics_engine
LEVERAGE_EXTREME_CRISIS = 2.0
HIGH_PRESSURE_THRESHOLD = 64.0
EXTREME_PRESSURE_THRESHOLD = 80.0

# =============================================================================
# 3. STAGE 2: ENDOGENOUS WICKET VALUATION & BIPARTITE RIDGE REGRESSION
# =============================================================================
# Option B: Wicket cost C_wkt is predicted by ML Remaining-Runs Regressor (rem_runs_model)
# Bowler wicket scaling factor lambda_wkt is learned via std(run_res) / std(wkt_res)
# Ridge alpha* is learned via 5-Fold GroupKFold cross-validation over [50, 500]
RIDGE_ALPHA_GRID = [50.0, 100.0, 150.0, 200.0, 250.0, 350.0, 500.0]
RIDGE_ALPHA = 200.0                # Baseline / fallback alpha

# =============================================================================
# 4. STAGES 3 & 4: EMPIRICAL BAYES ANOVA VARIANCE DECOMPOSITION & PCA WEIGHTING
# =============================================================================
# Option B: K_base and K_HP are estimated via ANOVA variance decomposition: sigma2_within / sigma2_between
# Beta prior (alpha_0, beta_0) is estimated via Method of Moments from sample match consistency rates
# Composite weights are learned via PCA(n_components=1) on [Z(mu_HP), Z(alpha), Z(Delta), Z(Consistency)]
# Volume cutoffs are computed as the empirical 75th percentile (upper quartile) of qualified players
Z_LEVERAGE_MIN = -0.8
Z_LEVERAGE_MAX = 3.5

# Fallback values
K_SHRINK_BASE = 250.0
K_SHRINK_HP = 140.0
PRR_WEIGHT_MU_HP = 0.55
PRR_WEIGHT_DELTA = 0.20
PRR_WEIGHT_CONSISTENCY = 0.25
PRR_PENALTY_SE_MATCH = 0.35
PRR_PROBIT_SCALE = 1.0             # Exact standard normal CDF Phi(Z) with sigma=1.0 under Option B
PRIOR_MATCH_PSEUDO_COUNT = 10.0
CONSISTENCY_FLOOR = 50.0

BATTER_MIN_HP_BALLS = 100
BATTER_MIN_HP_MATCHES = 12
BATTER_MIN_TOTAL_BALLS = 350

BOWLER_MIN_HP_BALLS = 120
BOWLER_MIN_HP_MATCHES = 14
BOWLER_MIN_TOTAL_BALLS = 500

# =============================================================================
# 6. VENUE MICRO-CLIMATES (Rajasthan Royals Focus)
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
