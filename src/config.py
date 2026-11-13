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
LEVERAGE_BASELINE_REGIME = 1.0     # L_i <= 1.0: Ordinary / Baseline Regime
LEVERAGE_HIGH_PRESSURE = 1.35      # L_i >= 1.35: True High-Leverage Crunch Regime (~top 18% deliveries)
LEVERAGE_EXTREME_CRISIS = 2.0      # L_i >= 2.0: Extreme Crisis Regime (2x match average leverage)
BPI_EXPONENTIAL_DECAY = 0.70       # BPI = 100 * (1 - exp(-0.70 * L_i))
HIGH_PRESSURE_THRESHOLD = 64.0      # Display index boundary
EXTREME_PRESSURE_THRESHOLD = 80.0   # Display crisis boundary
PAR_RUN_RATE_T20 = 8.15             # Global par RPO
DEFAULT_VENUE_PAR_RPO = 8.35        # Venue par baseline

# =============================================================================
# 3. STAGE 2: ENDOGENOUS WICKET VALUATION & BIPARTITE RIDGE REGRESSION
# =============================================================================
WICKET_VALUE_MIN = 4.0             # Minimum endogenous run value of a wicket
WICKET_VALUE_MAX = 24.0            # Maximum endogenous run value of a wicket
RIDGE_ALPHA = 200.0                # L2 regularization for simultaneous bipartite matchup network
K_SHRINK_OPPONENT_SKILL = 200.0    # Baseline shrinkage constant

# =============================================================================
# 4. STAGE 3: INDEPENDENT EMPIRICAL BAYES SHRINKAGE & CLUTCH ELEVATION
# =============================================================================
K_SHRINK_BASE = 250.0              # Sample shrinkage for baseline mean alpha (total balls)
K_SHRINK_HP = 140.0                # Sample shrinkage for High-Pressure mean mu_HP (HP balls)
K_SHRINK_ALPHA = 250.0             # Backward compatibility alias
K_SHRINK_BETA = 140.0              # Backward compatibility alias
Z_LEVERAGE_MIN = -0.8              # Centered leverage z_L = clip(L_i - 1.0, -0.8, 3.5)
Z_LEVERAGE_MAX = 3.5

# =============================================================================
# 5. STAGE 4: STANDARDIZED BAYESIAN LCB RANKINGS & UNIVERSAL ELIGIBILITY
# =============================================================================
# Standardized LCB Pressure Reliability Rating Formula:
# PRR_Raw = 0.55 * Z(mu_HP_EB) + 0.20 * Z(Delta_EB) + 0.25 * Z(Consistency_Pct) - 0.35 * Z(SE_match)
# PRR_Final = norm.cdf(0.85 * Z(PRR_Raw)) * 100
PRR_WEIGHT_MU_HP = 0.55
PRR_WEIGHT_DELTA = 0.20
PRR_WEIGHT_CONSISTENCY = 0.25
PRR_PENALTY_SE_MATCH = 0.35
PRR_PROBIT_SCALE = 0.85
PRIOR_MATCH_PSEUDO_COUNT = 10.0
CONSISTENCY_FLOOR = 55.0
ALPHA_FLOOR = -2.0

LCB_WEIGHT_EXPECTED_HP = 0.55      # Backward compatibility aliases
LCB_WEIGHT_CLUTCH_DELTA = 0.20
LCB_PENALTY_SE_MATCH = 0.35

# Universal Sample Size Eligibility:
# Batters: N_HP >= 100, M_HP >= 12, total_balls_faced >= 350, total_balls_faced >= total_balls_bowled
# Bowlers: N_HP >= 120, M_HP >= 14, total_balls_bowled >= 500, total_balls_bowled >= total_balls_faced
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
