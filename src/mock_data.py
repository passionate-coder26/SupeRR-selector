"""
src/mock_data.py — Synthetic Pre-6PM Domestic Scouting Dataset Generator
========================================================================
The Pink Canvas OS | SupeRR Selector 2.0 — IPL 2027 Uncapped Scouting Engine
Generates 35 realistic uncapped Indian domestic T20 players.
"""

import os
import sys
import numpy as np
import pandas as pd

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.analytics_engine import run_full_pipeline

np.random.seed(2027)

# =============================================================================
# PLAYER ARCHETYPES
# =============================================================================
PLAYER_DATA = [
    # --- FINISHER BATTERS (12 players) ---
    {"Player_ID": "RR-UC-001", "Player_Name": "Aakash Vashisht", "Age": 20, "State_Team": "Rajasthan", "League": "SMAT", "Role": "Finisher Batter",
     "Innings": 14, "Balls_Faced": 210, "Raw_SR": 168.5, "Non_Boundary_SR": 96.2, "Boundary_Pct": 61.0, "SR_vs_Spin": 172.3, "SR_vs_Pace": 162.8, "Death_SR": 182.0, "Optimal_Entry_Over": 15,
     "Balls_Bowled": 0, "Raw_Economy": np.nan, "Death_Economy": np.nan, "Middle_Dot_Pct": np.nan, "Wickets_Per_Match": np.nan,
     "Knockout_SR_Boost": 12.5, "Recency_Last_10_Trend": "145,152,165,171,168,175,180,178,185,182",
     "Played_At_Jaipur_Guwahati": True, "Venue_Note": "Played 3 SMAT matches at Sawai Mansingh. SR 175+ in 2/3.",
     "Impact_Sub_Specialist": True, "Specialist_Tag": "Death Finisher Impact Sub",
     "Expected_Price_Cr": 1.80, "Walk_Away_Ceiling_Cr": 3.20, "Rival_Threat_Teams": "CSK,PBKS",
     "Biomechanical_Note": "Deep-crease stable base generates 360-degree access vs wide yorkers. Hyper-extended wrist through cover drive creates flat-bat power through extra-cover and mid-wicket."},

    {"Player_ID": "RR-UC-002", "Player_Name": "Devraj Patel", "Age": 21, "State_Team": "Gujarat", "League": "SMAT", "Role": "Finisher Batter",
     "Innings": 16, "Balls_Faced": 278, "Raw_SR": 158.2, "Non_Boundary_SR": 92.5, "Boundary_Pct": 56.8, "SR_vs_Spin": 165.0, "SR_vs_Pace": 152.4, "Death_SR": 172.5, "Optimal_Entry_Over": 14,
     "Balls_Bowled": 12, "Raw_Economy": np.nan, "Death_Economy": np.nan, "Middle_Dot_Pct": np.nan, "Wickets_Per_Match": np.nan,
     "Knockout_SR_Boost": 8.3, "Recency_Last_10_Trend": "138,142,155,158,160,162,165,160,168,172",
     "Played_At_Jaipur_Guwahati": False, "Venue_Note": "No Jaipur/Guwahati exposure. Ahmedabad-heavy sample.",
     "Impact_Sub_Specialist": False, "Specialist_Tag": "",
     "Expected_Price_Cr": 1.20, "Walk_Away_Ceiling_Cr": 2.10, "Rival_Threat_Teams": "MI,CSK",
     "Biomechanical_Note": "Low backlift with quick hands generates bat speed from minimal trigger movement. Strong bottom-hand player, excellent at picking length early against death-overs pace."},

    {"Player_ID": "RR-UC-003", "Player_Name": "Harpreet Singh Bhatia", "Age": 21, "State_Team": "Punjab", "League": "SMAT", "Role": "Finisher Batter",
     "Innings": 18, "Balls_Faced": 312, "Raw_SR": 155.8, "Non_Boundary_SR": 94.8, "Boundary_Pct": 54.2, "SR_vs_Spin": 161.5, "SR_vs_Pace": 150.2, "Death_SR": 170.8, "Optimal_Entry_Over": 13,
     "Balls_Bowled": 36, "Raw_Economy": 9.8, "Death_Economy": 11.2, "Middle_Dot_Pct": 22.0, "Wickets_Per_Match": 0.2,
     "Knockout_SR_Boost": 15.2, "Recency_Last_10_Trend": "140,145,148,155,160,158,162,168,170,172",
     "Played_At_Jaipur_Guwahati": True, "Venue_Note": "2 SMAT matches at Barsapara. SR 180 in chase under dew.",
     "Impact_Sub_Specialist": True, "Specialist_Tag": "Chase Specialist Impact Sub",
     "Expected_Price_Cr": 1.50, "Walk_Away_Ceiling_Cr": 2.70, "Rival_Threat_Teams": "PBKS,RCB",
     "Biomechanical_Note": "High elbow set-up with open-face late adjustment enables precise placement into gaps on large Jaipur outfield. Elite runner between wickets — 2.8s average for 2-run completions."},

    {"Player_ID": "RR-UC-004", "Player_Name": "Keshav Ranjan", "Age": 19, "State_Team": "Bihar", "League": "UPT20", "Role": "Finisher Batter",
     "Innings": 10, "Balls_Faced": 142, "Raw_SR": 175.2, "Non_Boundary_SR": 88.5, "Boundary_Pct": 65.3, "SR_vs_Spin": 180.0, "SR_vs_Pace": 168.5, "Death_SR": 192.0, "Optimal_Entry_Over": 16,
     "Balls_Bowled": 0, "Raw_Economy": np.nan, "Death_Economy": np.nan, "Middle_Dot_Pct": np.nan, "Wickets_Per_Match": np.nan,
     "Knockout_SR_Boost": 18.5, "Recency_Last_10_Trend": "155,162,168,175,180,178,185,190,195,192",
     "Played_At_Jaipur_Guwahati": False, "Venue_Note": "No top-venue exposure. UP domestic circuit only.",
     "Impact_Sub_Specialist": True, "Specialist_Tag": "Death Overs Power Hitter",
     "Expected_Price_Cr": 0.60, "Walk_Away_Ceiling_Cr": 1.10, "Rival_Threat_Teams": "MI,PBKS",
     "Biomechanical_Note": "Explosive weight transfer from back foot to front generates 95m+ sixes. Bat swing arc peaks late — timing-dependent but devastating when set. Small sample: 142 balls."},

    {"Player_ID": "RR-UC-005", "Player_Name": "Tanay Thyagarajan", "Age": 20, "State_Team": "Tamil Nadu", "League": "TNPL", "Role": "Finisher Batter",
     "Innings": 15, "Balls_Faced": 245, "Raw_SR": 162.0, "Non_Boundary_SR": 98.5, "Boundary_Pct": 58.2, "SR_vs_Spin": 168.5, "SR_vs_Pace": 155.2, "Death_SR": 178.0, "Optimal_Entry_Over": 14,
     "Balls_Bowled": 24, "Raw_Economy": 9.2, "Death_Economy": 10.5, "Middle_Dot_Pct": 25.0, "Wickets_Per_Match": 0.3,
     "Knockout_SR_Boost": 10.8, "Recency_Last_10_Trend": "148,152,158,162,165,168,170,175,178,180",
     "Played_At_Jaipur_Guwahati": False, "Venue_Note": "TNPL exclusively. Chennai spin-friendly conditions.",
     "Impact_Sub_Specialist": False, "Specialist_Tag": "",
     "Expected_Price_Cr": 1.00, "Walk_Away_Ceiling_Cr": 1.80, "Rival_Threat_Teams": "CSK,DC",
     "Biomechanical_Note": "Compact technique with strong wrist roll through mid-wicket. TNPL spin-hitting credentials: SR 168 vs spin on Chennai turners proves genuine skill vs flight and dip."},

    {"Player_ID": "RR-UC-006", "Player_Name": "Rudra Pratap Chauhan", "Age": 18, "State_Team": "Uttar Pradesh", "League": "UPT20", "Role": "Finisher Batter",
     "Innings": 8, "Balls_Faced": 98, "Raw_SR": 178.5, "Non_Boundary_SR": 85.0, "Boundary_Pct": 68.2, "SR_vs_Spin": 182.0, "SR_vs_Pace": 172.5, "Death_SR": 198.0, "Optimal_Entry_Over": 17,
     "Balls_Bowled": 0, "Raw_Economy": np.nan, "Death_Economy": np.nan, "Middle_Dot_Pct": np.nan, "Wickets_Per_Match": np.nan,
     "Knockout_SR_Boost": 20.0, "Recency_Last_10_Trend": "160,165,170,178,182,185,190,195,198,200",
     "Played_At_Jaipur_Guwahati": False, "Venue_Note": "No venue exposure outside UP. 98 balls only — extreme small sample.",
     "Impact_Sub_Specialist": True, "Specialist_Tag": "Power Hitter Impact Sub",
     "Expected_Price_Cr": 0.40, "Walk_Away_Ceiling_Cr": 0.72, "Rival_Threat_Teams": "MI,PBKS,KKR",
     "Biomechanical_Note": "Raw power from strong forearms and fast bat speed. Minimal footwork — relies on hand-eye. HIGH CEILING but only 98 balls: Bayes will heavily regress these numbers."},

    {"Player_ID": "RR-UC-007", "Player_Name": "Mohit Agarwal", "Age": 21, "State_Team": "Karnataka", "League": "Maharaja T20", "Role": "Finisher Batter",
     "Innings": 12, "Balls_Faced": 195, "Raw_SR": 152.5, "Non_Boundary_SR": 100.5, "Boundary_Pct": 50.2, "SR_vs_Spin": 148.0, "SR_vs_Pace": 155.8, "Death_SR": 165.0, "Optimal_Entry_Over": 13,
     "Balls_Bowled": 0, "Raw_Economy": np.nan, "Death_Economy": np.nan, "Middle_Dot_Pct": np.nan, "Wickets_Per_Match": np.nan,
     "Knockout_SR_Boost": 5.2, "Recency_Last_10_Trend": "145,148,150,152,155,152,150,155,158,162",
     "Played_At_Jaipur_Guwahati": False, "Venue_Note": "Bangalore domestic circuit. Chinnaswamy-calibrated numbers.",
     "Impact_Sub_Specialist": False, "Specialist_Tag": "",
     "Expected_Price_Cr": 0.45, "Walk_Away_Ceiling_Cr": 0.80, "Rival_Threat_Teams": "RCB",
     "Biomechanical_Note": "Textbook technique with full face of bat. Consistent but lacks x-factor power. Reliable rotator on large grounds — ideal Jaipur running game complement."},

    {"Player_ID": "RR-UC-008", "Player_Name": "Vishnu Solanki", "Age": 17, "State_Team": "Rajasthan", "League": "SMAT", "Role": "Finisher Batter",
     "Innings": 20, "Balls_Faced": 345, "Raw_SR": 148.5, "Non_Boundary_SR": 102.0, "Boundary_Pct": 48.5, "SR_vs_Spin": 145.0, "SR_vs_Pace": 150.2, "Death_SR": 158.0, "Optimal_Entry_Over": 12,
     "Balls_Bowled": 48, "Raw_Economy": 8.8, "Death_Economy": 10.0, "Middle_Dot_Pct": 28.0, "Wickets_Per_Match": 0.4,
     "Knockout_SR_Boost": 6.8, "Recency_Last_10_Trend": "142,145,148,150,148,150,152,150,155,158",
     "Played_At_Jaipur_Guwahati": True, "Venue_Note": "Home ground Jaipur. 8 SMAT innings at Sawai Mansingh. NBSR 105 there.",
     "Impact_Sub_Specialist": False, "Specialist_Tag": "",
     "Expected_Price_Cr": 0.35, "Walk_Away_Ceiling_Cr": 0.63, "Rival_Threat_Teams": "DC",
     "Biomechanical_Note": "Jaipur local knowledge is valuable. Excellent running between wickets (NBSR 102). Ceiling is limited but floor is high. Safe domestic backup."},

    {"Player_ID": "RR-UC-009", "Player_Name": "Nikhil Gangta", "Age": 19, "State_Team": "Himachal Pradesh", "League": "SMAT", "Role": "Finisher Batter",
     "Innings": 11, "Balls_Faced": 165, "Raw_SR": 164.8, "Non_Boundary_SR": 90.2, "Boundary_Pct": 60.5, "SR_vs_Spin": 170.0, "SR_vs_Pace": 158.5, "Death_SR": 180.5, "Optimal_Entry_Over": 15,
     "Balls_Bowled": 0, "Raw_Economy": np.nan, "Death_Economy": np.nan, "Middle_Dot_Pct": np.nan, "Wickets_Per_Match": np.nan,
     "Knockout_SR_Boost": 14.0, "Recency_Last_10_Trend": "150,155,158,162,168,170,172,175,178,180",
     "Played_At_Jaipur_Guwahati": False, "Venue_Note": "Dharamsala conditions. Different from Jaipur/Guwahati.",
     "Impact_Sub_Specialist": True, "Specialist_Tag": "Spin Basher Impact Sub",
     "Expected_Price_Cr": 0.80, "Walk_Away_Ceiling_Cr": 1.44, "Rival_Threat_Teams": "MI,CSK,PBKS",
     "Biomechanical_Note": "Uses depth of crease to manufacture room against spin. Quick wrists snap through the ball at contact — generates power without big swing arc. Reminiscent of young Hetmyer."},

    {"Player_ID": "RR-UC-010", "Player_Name": "Anuj Rawat", "Age": 21, "State_Team": "Delhi", "League": "SMAT", "Role": "Finisher Batter",
     "Innings": 17, "Balls_Faced": 290, "Raw_SR": 151.2, "Non_Boundary_SR": 95.0, "Boundary_Pct": 52.8, "SR_vs_Spin": 155.0, "SR_vs_Pace": 148.0, "Death_SR": 168.0, "Optimal_Entry_Over": 13,
     "Balls_Bowled": 0, "Raw_Economy": np.nan, "Death_Economy": np.nan, "Middle_Dot_Pct": np.nan, "Wickets_Per_Match": np.nan,
     "Knockout_SR_Boost": 7.5, "Recency_Last_10_Trend": "142,145,148,152,155,152,158,160,162,165",
     "Played_At_Jaipur_Guwahati": True, "Venue_Note": "Played at Barsapara once. SR 162 in dew chase.",
     "Impact_Sub_Specialist": False, "Specialist_Tag": "",
     "Expected_Price_Cr": 0.50, "Walk_Away_Ceiling_Cr": 0.90, "Rival_Threat_Teams": "DC,RCB",
     "Biomechanical_Note": "Left-hand option adds variety. Clean ball-striking through off-side. Needs to develop death-overs power against pace."},

    {"Player_ID": "RR-UC-011", "Player_Name": "Priyam Garg", "Age": 20, "State_Team": "Uttar Pradesh", "League": "UPT20", "Role": "Finisher Batter",
     "Innings": 19, "Balls_Faced": 320, "Raw_SR": 146.5, "Non_Boundary_SR": 98.0, "Boundary_Pct": 48.0, "SR_vs_Spin": 142.0, "SR_vs_Pace": 150.0, "Death_SR": 160.0, "Optimal_Entry_Over": 12,
     "Balls_Bowled": 30, "Raw_Economy": 9.5, "Death_Economy": 10.8, "Middle_Dot_Pct": 24.0, "Wickets_Per_Match": 0.3,
     "Knockout_SR_Boost": 4.5, "Recency_Last_10_Trend": "140,142,145,148,145,148,150,148,152,150",
     "Played_At_Jaipur_Guwahati": False, "Venue_Note": "UP/Dehradun grounds. Average domestic circuit.",
     "Impact_Sub_Specialist": False, "Specialist_Tag": "",
     "Expected_Price_Cr": 0.30, "Walk_Away_Ceiling_Cr": 0.54, "Rival_Threat_Teams": "DC",
     "Biomechanical_Note": "Former India U19 captain. Mature head but T20 power game hasn't evolved. Good anchor but RR need explosiveness in this slot."},

    {"Player_ID": "RR-UC-012", "Player_Name": "Rajat Patidar Jr", "Age": 18, "State_Team": "Madhya Pradesh", "League": "MPT20", "Role": "Finisher Batter",
     "Innings": 7, "Balls_Faced": 85, "Raw_SR": 172.0, "Non_Boundary_SR": 82.5, "Boundary_Pct": 66.0, "SR_vs_Spin": 178.0, "SR_vs_Pace": 165.0, "Death_SR": 195.0, "Optimal_Entry_Over": 16,
     "Balls_Bowled": 0, "Raw_Economy": np.nan, "Death_Economy": np.nan, "Middle_Dot_Pct": np.nan, "Wickets_Per_Match": np.nan,
     "Knockout_SR_Boost": 22.0, "Recency_Last_10_Trend": "158,165,170,175,178,182,188,192,195,198",
     "Played_At_Jaipur_Guwahati": False, "Venue_Note": "MP domestic only. 85 balls — EXTREME small sample.",
     "Impact_Sub_Specialist": True, "Specialist_Tag": "Power Finisher Impact Sub",
     "Expected_Price_Cr": 0.30, "Walk_Away_Ceiling_Cr": 0.54, "Rival_Threat_Teams": "MI,RCB",
     "Biomechanical_Note": "Spectacular raw talent. Generates backspin on lofted drives creating carry. Only 85 balls in MPT20 — needs heavy Bayesian regression. Watch-list prospect for 2028 if not ready."},

    # --- DEATH PACERS (10 players) ---
    {"Player_ID": "RR-UC-013", "Player_Name": "Rajvardhan Hangargekar", "Age": 21, "State_Team": "Maharashtra", "League": "SMAT", "Role": "Death Pacer",
     "Innings": 16, "Balls_Faced": 42, "Raw_SR": 138.0, "Non_Boundary_SR": 78.0, "Boundary_Pct": 48.0, "SR_vs_Spin": 125.0, "SR_vs_Pace": 145.0, "Death_SR": 150.0, "Optimal_Entry_Over": 17,
     "Balls_Bowled": 348, "Raw_Economy": 7.2, "Death_Economy": 8.1, "Middle_Dot_Pct": 42.0, "Wickets_Per_Match": 1.8,
     "Knockout_SR_Boost": 5.0, "Recency_Last_10_Trend": "7.8,7.5,7.2,7.0,6.8,7.2,7.0,6.9,7.1,6.8",
     "Played_At_Jaipur_Guwahati": True, "Venue_Note": "1 match at Barsapara. Death economy 7.5 in dew.",
     "Impact_Sub_Specialist": False, "Specialist_Tag": "",
     "Expected_Price_Cr": 2.00, "Walk_Away_Ceiling_Cr": 3.60, "Rival_Threat_Teams": "MI,PBKS,CSK",
     "Biomechanical_Note": "Slingy low release point skids under bat in heavy Guwahati dew. 140+ kph consistently with excellent yorker accuracy. High front-arm action generates late inswing."},

    {"Player_ID": "RR-UC-014", "Player_Name": "Vyshak Vijaykumar", "Age": 20, "State_Team": "Karnataka", "League": "Maharaja T20", "Role": "Death Pacer",
     "Innings": 18, "Balls_Faced": 28, "Raw_SR": 130.0, "Non_Boundary_SR": 72.0, "Boundary_Pct": 42.0, "SR_vs_Spin": 118.0, "SR_vs_Pace": 138.0, "Death_SR": 142.0, "Optimal_Entry_Over": 16,
     "Balls_Bowled": 396, "Raw_Economy": 7.5, "Death_Economy": 8.4, "Middle_Dot_Pct": 40.0, "Wickets_Per_Match": 1.6,
     "Knockout_SR_Boost": 3.0, "Recency_Last_10_Trend": "8.0,7.8,7.5,7.6,7.4,7.2,7.5,7.3,7.2,7.0",
     "Played_At_Jaipur_Guwahati": False, "Venue_Note": "Bangalore/Hubli pitches. Seam-friendly conditions.",
     "Impact_Sub_Specialist": False, "Specialist_Tag": "",
     "Expected_Price_Cr": 1.50, "Walk_Away_Ceiling_Cr": 2.70, "Rival_Threat_Teams": "RCB,MI",
     "Biomechanical_Note": "Strong seam position through delivery stride. Uses crease width intelligently. Generates steep bounce from good length — effective even on flat Guwahati pitches."},

    {"Player_ID": "RR-UC-015", "Player_Name": "Arjun Tendulkar", "Age": 21, "State_Team": "Goa", "League": "SMAT", "Role": "Death Pacer",
     "Innings": 14, "Balls_Faced": 38, "Raw_SR": 125.0, "Non_Boundary_SR": 68.0, "Boundary_Pct": 38.0, "SR_vs_Spin": 112.0, "SR_vs_Pace": 132.0, "Death_SR": 135.0, "Optimal_Entry_Over": 18,
     "Balls_Bowled": 288, "Raw_Economy": 8.0, "Death_Economy": 8.8, "Middle_Dot_Pct": 36.0, "Wickets_Per_Match": 1.2,
     "Knockout_SR_Boost": 2.0, "Recency_Last_10_Trend": "8.5,8.2,8.0,7.8,8.2,8.0,7.8,7.5,7.8,8.0",
     "Played_At_Jaipur_Guwahati": False, "Venue_Note": "Goa domestic. Left-arm angle adds variety.",
     "Impact_Sub_Specialist": False, "Specialist_Tag": "",
     "Expected_Price_Cr": 0.80, "Walk_Away_Ceiling_Cr": 1.44, "Rival_Threat_Teams": "MI",
     "Biomechanical_Note": "Left-arm over with natural angle into right-handers. Developing yorker but wet-ball accuracy improving. Lower release generates skid. Media attention is a distraction factor."},

    {"Player_ID": "RR-UC-016", "Player_Name": "Vidwath Kaverappa", "Age": 20, "State_Team": "Karnataka", "League": "SMAT", "Role": "Death Pacer",
     "Innings": 15, "Balls_Faced": 22, "Raw_SR": 120.0, "Non_Boundary_SR": 65.0, "Boundary_Pct": 35.0, "SR_vs_Spin": 105.0, "SR_vs_Pace": 128.0, "Death_SR": 130.0, "Optimal_Entry_Over": 18,
     "Balls_Bowled": 324, "Raw_Economy": 7.8, "Death_Economy": 8.6, "Middle_Dot_Pct": 38.5, "Wickets_Per_Match": 1.4,
     "Knockout_SR_Boost": 4.0, "Recency_Last_10_Trend": "8.2,8.0,7.8,7.5,7.8,7.6,7.4,7.8,7.5,7.2",
     "Played_At_Jaipur_Guwahati": False, "Venue_Note": "Karnataka domestic circuit. Good seam conditions.",
     "Impact_Sub_Specialist": False, "Specialist_Tag": "",
     "Expected_Price_Cr": 0.50, "Walk_Away_Ceiling_Cr": 0.90, "Rival_Threat_Teams": "RCB,DC",
     "Biomechanical_Note": "Tall (6'3\") with steep bounce generation. Hits hard lengths consistently. Death yorker is work in progress but raw pace (138+) and bounce are assets on any surface."},

    {"Player_ID": "RR-UC-017", "Player_Name": "Harsh Dubey", "Age": 20, "State_Team": "Madhya Pradesh", "League": "MPT20", "Role": "Death Pacer",
     "Innings": 12, "Balls_Faced": 18, "Raw_SR": 115.0, "Non_Boundary_SR": 62.0, "Boundary_Pct": 32.0, "SR_vs_Spin": 100.0, "SR_vs_Pace": 122.0, "Death_SR": 125.0, "Optimal_Entry_Over": 17,
     "Balls_Bowled": 258, "Raw_Economy": 7.0, "Death_Economy": 7.8, "Middle_Dot_Pct": 44.0, "Wickets_Per_Match": 1.9,
     "Knockout_SR_Boost": 6.0, "Recency_Last_10_Trend": "7.5,7.2,7.0,6.8,7.0,6.8,6.5,6.8,7.0,6.5",
     "Played_At_Jaipur_Guwahati": True, "Venue_Note": "1 match at Sawai Mansingh. Economy 6.8 — swing in early overs.",
     "Impact_Sub_Specialist": False, "Specialist_Tag": "",
     "Expected_Price_Cr": 1.20, "Walk_Away_Ceiling_Cr": 2.16, "Rival_Threat_Teams": "MI,PBKS,CSK",
     "Biomechanical_Note": "Exceptional death-overs yorker accuracy — 78% hitting the blockhole in MPT20. Low-arm sling delivery creates skid and late movement. Wet-ball specialist potential for Guwahati dew games."},

    {"Player_ID": "RR-UC-018", "Player_Name": "Baltej Singh", "Age": 21, "State_Team": "Punjab", "League": "SMAT", "Role": "Death Pacer",
     "Innings": 17, "Balls_Faced": 15, "Raw_SR": 110.0, "Non_Boundary_SR": 58.0, "Boundary_Pct": 28.0, "SR_vs_Spin": 95.0, "SR_vs_Pace": 118.0, "Death_SR": 120.0, "Optimal_Entry_Over": 17,
     "Balls_Bowled": 372, "Raw_Economy": 7.8, "Death_Economy": 8.5, "Middle_Dot_Pct": 40.0, "Wickets_Per_Match": 1.5,
     "Knockout_SR_Boost": 3.5, "Recency_Last_10_Trend": "8.2,8.0,7.8,7.5,7.8,7.6,7.5,7.8,7.5,7.2",
     "Played_At_Jaipur_Guwahati": False, "Venue_Note": "Mohali/Dharamsala conditions. Swing-friendly.",
     "Impact_Sub_Specialist": False, "Specialist_Tag": "",
     "Expected_Price_Cr": 0.40, "Walk_Away_Ceiling_Cr": 0.72, "Rival_Threat_Teams": "PBKS",
     "Biomechanical_Note": "Tall left-arm seamer with natural shape away from right-handers. Consistent 135+ kph. Durable — bowled 372 balls across SMAT season without injury concern. Workhorse profile."},

    {"Player_ID": "RR-UC-019", "Player_Name": "Saurabh Netravalkar Jr", "Age": 19, "State_Team": "Mumbai", "League": "SMAT", "Role": "Death Pacer",
     "Innings": 10, "Balls_Faced": 12, "Raw_SR": 108.0, "Non_Boundary_SR": 55.0, "Boundary_Pct": 25.0, "SR_vs_Spin": 92.0, "SR_vs_Pace": 115.0, "Death_SR": 118.0, "Optimal_Entry_Over": 18,
     "Balls_Bowled": 210, "Raw_Economy": 7.4, "Death_Economy": 8.2, "Middle_Dot_Pct": 43.0, "Wickets_Per_Match": 1.7,
     "Knockout_SR_Boost": 8.0, "Recency_Last_10_Trend": "8.0,7.8,7.5,7.2,7.0,7.2,7.0,6.8,7.2,7.0",
     "Played_At_Jaipur_Guwahati": False, "Venue_Note": "Wankhede/BKC conditions. Mumbai SMAT elite opposition.",
     "Impact_Sub_Specialist": False, "Specialist_Tag": "",
     "Expected_Price_Cr": 0.90, "Walk_Away_Ceiling_Cr": 1.62, "Rival_Threat_Teams": "MI,CSK",
     "Biomechanical_Note": "Left-arm medium-fast with exceptional new-ball movement. Uses crease angle to create doubt. Death-overs wide yorker improving. Mumbai system product — mentally tough."},

    {"Player_ID": "RR-UC-020", "Player_Name": "Raj Bawa", "Age": 20, "State_Team": "Chandigarh", "League": "SMAT", "Role": "Death Pacer",
     "Innings": 13, "Balls_Faced": 35, "Raw_SR": 132.0, "Non_Boundary_SR": 75.0, "Boundary_Pct": 44.0, "SR_vs_Spin": 120.0, "SR_vs_Pace": 140.0, "Death_SR": 145.0, "Optimal_Entry_Over": 16,
     "Balls_Bowled": 282, "Raw_Economy": 8.2, "Death_Economy": 9.0, "Middle_Dot_Pct": 35.0, "Wickets_Per_Match": 1.3,
     "Knockout_SR_Boost": 5.0, "Recency_Last_10_Trend": "8.8,8.5,8.2,8.0,8.2,8.0,7.8,8.0,8.2,7.8",
     "Played_At_Jaipur_Guwahati": False, "Venue_Note": "Chandigarh/Mohali. Moderate quality opposition.",
     "Impact_Sub_Specialist": False, "Specialist_Tag": "",
     "Expected_Price_Cr": 0.40, "Walk_Away_Ceiling_Cr": 0.72, "Rival_Threat_Teams": "PBKS,DC",
     "Biomechanical_Note": "All-rounder profile — useful batting adds value. Pace is moderate (130-135) but disciplined lines. Death bowling economy needs improvement but has the athletic build to develop."},

    {"Player_ID": "RR-UC-021", "Player_Name": "Nishant Sindhu", "Age": 20, "State_Team": "Haryana", "League": "SMAT", "Role": "Death Pacer",
     "Innings": 11, "Balls_Faced": 20, "Raw_SR": 118.0, "Non_Boundary_SR": 60.0, "Boundary_Pct": 30.0, "SR_vs_Spin": 102.0, "SR_vs_Pace": 125.0, "Death_SR": 128.0, "Optimal_Entry_Over": 17,
     "Balls_Bowled": 234, "Raw_Economy": 7.6, "Death_Economy": 8.5, "Middle_Dot_Pct": 41.0, "Wickets_Per_Match": 1.5,
     "Knockout_SR_Boost": 4.5, "Recency_Last_10_Trend": "8.0,7.8,7.6,7.5,7.4,7.6,7.4,7.2,7.5,7.2",
     "Played_At_Jaipur_Guwahati": True, "Venue_Note": "1 SMAT match at Jaipur. Took 3 wickets with swing.",
     "Impact_Sub_Specialist": False, "Specialist_Tag": "",
     "Expected_Price_Cr": 0.45, "Walk_Away_Ceiling_Cr": 0.81, "Rival_Threat_Teams": "DC,KKR",
     "Biomechanical_Note": "Right-arm medium-fast with good cutter variation. Uses slower bouncer effectively at death. Consistent rather than spectacular — reliable backup pacer profile."},

    {"Player_ID": "RR-UC-022", "Player_Name": "Yash Dayal Jr", "Age": 19, "State_Team": "Uttar Pradesh", "League": "UPT20", "Role": "Death Pacer",
     "Innings": 9, "Balls_Faced": 10, "Raw_SR": 105.0, "Non_Boundary_SR": 50.0, "Boundary_Pct": 22.0, "SR_vs_Spin": 88.0, "SR_vs_Pace": 112.0, "Death_SR": 115.0, "Optimal_Entry_Over": 18,
     "Balls_Bowled": 186, "Raw_Economy": 7.1, "Death_Economy": 7.9, "Middle_Dot_Pct": 45.0, "Wickets_Per_Match": 2.0,
     "Knockout_SR_Boost": 10.0, "Recency_Last_10_Trend": "7.8,7.5,7.2,7.0,6.8,7.0,6.8,6.5,6.8,6.5",
     "Played_At_Jaipur_Guwahati": False, "Venue_Note": "UP domestic. Left-arm adds variety to RR's attack.",
     "Impact_Sub_Specialist": True, "Specialist_Tag": "Death Overs Specialist Impact Sub",
     "Expected_Price_Cr": 0.70, "Walk_Away_Ceiling_Cr": 1.26, "Rival_Threat_Teams": "MI,PBKS,CSK",
     "Biomechanical_Note": "Left-arm fast with genuine pace (140+). Natural angle creates problems for right-handers. Death yorker at the stumps is elite. 186 balls — decent sample but needs more data."},

    # --- WRIST SPINNERS (8 players) ---
    {"Player_ID": "RR-UC-023", "Player_Name": "Noor Ahmad (Domestic)", "Age": 20, "State_Team": "Saurashtra", "League": "SMAT", "Role": "Wrist Spinner",
     "Innings": 15, "Balls_Faced": 18, "Raw_SR": 95.0, "Non_Boundary_SR": 55.0, "Boundary_Pct": 22.0, "SR_vs_Spin": 80.0, "SR_vs_Pace": 102.0, "Death_SR": 110.0, "Optimal_Entry_Over": 7,
     "Balls_Bowled": 342, "Raw_Economy": 6.5, "Death_Economy": 7.8, "Middle_Dot_Pct": 52.0, "Wickets_Per_Match": 1.8,
     "Knockout_SR_Boost": 2.0, "Recency_Last_10_Trend": "7.0,6.8,6.5,6.2,6.5,6.3,6.0,6.2,6.5,6.2",
     "Played_At_Jaipur_Guwahati": True, "Venue_Note": "2 matches at Sawai Mansingh. Economy 5.8 — spin gripped beautifully.",
     "Impact_Sub_Specialist": False, "Specialist_Tag": "",
     "Expected_Price_Cr": 1.80, "Walk_Away_Ceiling_Cr": 3.24, "Rival_Threat_Teams": "KKR,MI,PBKS",
     "Biomechanical_Note": "Hyper-extended wrist release adds +220 RPM overspin on dry Jaipur surfaces. Left-arm wrist spin with big googly variation. Loop and dip on Jaipur's large square boundaries is devastating."},

    {"Player_ID": "RR-UC-024", "Player_Name": "Suyash Sharma", "Age": 20, "State_Team": "Uttar Pradesh", "League": "UPT20", "Role": "Wrist Spinner",
     "Innings": 13, "Balls_Faced": 15, "Raw_SR": 88.0, "Non_Boundary_SR": 48.0, "Boundary_Pct": 18.0, "SR_vs_Spin": 72.0, "SR_vs_Pace": 95.0, "Death_SR": 105.0, "Optimal_Entry_Over": 7,
     "Balls_Bowled": 294, "Raw_Economy": 6.8, "Death_Economy": 8.0, "Middle_Dot_Pct": 48.0, "Wickets_Per_Match": 1.6,
     "Knockout_SR_Boost": 3.0, "Recency_Last_10_Trend": "7.2,7.0,6.8,6.5,6.8,6.5,6.2,6.5,6.8,6.5",
     "Played_At_Jaipur_Guwahati": False, "Venue_Note": "UP/Lucknow conditions. Flat surfaces.",
     "Impact_Sub_Specialist": False, "Specialist_Tag": "",
     "Expected_Price_Cr": 1.00, "Walk_Away_Ceiling_Cr": 1.80, "Rival_Threat_Teams": "KKR,DC",
     "Biomechanical_Note": "Mystery spinner with carrom ball and googly. High-arm action creates dip. Overs 7-15 specialist — dot ball machine. Needs to develop death-overs plan."},

    {"Player_ID": "RR-UC-025", "Player_Name": "Manav Suthar", "Age": 21, "State_Team": "Rajasthan", "League": "SMAT", "Role": "Wrist Spinner",
     "Innings": 18, "Balls_Faced": 25, "Raw_SR": 92.0, "Non_Boundary_SR": 52.0, "Boundary_Pct": 20.0, "SR_vs_Spin": 78.0, "SR_vs_Pace": 100.0, "Death_SR": 108.0, "Optimal_Entry_Over": 7,
     "Balls_Bowled": 402, "Raw_Economy": 6.2, "Death_Economy": 7.5, "Middle_Dot_Pct": 54.0, "Wickets_Per_Match": 2.0,
     "Knockout_SR_Boost": 4.0, "Recency_Last_10_Trend": "6.8,6.5,6.2,6.0,6.2,6.0,5.8,6.0,6.2,5.8",
     "Played_At_Jaipur_Guwahati": True, "Venue_Note": "HOME GROUND Jaipur — 6 matches at Sawai Mansingh. Econ 5.5 avg!",
     "Impact_Sub_Specialist": False, "Specialist_Tag": "",
     "Expected_Price_Cr": 1.50, "Walk_Away_Ceiling_Cr": 2.70, "Rival_Threat_Teams": "KKR,MI,CSK,PBKS",
     "Biomechanical_Note": "Left-arm orthodox with exceptional control. Flight and dip on Jaipur's dry surfaces. 54% dot ball rate in middle overs is elite. Local Jaipur knowledge invaluable for 1st innings spin choke strategy."},

    {"Player_ID": "RR-UC-026", "Player_Name": "Rahul Chahar Jr", "Age": 19, "State_Team": "Rajasthan", "League": "SMAT", "Role": "Wrist Spinner",
     "Innings": 10, "Balls_Faced": 12, "Raw_SR": 85.0, "Non_Boundary_SR": 45.0, "Boundary_Pct": 15.0, "SR_vs_Spin": 68.0, "SR_vs_Pace": 92.0, "Death_SR": 100.0, "Optimal_Entry_Over": 8,
     "Balls_Bowled": 216, "Raw_Economy": 6.8, "Death_Economy": 8.2, "Middle_Dot_Pct": 50.0, "Wickets_Per_Match": 1.5,
     "Knockout_SR_Boost": 5.0, "Recency_Last_10_Trend": "7.2,7.0,6.8,6.5,6.8,6.5,6.2,6.5,6.8,6.5",
     "Played_At_Jaipur_Guwahati": True, "Venue_Note": "3 matches at Sawai Mansingh. Leg spin + googly effective.",
     "Impact_Sub_Specialist": True, "Specialist_Tag": "Middle Overs Strangler",
     "Expected_Price_Cr": 0.50, "Walk_Away_Ceiling_Cr": 0.90, "Rival_Threat_Teams": "KKR",
     "Biomechanical_Note": "Classic leg-spinner with good loop and turn. Googly is well-disguised. Middle-overs control artist. Needs more volume (216 balls) for confident projection."},

    {"Player_ID": "RR-UC-027", "Player_Name": "Venkatesh Iyer Jr", "Age": 21, "State_Team": "Madhya Pradesh", "League": "MPT20", "Role": "Wrist Spinner",
     "Innings": 14, "Balls_Faced": 20, "Raw_SR": 90.0, "Non_Boundary_SR": 50.0, "Boundary_Pct": 18.0, "SR_vs_Spin": 75.0, "SR_vs_Pace": 98.0, "Death_SR": 105.0, "Optimal_Entry_Over": 8,
     "Balls_Bowled": 312, "Raw_Economy": 7.0, "Death_Economy": 8.5, "Middle_Dot_Pct": 46.0, "Wickets_Per_Match": 1.4,
     "Knockout_SR_Boost": 3.5, "Recency_Last_10_Trend": "7.5,7.2,7.0,6.8,7.0,6.8,6.5,6.8,7.0,6.8",
     "Played_At_Jaipur_Guwahati": False, "Venue_Note": "Indore/Gwalior conditions. Decent spin tracks.",
     "Impact_Sub_Specialist": False, "Specialist_Tag": "",
     "Expected_Price_Cr": 0.40, "Walk_Away_Ceiling_Cr": 0.72, "Rival_Threat_Teams": "DC",
     "Biomechanical_Note": "Right-arm leg-break with decent turn but limited variation. Dot ball % is solid but lacks wicket-taking threat. Budget backup option."},

    {"Player_ID": "RR-UC-028", "Player_Name": "Akash Senthilnathan", "Age": 20, "State_Team": "Tamil Nadu", "League": "TNPL", "Role": "Wrist Spinner",
     "Innings": 16, "Balls_Faced": 22, "Raw_SR": 92.0, "Non_Boundary_SR": 52.0, "Boundary_Pct": 20.0, "SR_vs_Spin": 78.0, "SR_vs_Pace": 100.0, "Death_SR": 108.0, "Optimal_Entry_Over": 7,
     "Balls_Bowled": 358, "Raw_Economy": 6.6, "Death_Economy": 7.9, "Middle_Dot_Pct": 50.0, "Wickets_Per_Match": 1.7,
     "Knockout_SR_Boost": 6.0, "Recency_Last_10_Trend": "7.0,6.8,6.5,6.2,6.5,6.3,6.0,6.2,6.5,6.2",
     "Played_At_Jaipur_Guwahati": False, "Venue_Note": "TNPL spin-friendly conditions. Chepauk-calibrated numbers.",
     "Impact_Sub_Specialist": False, "Specialist_Tag": "",
     "Expected_Price_Cr": 0.80, "Walk_Away_Ceiling_Cr": 1.44, "Rival_Threat_Teams": "CSK,KKR",
     "Biomechanical_Note": "Right-arm mystery spinner with carrom ball as primary weapon. High overspin generates dip on any surface. TNPL numbers are gold standard for spin validation."},

    {"Player_ID": "RR-UC-029", "Player_Name": "Prashant Solanki", "Age": 20, "State_Team": "Gujarat", "League": "SMAT", "Role": "Wrist Spinner",
     "Innings": 17, "Balls_Faced": 28, "Raw_SR": 95.0, "Non_Boundary_SR": 55.0, "Boundary_Pct": 22.0, "SR_vs_Spin": 82.0, "SR_vs_Pace": 105.0, "Death_SR": 112.0, "Optimal_Entry_Over": 8,
     "Balls_Bowled": 378, "Raw_Economy": 7.2, "Death_Economy": 8.4, "Middle_Dot_Pct": 44.0, "Wickets_Per_Match": 1.4,
     "Knockout_SR_Boost": 3.0, "Recency_Last_10_Trend": "7.5,7.2,7.0,7.2,7.0,6.8,7.0,7.2,7.0,6.8",
     "Played_At_Jaipur_Guwahati": False, "Venue_Note": "Ahmedabad/Rajkot conditions.",
     "Impact_Sub_Specialist": False, "Specialist_Tag": "",
     "Expected_Price_Cr": 0.35, "Walk_Away_Ceiling_Cr": 0.63, "Rival_Threat_Teams": "DC",
     "Biomechanical_Note": "Left-arm chinaman with deceptive flight. Economy is steady but not elite. Useful variation bowler but unlikely to be primary threat. Budget depth option."},

    {"Player_ID": "RR-UC-030", "Player_Name": "Karan Lal", "Age": 18, "State_Team": "Rajasthan", "League": "SMAT", "Role": "Wrist Spinner",
     "Innings": 7, "Balls_Faced": 8, "Raw_SR": 82.0, "Non_Boundary_SR": 42.0, "Boundary_Pct": 12.0, "SR_vs_Spin": 65.0, "SR_vs_Pace": 88.0, "Death_SR": 95.0, "Optimal_Entry_Over": 8,
     "Balls_Bowled": 150, "Raw_Economy": 6.4, "Death_Economy": 7.6, "Middle_Dot_Pct": 55.0, "Wickets_Per_Match": 2.1,
     "Knockout_SR_Boost": 8.0, "Recency_Last_10_Trend": "7.0,6.8,6.5,6.2,6.0,6.2,6.0,5.8,6.2,5.8",
     "Played_At_Jaipur_Guwahati": True, "Venue_Note": "Home ground Jaipur. 4 matches. Economy 5.2! But only 150 balls.",
     "Impact_Sub_Specialist": True, "Specialist_Tag": "Dot Ball Machine Impact Sub",
     "Expected_Price_Cr": 0.30, "Walk_Away_Ceiling_Cr": 0.54, "Rival_Threat_Teams": "KKR,MI",
     "Biomechanical_Note": "Precocious 18-year-old leg-spinner. Extraordinary dot ball rate (55%) but ONLY 150 balls — heavy Bayes regression needed. Jaipur local. High ceiling developmental prospect."},

    # --- POWERPLAY OPENERS (3 players) ---
    {"Player_ID": "RR-UC-031", "Player_Name": "Abhishek Sharma Jr", "Age": 21, "State_Team": "Punjab", "League": "SMAT", "Role": "Powerplay Opener",
     "Innings": 16, "Balls_Faced": 380, "Raw_SR": 155.0, "Non_Boundary_SR": 88.0, "Boundary_Pct": 62.0, "SR_vs_Spin": 148.0, "SR_vs_Pace": 162.0, "Death_SR": 170.0, "Optimal_Entry_Over": 1,
     "Balls_Bowled": 48, "Raw_Economy": 8.5, "Death_Economy": 10.0, "Middle_Dot_Pct": 28.0, "Wickets_Per_Match": 0.3,
     "Knockout_SR_Boost": 8.0, "Recency_Last_10_Trend": "148,152,155,158,160,155,158,162,160,165",
     "Played_At_Jaipur_Guwahati": True, "Venue_Note": "1 SMAT match at Barsapara. SR 172 in powerplay chase.",
     "Impact_Sub_Specialist": False, "Specialist_Tag": "",
     "Expected_Price_Cr": 1.50, "Walk_Away_Ceiling_Cr": 2.70, "Rival_Threat_Teams": "PBKS,MI,CSK",
     "Biomechanical_Note": "Left-hand opener with powerful pull shot. Clears front leg to drive through covers. Powerplay boundary % elite (62%). Can bowl part-time left-arm spin."},

    {"Player_ID": "RR-UC-032", "Player_Name": "Prabhsimran Singh", "Age": 20, "State_Team": "Punjab", "League": "SMAT", "Role": "Powerplay Opener",
     "Innings": 18, "Balls_Faced": 420, "Raw_SR": 148.5, "Non_Boundary_SR": 92.0, "Boundary_Pct": 55.0, "SR_vs_Spin": 142.0, "SR_vs_Pace": 155.0, "Death_SR": 162.0, "Optimal_Entry_Over": 1,
     "Balls_Bowled": 0, "Raw_Economy": np.nan, "Death_Economy": np.nan, "Middle_Dot_Pct": np.nan, "Wickets_Per_Match": np.nan,
     "Knockout_SR_Boost": 5.0, "Recency_Last_10_Trend": "142,145,148,150,148,152,150,155,152,158",
     "Played_At_Jaipur_Guwahati": False, "Venue_Note": "Mohali/Dharamsala. Consistent SMAT opener.",
     "Impact_Sub_Specialist": False, "Specialist_Tag": "",
     "Expected_Price_Cr": 0.50, "Walk_Away_Ceiling_Cr": 0.90, "Rival_Threat_Teams": "PBKS",
     "Biomechanical_Note": "Aggressive right-hand opener. Strong through the arc from mid-wicket to long-on. High balls-faced volume (420) gives statistical confidence. Solid backup opener."},

    {"Player_ID": "RR-UC-033", "Player_Name": "Dhruv Shorey", "Age": 17, "State_Team": "Delhi", "League": "SMAT", "Role": "Powerplay Opener",
     "Innings": 15, "Balls_Faced": 350, "Raw_SR": 142.0, "Non_Boundary_SR": 95.0, "Boundary_Pct": 48.0, "SR_vs_Spin": 138.0, "SR_vs_Pace": 148.0, "Death_SR": 155.0, "Optimal_Entry_Over": 1,
     "Balls_Bowled": 0, "Raw_Economy": np.nan, "Death_Economy": np.nan, "Middle_Dot_Pct": np.nan, "Wickets_Per_Match": np.nan,
     "Knockout_SR_Boost": 4.0, "Recency_Last_10_Trend": "138,140,142,145,142,148,145,148,150,152",
     "Played_At_Jaipur_Guwahati": True, "Venue_Note": "1 match at Sawai Mansingh. SR 155 with 8 boundaries.",
     "Impact_Sub_Specialist": False, "Specialist_Tag": "",
     "Expected_Price_Cr": 0.30, "Walk_Away_Ceiling_Cr": 0.54, "Rival_Threat_Teams": "DC",
     "Biomechanical_Note": "Left-hand opener with high elbow technique. Good against pace but spin SR needs work. Jaipur running game suits his style. Budget anchor option."},

    # --- SPIN ALL-ROUNDERS (2 players) ---
    {"Player_ID": "RR-UC-034", "Player_Name": "Ravisrinivasan Sai Kishore", "Age": 21, "State_Team": "Tamil Nadu", "League": "TNPL", "Role": "Spin All-Rounder",
     "Innings": 16, "Balls_Faced": 180, "Raw_SR": 142.0, "Non_Boundary_SR": 90.0, "Boundary_Pct": 48.0, "SR_vs_Spin": 135.0, "SR_vs_Pace": 148.0, "Death_SR": 158.0, "Optimal_Entry_Over": 12,
     "Balls_Bowled": 360, "Raw_Economy": 6.8, "Death_Economy": 8.0, "Middle_Dot_Pct": 46.0, "Wickets_Per_Match": 1.4,
     "Knockout_SR_Boost": 6.0, "Recency_Last_10_Trend": "7.2,7.0,6.8,6.5,6.8,6.5,6.2,6.5,6.8,6.5",
     "Played_At_Jaipur_Guwahati": False, "Venue_Note": "Chennai/TNPL circuit. Elite opposition quality.",
     "Impact_Sub_Specialist": True, "Specialist_Tag": "Batting All-Rounder Impact Sub",
     "Expected_Price_Cr": 1.20, "Walk_Away_Ceiling_Cr": 2.16, "Rival_Threat_Teams": "CSK,KKR,MI",
     "Biomechanical_Note": "Left-arm orthodox with TNPL-proven economy. Batting has improved — SR 142 with genuine power. Dual-role value makes him a premium uncapped asset. Jadeja-lite comp."},

    {"Player_ID": "RR-UC-035", "Player_Name": "Shams Mulani", "Age": 21, "State_Team": "Mumbai", "League": "SMAT", "Role": "Spin All-Rounder",
     "Innings": 17, "Balls_Faced": 200, "Raw_SR": 138.0, "Non_Boundary_SR": 88.0, "Boundary_Pct": 45.0, "SR_vs_Spin": 130.0, "SR_vs_Pace": 145.0, "Death_SR": 152.0, "Optimal_Entry_Over": 11,
     "Balls_Bowled": 384, "Raw_Economy": 7.2, "Death_Economy": 8.5, "Middle_Dot_Pct": 42.0, "Wickets_Per_Match": 1.2,
     "Knockout_SR_Boost": 5.0, "Recency_Last_10_Trend": "7.5,7.2,7.0,7.2,7.0,6.8,7.0,7.2,7.0,6.8",
     "Played_At_Jaipur_Guwahati": False, "Venue_Note": "Mumbai SMAT — Wankhede. Elite opposition (Mumbai cricket).",
     "Impact_Sub_Specialist": False, "Specialist_Tag": "",
     "Expected_Price_Cr": 0.50, "Walk_Away_Ceiling_Cr": 0.90, "Rival_Threat_Teams": "MI,DC",
     "Biomechanical_Note": "Left-arm orthodox with excellent control. Mumbai domestic cricket has hardened his temperament. Batting SR 138 makes him genuine #7-8 option. Budget Jadeja profile."},
]


def generate_mock_dataset() -> pd.DataFrame:
    """Generate the pre-6PM synthetic domestic scouting dataset."""
    df = pd.DataFrame(PLAYER_DATA)

    # Add Season column for recency weighting
    df["Season"] = "2025-26"

    # Add Stage for knockout weighting
    np.random.seed(42)
    stages = ["League", "League", "League", "Quarterfinal", "Semifinal", "Final"]
    df["Stage"] = np.random.choice(stages, size=len(df), p=[0.5, 0.2, 0.1, 0.1, 0.05, 0.05])

    return df


def main():
    """Generate mock data, run the full pipeline, and export CSVs."""
    print("=" * 70)
    print("THE PINK CANVAS OS | SupeRR Selector 2.0")
    print("Generating Pre-6PM Synthetic Domestic Scouting Dataset...")
    print("=" * 70)

    # Step 1: Generate raw mock data
    df_raw = generate_mock_dataset()
    raw_path = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "mock_domestic_pack.csv")
    df_raw.to_csv(raw_path, index=False)
    print(f"\n✅ Raw dataset saved: {raw_path}")
    print(f"   → {len(df_raw)} players | {len(df_raw.columns)} columns")

    # Step 2: Run full analytics pipeline
    print("\n🔬 Running Anti-Overfitting Bayesian Pipeline...")
    df_processed = run_full_pipeline(df_raw)

    # Step 3: Export processed data
    processed_path = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "scouting_war_room.csv")
    df_processed.to_csv(processed_path, index=False)
    print(f"\n✅ Processed War Room dataset saved: {processed_path}")
    print(f"   → {len(df_processed)} players | {len(df_processed.columns)} columns")

    # Step 4: Summary
    print("\n" + "=" * 70)
    print("📊 PIPELINE SUMMARY")
    print("=" * 70)

    # Primary targets
    primaries = df_processed[df_processed["Is_Primary_Target"] == True]
    pivots = df_processed[df_processed["Is_Pivot_Clone"] == True]

    print(f"\n🎯 TOP 5 PRIMARY TARGETS:")
    for _, row in primaries.iterrows():
        anchor = row.get("IPL_Anchor_Name", "N/A")
        sim = row.get("Anchor_Similarity_Pct", 0)
        vfm = row.get("VfM_Score", 0)
        print(f"   {row['Player_Name']:25s} | {row['Role']:20s} | "
              f"{sim:.1f}% → {anchor} | VfM: {vfm:.1f}")

    print(f"\n🔄 5 PIVOT CLONES:")
    for _, row in pivots.iterrows():
        paired = row.get("Paired_With", "N/A")
        price = row.get("Expected_Price_Cr", 0)
        print(f"   {row['Player_Name']:25s} | Pivot for: {paired:20s} | ₹{price:.2f} Cr")

    print(f"\n📈 Bayesian Columns Added: EB_SR, EB_Death_SR, EB_Economy, EB_Death_Economy, "
          f"EB_Middle_Dot_Pct")
    print(f"📈 Venue Scores: Jaipur_Suitability_Score, Guwahati_Suitability_Score")
    print(f"📈 Impact Metrics: xRAA_per_100_balls, Composite_Bayesian_Impact, VfM_Score")
    print(f"📈 KNN Anchors: IPL_Anchor_Name, Anchor_Similarity_Pct")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()

