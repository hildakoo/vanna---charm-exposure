import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
from math import log, sqrt
import matplotlib.pyplot as plt

# -----------------------------
# 1️⃣ Paramètres
# -----------------------------
symbol = "SPY"
risk_free = 0.05
exp_index = 0  # prochaine expiration (0 = la plus proche)

# -----------------------------
# 2️⃣ Récupérer Spot & Options Chain
# -----------------------------
ticker = yf.Ticker(symbol)
spot = ticker.history(period='1d')['Close'].iloc[-1]
exp_date = ticker.options[exp_index]

options_chain = ticker.option_chain(exp_date)

calls = options_chain.calls.dropna(subset=['impliedVolatility', 'openInterest'])
puts  = options_chain.puts.dropna(subset=['impliedVolatility', 'openInterest'])

# -----------------------------
# 3️⃣ Fonction VANNA (greek isolé)
# -----------------------------
def vanna(S, K, T, r, sigma):
    if sigma is None or np.isnan(sigma) or sigma <= 0 or T <= 0:
        return 0
    d1 = (log(S/K) + (r + 0.5*sigma**2)*T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    vega = S * norm.pdf(d1) * sqrt(T)
    return vega * d2 / sigma  # ✅ VANNA isolée

# -----------------------------
# 4️⃣ Temps jusqu'à expiration
# -----------------------------
T = max((pd.to_datetime(exp_date) - pd.Timestamp.today()).days / 365, 1/365)

SCALE = 1000  # pour lisibilité sur le graph

for df in [calls, puts]:
    df['Vanna'] = df.apply(lambda row: vanna(spot, row['strike'], T, risk_free, row['impliedVolatility']), axis=1)
    df['VEX']   = df['Vanna'] * df['openInterest'] * SCALE  # ✅ Vanna Exposure (dealer hedging)

# 📊 Total VEX des calls + puts
VEX_total = calls['VEX'].sum() + puts['VEX'].sum()
print(f"\nTotal VEX pour {exp_date} : {VEX_total:,.0f}")

# -----------------------------
# 6️⃣ Peak VEX + et Peak VEX - (FIX : aucune erreur possible)
# -----------------------------
combined_vex = pd.concat([calls[['strike', 'VEX']], puts[['strike', 'VEX']]])

peak_vex_pos = combined_vex[combined_vex['VEX'] == combined_vex['VEX'].max()].iloc[0]
peak_vex_neg = combined_vex[combined_vex['VEX'] == combined_vex['VEX'].min()].iloc[0]

print("\n━━ PEAK VEX (EXPOSURE DEALER) ━━")
print(f"🟢 Peak VEX + (magnet) : {peak_vex_pos['VEX']:,.0f}  sur strike {peak_vex_pos['strike']}")
print(f"🔴 Peak VEX - (wall)   : {peak_vex_neg['VEX']:,.0f}  sur strike {peak_vex_neg['strike']}")

# -----------------------------
# 7️⃣ Peak VANNA (greek isolé pour info)
# -----------------------------
combined_greek = pd.concat([calls[['strike', 'Vanna']], puts[['strike', 'Vanna']]])

max_vanna = combined_greek[combined_greek['Vanna'] == combined_greek['Vanna'].max()].iloc[0]
min_vanna = combined_greek[combined_greek['Vanna'] == combined_greek['Vanna'].min()].iloc[0]

print("\n━━ VANNA (GREEK ISOLÉ, PAS EXPOSURE) ━━")
print(f"📈 Max Vanna (greek)  : {float(max_vanna['Vanna']):.6f} sur strike {float(max_vanna['strike'])}")
print(f"📉 Min Vanna (greek)  : {float(min_vanna['Vanna']):.6f} sur strike {float(min_vanna['strike'])}")

# -----------------------------
# 8️⃣ Graphique VEX
# -----------------------------
plt.figure(figsize=(13,5))
plt.bar(calls['strike'], calls['VEX'], alpha=0.6, label='Calls (VEX contrib)', color='tab:blue')
plt.bar(puts['strike'], puts['VEX'], alpha=0.6, label='Puts (VEX contrib)', color='tab:orange')

plt.axvline(float(peak_vex_pos['strike']), linestyle='--', linewidth=1.8, color='green',
            label='Peak VEX + (AIMANT / Bullish)')

plt.axvline(float(peak_vex_neg['strike']), linestyle='--', linewidth=1.8, color='red',
            label='Peak VEX - (MUR / Bearish)')

plt.xlabel('Strike')
plt.ylabel(f'VEX (x{SCALE})')
plt.title(f'Vanna Exposure (VEX) — {symbol} — Expiration {exp_date}')
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
plt.show()
