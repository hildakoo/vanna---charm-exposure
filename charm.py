import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

# ============================================================
# CONFIG
# ============================================================
symbol = "SPY"          
risk_free_rate = 0.05
exp_index = 0           
contract_size = 100     

# ============================================================
# FORMULE CHARM (GREEK ISOLÉ)
# ============================================================
def charm(S, K, T, r, sigma, option_type):
    if sigma is None or np.isnan(sigma) or sigma <= 0 or T <= 0:
        return 0

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    
    if option_type == "call":
        return -norm.pdf(d1) * (2*r*T - d2*sigma*np.sqrt(T)) / (2 * T * sigma * np.sqrt(T))
    else:  # put
        return  norm.pdf(d1) * (2*r*T - d2*sigma*np.sqrt(T)) / (2 * T * sigma * np.sqrt(T))


# ============================================================
# RÉCUPÉRATION DES OPTIONS
# ============================================================
ticker = yf.Ticker(symbol)
spot = ticker.history(period="1d")["Close"].iloc[-1]

expiration = ticker.options[exp_index]
chain = ticker.option_chain(expiration)
calls = chain.calls.copy()
puts = chain.puts.copy()

# Temps jusqu'à expiration
days_left = max((pd.to_datetime(expiration) - pd.Timestamp.today()).days, 1)
T = days_left / 365


# ============================================================
# CALCUL DU CHARM EXPOSURE
# ============================================================
for opt, opt_type in [(calls, "call"), (puts, "put")]:
    opt["CharmExposure"] = opt.apply(
        lambda row: charm(
            S=spot,
            K=row["strike"],
            T=T,
            r=risk_free_rate,
            sigma=row["impliedVolatility"],
            option_type=opt_type
        ) * row["openInterest"] * contract_size * spot,   # ✅ EXPOSURE
        axis=1
    )

# fusion des calls + puts
df_charm = pd.concat([calls[['strike', 'CharmExposure']],
                      puts[['strike', 'CharmExposure']]])


# ============================================================
# >>>>>>>  AFFICHAGE CONSOLE COMPLET  <<<<<<<<
# ============================================================
total_charm = df_charm["CharmExposure"].sum()

peak_charm_pos = df_charm[df_charm["CharmExposure"] == df_charm["CharmExposure"].max()].iloc[0]
peak_charm_neg = df_charm[df_charm["CharmExposure"] == df_charm["CharmExposure"].min()].iloc[0]

print("\n================= CHARM EXPOSURE =================")
print(f"🌡️  Charm Total (Dealer Exposure) : {total_charm:,.0f}")

print("\n--- PEAK CHARM (EXPOSURE PAR STRIKE) ---")
print(f"🟢 Peak Charm + (AIMANT / drift bull) : {peak_charm_pos['CharmExposure']:,.0f}  → Strike : {peak_charm_pos['strike']}")
print(f"🔴 Peak Charm - (MUR / drift bear)   : {peak_charm_neg['CharmExposure']:,.0f}  → Strike : {peak_charm_neg['strike']}")
print("===================================================\n")


# ============================================================
# GRAPH Charm per strike + zones aimant / mur
# ============================================================
plt.figure(figsize=(16,6))
plt.bar(df_charm["strike"], df_charm["CharmExposure"], color="tab:purple", alpha=0.6)

plt.axvline(float(peak_charm_pos['strike']), linestyle="--", linewidth=2, color="green",
            label=f"Peak Charm + (aimant) : {peak_charm_pos['strike']}")
plt.axvline(float(peak_charm_neg['strike']), linestyle="--", linewidth=2, color="red",
            label=f"Peak Charm - (mur) : {peak_charm_neg['strike']}")

plt.axhline(0, color="black")
plt.title(f"Charm Exposure par strike ({symbol}) — Expiration {expiration}")
plt.xlabel("Strike")
plt.ylabel("Charm Exposure (dealer delta drift)")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
plt.show()
