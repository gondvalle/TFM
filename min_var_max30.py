# ============================================================
#  Discretización binaria (B bits) con tope w_i ≤ δ = 0.30
#  + 4 gráficos: pesos, dispersión válida, histograma de energías,
#    convergencia top-10
# ============================================================

import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# D-Wave & dimod
from dwave.system import DWaveSampler, EmbeddingComposite
from dwave.samplers import SimulatedAnnealingSampler
from dimod import BinaryQuadraticModel

# ------------------------------------------------------------
#  1. Datos
# ------------------------------------------------------------
tickers = ["AAPL", "GOOGL", "JNJ", "JPM", "MSFT"]
prices  = yf.download(tickers, start="2020-01-01", end="2025-12-31", auto_adjust=False)["Adj Close"].dropna()
returns = prices.pct_change().dropna()
Σ       = returns.cov().values * 252      # matriz de covarianza anualizada

# ------------------------------------------------------------
#  2. Optimizador cuántico con límite de peso
# ------------------------------------------------------------
class DWaveBinaryCap:
    def __init__(self, cov, tickers, B=7, δ=0.30, λ=100):
        self.Σ        = cov
        self.tickers  = tickers
        self.n        = len(tickers)
        self.B        = B
        self.δ        = δ
        self.λ        = λ
        self.max_lvl  = 2**B - 1
        self.α        = [2**b / self.max_lvl for b in range(B)]
        self.var      = {(i,b): f"x_{i}_{b}" for i in range(self.n) for b in range(B)}

    def build_bqm(self):
        bqm = BinaryQuadraticModel('BINARY')
        # Varianza
        for i in range(self.n):
            for j in range(self.n):
                for bi in range(self.B):
                    for bj in range(self.B):
                        coeff = (self.δ**2)*self.α[bi]*self.α[bj]*self.Σ[i,j]
                        if coeff:
                            vi, vj = self.var[(i,bi)], self.var[(j,bj)]
                            if vi==vj: bqm.add_variable(vi, coeff)
                            else:      bqm.add_interaction(vi, vj, coeff)
        # Restricción suma=1
        for i in range(self.n):
            for j in range(self.n):
                for bi in range(self.B):
                    for bj in range(self.B):
                        coeff = self.λ*(self.δ**2)*self.α[bi]*self.α[bj]
                        vi, vj = self.var[(i,bi)], self.var[(j,bj)]
                        if vi==vj: bqm.add_variable(vi, coeff)
                        else:      bqm.add_interaction(vi, vj, coeff)
        for i in range(self.n):
            for b in range(self.B):
                bqm.add_variable(self.var[(i,b)], -2*self.λ*self.δ*self.α[b])
        bqm.add_offset(self.λ)
        return bqm

    def decode(self, sample):
        w = np.zeros(self.n)
        for i in range(self.n):
            acc = sum(sample.get(self.var[(i,b)],0)*2**b for b in range(self.B))
            w[i] = self.δ * acc / self.max_lvl
        return w

    def validate(self, sample, atol=1e-3):
        w = self.decode(sample)
        return not np.any(w>self.δ+1e-6) and abs(w.sum()-1)<atol

    def variance(self, w):
        return float(w @ self.Σ @ w)

# ------------------------------------------------------------
#  3. Construcción y muestreo
# ------------------------------------------------------------
opt    = DWaveBinaryCap(Σ, tickers, B=7, δ=0.30, λ=100)
bqm    = opt.build_bqm()
USE_QPU = False

if USE_QPU:
    sampler   = EmbeddingComposite(DWaveSampler())
    sampleset = sampler.sample(bqm, num_reads=1000, label="MV-Bin-Cap30")
else:
    sampler   = SimulatedAnnealingSampler()
    sampleset = sampler.sample(bqm, num_reads=5000, seed=42)

# ------------------------------------------------------------
#  4. Procesado de soluciones
# ------------------------------------------------------------
valid_sols = []
all_energies = []
for sample, E, occ in sampleset.data(['sample','energy','num_occurrences']):
    all_energies.append(E)
    if not opt.validate(sample):
        continue
    w   = opt.decode(sample)
    var = opt.variance(w)
    valid_sols.append((w, var, E, occ))

# Mejor solución válida
best = min(valid_sols, key=lambda x: x[1], default=(None,np.inf,None,None))
best_w, best_var, best_E, _ = best

# ------------------------------------------------------------
#  5. Comparación clásica (opcional)
# ------------------------------------------------------------
try:
    from skfolio import RiskMeasure
    from skfolio.optimization import MeanRisk

    model = MeanRisk(risk_measure=RiskMeasure.VARIANCE,
                     max_weights=0.30, min_weights=0.0)
    model.fit(returns)
    port = model.predict(returns)
    w_cl   = port.weights
    var_cl = float(w_cl @ Σ @ w_cl)
except ImportError:
    w_cl, var_cl = None, None

# ------------------------------------------------------------
#  6. Visualización 2×2
# ------------------------------------------------------------
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=(14,10))

# 6.1 Comparación de pesos
x = np.arange(len(tickers))
width = 0.35
ax1.bar(x-width/2, best_w, width, label='Cuántico', alpha=0.8, color='#2E8B57')
if w_cl is not None:
    ax1.bar(x+width/2, w_cl,   width, label='Clásico', alpha=0.8, color='#FF6347')
ax1.set_xticks(x); ax1.set_xticklabels(tickers)
ax1.set_ylabel("Peso"); ax1.set_title("Pesos óptimos")
ax1.legend(); ax1.grid(alpha=0.3)

# 6.2 Dispersión varianza vs energía
vars_ = [v for _,v,_,_ in valid_sols]
E_    = [e for _,_,e,_ in valid_sols]
occ_  = [o for _,_,_,o in valid_sols]
scatter = ax2.scatter(vars_, E_, c=occ_, s=50, alpha=0.7, cmap='viridis')
ax2.axvline(best_var, color='red',   linestyle='--', label=f'Quantum: {best_var:.6f}')
if var_cl is not None:
    ax2.axvline(var_cl, color='blue', linestyle='--', label=f'Clásico: {var_cl:.6f}')
ax2.set_xlabel("Varianza"); ax2.set_ylabel("Energía")
ax2.set_title(f"Soluciones válidas ({len(valid_sols)})")
ax2.legend(); ax2.grid(alpha=0.3)
plt.colorbar(scatter, ax=ax2, label="Ocurrencias")

# 6.3 Histograma de energías
ax3.hist(all_energies, bins=50, alpha=0.7, edgecolor='black')
ax3.axvline(best_E, color='green', linestyle='--', label=f'Mín Energy: {best_E:.3f}')
ax3.set_xlabel("Energía"); ax3.set_ylabel("Frecuencia")
ax3.set_title("Distribución de energías")
ax3.legend(); ax3.grid(alpha=0.3)

# 6.4 Convergencia top-10 varianzas
if len(valid_sols) >= 10:
    top10 = sorted(valid_sols, key=lambda x: x[1])[:10]
    ax4.plot(range(1,11), [v for _,v,_,_ in top10], 'o-', alpha=0.7)
    if var_cl is not None:
        ax4.axhline(var_cl, color='blue', linestyle='--', label=f'Clásico: {var_cl:.6f}')
    ax4.set_xlabel("Rango"); ax4.set_ylabel("Varianza")
    ax4.set_title("Top 10 soluciones por varianza")
    ax4.legend(); ax4.grid(alpha=0.3)
else:
    ax4.text(0.5,0.5,f"{len(valid_sols)} válidas\nNecesitas ≥10", ha='center', va='center')
    ax4.set_title("Convergencia")

plt.tight_layout()
plt.show()
