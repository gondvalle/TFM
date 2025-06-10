import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# D-Wave imports
from dwave.system import DWaveSampler, EmbeddingComposite
from dwave.samplers import SimulatedAnnealingSampler
import dimod
from dimod import BinaryQuadraticModel

# Download and prepare data
tickers = ["AAPL", "GOOGL", "JPM", "MSFT"]
stocks = yf.download(tickers, start='2020-01-01', end='2025-12-31', auto_adjust=False)
stocks_adj_stocks = stocks["Adj Close"].dropna()
returns = stocks_adj_stocks.pct_change().dropna()

# Annualized covariance matrix
cov_matrix = returns.cov() * 252
print("Covariance Matrix:")
print(cov_matrix)

class DWavePortfolioOptimizer:
    def __init__(self, cov_matrix, tickers, K=2, lambda_penalty=50, gamma_penalty=100):
        """
        D-Wave Quantum Annealing Portfolio Optimizer
        
        Parameters:
        - cov_matrix: Covariance matrix of assets
        - tickers: List of asset names
        - K: Number of discretization levels (weights will be 0, 1/K, 2/K, ..., 1)
        - lambda_penalty: Penalty for sum of weights constraint
        - gamma_penalty: Penalty for one-hot encoding constraint
        """
        self.cov_matrix = cov_matrix.values
        self.tickers = tickers
        self.n_assets = len(tickers)
        self.K = K
        self.lambda_penalty = lambda_penalty
        self.gamma_penalty = gamma_penalty
        self.n_qubits = self.n_assets * (K + 1)
        
        # Create mapping from (asset, level) to variable name
        self.var_map = {}
        self.reverse_var_map = {}
        
        for i in range(self.n_assets):
            for p in range(K + 1):
                var_name = f"x_{i}_{p}"
                self.var_map[(i, p)] = var_name
                self.reverse_var_map[var_name] = (i, p)
        
        print(f"D-Wave Problem Setup:")
        print(f"- Assets: {self.n_assets}")
        print(f"- Discretization levels: {K + 1} (0, 1/{K}, 2/{K}, ..., 1)")
        print(f"- Total variables: {self.n_qubits}")
        print(f"- Lambda penalty: {lambda_penalty}")
        print(f"- Gamma penalty: {gamma_penalty}")
        
    def create_bqm(self):
        """Create Binary Quadratic Model for D-Wave"""
        # Initialize BQM
        bqm = BinaryQuadraticModel('BINARY')
        
        # 1. VARIANCE TERM
        print("Adding variance terms...")
        for i in range(self.n_assets):
            for j in range(self.n_assets):
                for p in range(self.K + 1):
                    for l in range(self.K + 1):
                        coeff = (p * l * self.cov_matrix[i, j]) / (self.K ** 2)
                        if coeff != 0:
                            var_i = self.var_map[(i, p)]
                            var_j = self.var_map[(j, l)]
                            
                            if var_i == var_j:
                                bqm.add_variable(var_i, coeff)
                            else:
                                bqm.add_interaction(var_i, var_j, coeff)
        
        # 2. WEIGHT SUM CONSTRAINT: λ * (Σ(p/K * x_{i,p}) - 1)²
        print("Adding weight sum constraint...")
        
        # Quadratic terms
        for i in range(self.n_assets):
            for p in range(self.K + 1):
                for j in range(self.n_assets):
                    for l in range(self.K + 1):
                        coeff = self.lambda_penalty * (p * l) / (self.K ** 2)
                        if coeff != 0:
                            var_i = self.var_map[(i, p)]
                            var_j = self.var_map[(j, l)]
                            
                            if var_i == var_j:
                                bqm.add_variable(var_i, coeff)
                            else:
                                bqm.add_interaction(var_i, var_j, coeff)
        
        # Linear terms
        for i in range(self.n_assets):
            for p in range(self.K + 1):
                coeff = -2 * self.lambda_penalty * p / self.K
                if coeff != 0:
                    var = self.var_map[(i, p)]
                    bqm.add_variable(var, coeff)
        
        # 3. ONE-HOT CONSTRAINT: γ * Σ_i (Σ_p x_{i,p} - 1)²
        print("Adding one-hot constraints...")
        
        for i in range(self.n_assets):
            # Quadratic terms within each asset
            for p in range(self.K + 1):
                for l in range(self.K + 1):
                    coeff = self.gamma_penalty
                    var_p = self.var_map[(i, p)]
                    var_l = self.var_map[(i, l)]
                    
                    if var_p == var_l:
                        bqm.add_variable(var_p, coeff)
                    else:
                        bqm.add_interaction(var_p, var_l, coeff)
            
            # Linear terms
            for p in range(self.K + 1):
                coeff = -2 * self.gamma_penalty
                var = self.var_map[(i, p)]
                bqm.add_variable(var, coeff)
        
        # Add constant offset
        offset_const = self.lambda_penalty + self.n_assets * self.gamma_penalty
        bqm.add_offset(offset_const)
        
        print(f"BQM created with {len(bqm.variables)} variables and {len(bqm.quadratic)} interactions")
        return bqm
    
    def decode_solution(self, sample_dict):
        """Decode D-Wave solution to portfolio weights"""
        weights = np.zeros(self.n_assets)
        
        for i in range(self.n_assets):
            for p in range(self.K + 1):
                var_name = self.var_map[(i, p)]
                if sample_dict.get(var_name, 0) == 1:
                    weights[i] = p / self.K
                    break
        
        return weights
    
    def validate_solution(self, sample_dict):
        """Check if solution satisfies constraints"""
        # Check one-hot constraint for each asset
        for i in range(self.n_assets):
            count = 0
            for p in range(self.K + 1):
                var_name = self.var_map[(i, p)]
                count += sample_dict.get(var_name, 0)
            
            if count != 1:
                return False, f"Asset {i} has {count} selections instead of 1"
        
        weights = self.decode_solution(sample_dict)
        weight_sum = np.sum(weights)
        
        # Check weight sum constraint
        if abs(weight_sum - 1.0) > 0.1:
            return False, f"Weight sum {weight_sum:.3f} != 1.0 (violates constraint)"
        
        return True, f"Valid solution with weight sum: {weight_sum:.3f}"
    
    def calculate_portfolio_variance(self, weights):
        """Calculate portfolio variance"""
        return np.dot(weights.T, np.dot(self.cov_matrix, weights))
    
    def analyze_solution_energy(self, sample_dict, bqm):
        """Analyze energy components of a solution"""
        weights = self.decode_solution(sample_dict)
        
        # Calculate energy using BQM
        energy = bqm.energy(sample_dict)
        
        # Manual calculation for verification
        variance = self.calculate_portfolio_variance(weights)
        weight_sum = np.sum(weights)
        weight_penalty = self.lambda_penalty * (weight_sum - 1.0)**2
        
        onehot_penalty = 0
        for i in range(self.n_assets):
            asset_sum = 0
            for p in range(self.K + 1):
                var_name = self.var_map[(i, p)]
                asset_sum += sample_dict.get(var_name, 0)
            onehot_penalty += self.gamma_penalty * (asset_sum - 1)**2
        
        manual_energy = variance + weight_penalty + onehot_penalty + (self.lambda_penalty + self.n_assets * self.gamma_penalty)
        
        print(f"Energy Analysis:")
        print(f"  BQM Energy: {energy:.6f}")
        print(f"  Manual Energy: {manual_energy:.6f}")
        print(f"  Variance: {variance:.6f}")
        print(f"  Weight penalty: {weight_penalty:.6f}")
        print(f"  One-hot penalty: {onehot_penalty:.6f}")
        print(f"  Weights: {weights}")
        print(f"  Weight sum: {weight_sum:.3f}")
        
        return energy

# Initialize D-Wave optimizer with higher penalties
optimizer = DWavePortfolioOptimizer(
    cov_matrix, 
    tickers, 
    K=50, 
    lambda_penalty=50,    # Moderate penalty for D-Wave
    gamma_penalty=100      # Higher penalty for one-hot constraint
)

# Create Binary Quadratic Model
bqm = optimizer.create_bqm()

print(f"\nBQM Statistics:")
print(f"Variables: {len(bqm.variables)}")
print(f"Interactions: {len(bqm.quadratic)}")
print(f"Offset: {bqm.offset}")

# Choose sampler (use SimulatedAnnealingSampler if no D-Wave access)
USE_DWAVE = False  # Set to True if you have D-Wave access

if USE_DWAVE:
    print("\n=== Using D-Wave Quantum Annealer ===")
    try:
        # Real D-Wave quantum annealer
        sampler = EmbeddingComposite(DWaveSampler())
        
        # Sample from D-Wave
        num_reads = 1000
        sampleset = sampler.sample(bqm, num_reads=num_reads, 
                                 label='Portfolio Optimization')
        
    except Exception as e:
        print(f"D-Wave connection failed: {e}")
        print("Falling back to Simulated Annealing...")
        USE_DWAVE = False

if not USE_DWAVE:
    print("\n=== Using Simulated Annealing ===")
    sampler = SimulatedAnnealingSampler()
    
    # Sample using simulated annealing
    num_reads = 5000
    sampleset = sampler.sample(bqm, num_reads=num_reads, 
                             seed=42)  # For reproducibility

print(f"Sampling completed with {len(sampleset)} samples")

# Process results
valid_solutions = []
all_solutions = []
best_energy = float('inf')
best_sample = None
best_valid_sample = None
best_valid_variance = float('inf')

print(f"\nProcessing {len(sampleset)} samples...")

for sample, energy, num_occurrences in sampleset.data(['sample', 'energy', 'num_occurrences']):
    all_solutions.append((sample, energy, num_occurrences))
    
    # Track best energy solution
    if energy < best_energy:
        best_energy = energy
        best_sample = sample
    
    # Check validity
    is_valid, msg = optimizer.validate_solution(sample)
    
    if is_valid:
        weights = optimizer.decode_solution(sample)
        variance = optimizer.calculate_portfolio_variance(weights)
        valid_solutions.append((sample, weights, variance, energy, num_occurrences))
        
        # Track best valid solution by variance
        if variance < best_valid_variance:
            best_valid_variance = variance
            best_valid_sample = sample

print(f"\nResults Summary:")
print(f"Total samples: {len(sampleset)}")
print(f"Valid solutions: {len(valid_solutions)}")
print(f"Success rate: {len(valid_solutions)/len(sampleset)*100:.2f}%")

# Analyze best solution by energy
if best_sample is not None:
    print(f"\n=== BEST SOLUTION BY ENERGY ===")
    print(f"Energy: {best_energy:.6f}")
    optimizer.analyze_solution_energy(best_sample, bqm)
    
    is_valid, msg = optimizer.validate_solution(best_sample)
    print(f"Validity: {is_valid} - {msg}")

# Analyze best valid solution
if best_valid_sample is not None:
    print(f"\n=== BEST VALID SOLUTION ===")
    best_valid_weights = optimizer.decode_solution(best_valid_sample)
    print(f"Weights: {best_valid_weights}")
    print(f"Portfolio variance: {best_valid_variance:.6f}")
    print(f"Sum of weights: {np.sum(best_valid_weights):.3f}")
    
    # Energy analysis
    valid_energy = None
    for sample, weights, variance, energy, occurrences in valid_solutions:
        if np.array_equal(weights, best_valid_weights):
            valid_energy = energy
            print(f"Solution energy: {energy:.6f}")
            print(f"Occurrences: {occurrences}")
            break
    
    # Compare with classical solution
    try:
        from skfolio import RiskMeasure
        from skfolio.optimization import MeanRisk

        model = MeanRisk(risk_measure=RiskMeasure.VARIANCE)
        model.fit(returns)
        portfolio = model.predict(returns)
        classical_weights = portfolio.weights
        classical_variance = np.dot(classical_weights.T, np.dot(cov_matrix.values, classical_weights))

        print(f"\n=== CLASSICAL COMPARISON ===")
        print(f"Classical weights: {classical_weights}")
        print(f"Classical variance: {classical_variance:.6f}")
        print(f"Quantum variance: {best_valid_variance:.6f}")
        print(f"Quantum/Classical variance ratio: {best_valid_variance/classical_variance:.3f}")
        
        # Create visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Portfolio weights comparison
        x = np.arange(len(tickers))
        width = 0.35
        ax1.bar(x - width/2, best_valid_weights, width, label='D-Wave Quantum', alpha=0.8, color='#2E8B57')
        ax1.bar(x + width/2, classical_weights, width, label='Classical', alpha=0.8, color='#FF6347')
        ax1.set_xlabel('Assets')
        ax1.set_ylabel('Weight')
        ax1.set_title('Portfolio Weights: D-Wave vs Classical')
        ax1.set_xticks(x)
        ax1.set_xticklabels(tickers)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Valid solutions distribution
        if len(valid_solutions) > 1:
            variances = [sol[2] for sol in valid_solutions]
            energies = [sol[3] for sol in valid_solutions]
            occurrences = [sol[4] for sol in valid_solutions]
            
            scatter = ax2.scatter(variances, energies, c=occurrences, s=50, alpha=0.7, cmap='viridis')
            ax2.axvline(best_valid_variance, color='red', linestyle='--', alpha=0.7,
                       label=f'Best Quantum: {best_valid_variance:.6f}')
            ax2.axvline(classical_variance, color='blue', linestyle='--', alpha=0.7,
                       label=f'Classical: {classical_variance:.6f}')
            ax2.set_xlabel('Portfolio Variance')
            ax2.set_ylabel('Energy')
            ax2.set_title('Valid Solutions: Variance vs Energy')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=ax2, label='Occurrences')
        
        # Energy histogram
        all_energies = [sol[1] for sol in all_solutions]
        ax3.hist(all_energies, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        if valid_energy:
            ax3.axvline(valid_energy, color='red', linestyle='--', 
                       label=f'Best Valid Energy: {valid_energy:.3f}')
        ax3.axvline(best_energy, color='green', linestyle='--',
                   label=f'Global Best Energy: {best_energy:.3f}')
        ax3.set_xlabel('Energy')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Energy Distribution of All Solutions')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Convergence analysis (if multiple valid solutions)
        if len(valid_solutions) > 5:
            sorted_valid = sorted(valid_solutions, key=lambda x: x[2])  # Sort by variance
            top_variances = [sol[2] for sol in sorted_valid[:10]]
            ax4.plot(range(len(top_variances)), top_variances, 'o-', color='purple', alpha=0.7)
            ax4.axhline(classical_variance, color='blue', linestyle='--', alpha=0.7,
                       label=f'Classical: {classical_variance:.6f}')
            ax4.set_xlabel('Solution Rank')
            ax4.set_ylabel('Portfolio Variance')
            ax4.set_title('Top 10 Quantum Solutions by Variance')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, f'Found {len(valid_solutions)} valid solutions\nNeed more for convergence analysis',
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Convergence Analysis')
        
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        print("skfolio not available for classical comparison")
        
        # Simple D-Wave only visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Portfolio weights
        ax1.bar(range(len(tickers)), best_valid_weights, alpha=0.8, color='#2E8B57')
        ax1.set_xlabel('Assets')
        ax1.set_ylabel('Weight')
        ax1.set_title('D-Wave Quantum Portfolio Weights')
        ax1.set_xticks(range(len(tickers)))
        ax1.set_xticklabels(tickers)
        ax1.grid(True, alpha=0.3)
        
        # Energy distribution
        all_energies = [sol[1] for sol in all_solutions]
        ax2.hist(all_energies, bins=30, alpha=0.7, color='skyblue')
        ax2.axvline(best_energy, color='red', linestyle='--', label=f'Best Energy: {best_energy:.3f}')
        ax2.set_xlabel('Energy')
        ax2.set_ylabel('Frequency')
        ax2.set_title('D-Wave Solution Energy Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

# else:
    print("\n=== NO VALID SOLUTIONS FOUND ===")
    print("Recommendations:")
    print("1. Increase penalty parameters (lambda_penalty, gamma_penalty)")
    print("2. Try different discretization levels (K)")
    print("3. Increase number of reads")
    print("4. Check problem formulation")
    
    # Show some invalid solutions for debugging
    print(f"\nSample invalid solutions:")
    for i, (sample, energy, occurrences) in enumerate(all_solutions[:5]):
        is_valid, msg = optimizer.validate_solution(sample)
        weights = optimizer.decode_solution(sample)
        print(f"Solution {i+1}: Energy={energy:.3f}, Occurrences={occurrences}")
        print(f"  Weights: {weights}")
        print(f"  Sum: {np.sum(weights):.3f}")
        print(f"  Valid: {is_valid} - {msg}")

# Final summary
print(f"\n=== FINAL SUMMARY ===")
if USE_DWAVE:
    print("Used: D-Wave Quantum Annealer")
else:
    print("Used: Simulated Annealing")

print(f"Problem size: {len(bqm.variables)} variables, {len(bqm.quadratic)} interactions")
print(f"Samples generated: {len(sampleset)}")
print(f"Valid solutions: {len(valid_solutions)} ({len(valid_solutions)/len(sampleset)*100:.1f}%)")

if valid_solutions:
    print(f"Best quantum variance: {best_valid_variance:.6f}")
    try:
        print(f"Quantum/Classical ratio: {best_valid_variance/classical_variance:.3f}")
    except:
        pass
