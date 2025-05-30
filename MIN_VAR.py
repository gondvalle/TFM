import pandas as pd
import numpy as np
import yfinance as yf

from pyqubo import Array, Constraint

import quantagonia
from quantagonia.qubo import QuboModel
from quantagonia.enums import HybridSolverOptSenses
from quantagonia import HybridSolver, HybridSolverParameters

tickers = ["AAPL", "GOOGL", "JNJ", "JPM", "MSFT"] 
start_date = "2020-01-01"
end_date = "2025-12-31"

print(f"Downloading data for: {tickers}")
try:
    stocks = yf.download(tickers, start=start_date, end=end_date, auto_adjust=False)
    if stocks.empty:
        raise ValueError("No data downloaded. Check tickers and date range.")
except Exception as e:
    print(f"Error downloading stock data: {e}. Creating dummy data instead.")

# --- PASO 1: Carga y preparación de datos ---
stocks_adj_close = stocks["Adj Close"]
stocks_adj_close = stocks_adj_close.dropna()
returns = stocks_adj_close[tickers].pct_change().dropna()

# --- PASO 2: Calcular la matriz de covarianza ---
cov_matrix = returns.cov()
cov_matrix_np = cov_matrix.values # Usar como numpy array para pyqubo

# --- PASO 3: Formulación del QUBO ---
num_assets = len(tickers)

# K+1 niveles discretos para los pesos w_i in {0, 1/K, 2/K, ..., K/K}
# Por ejemplo, K=4 significa que los pesos pueden ser 0.0, 0.25, ..., 1.0
K = 4 

# Variables binarias x_i_k
# x[i][k] = 1 si el activo 'i' tiene el peso k/K, y 0 en caso contrario.
# La dimensión es num_assets x (K+1)
x = Array.create('x', shape=(num_assets, K + 1), vartype='BINARY')

# Pesos w_i en términos de las variables binarias x_i_k
# w_i = sum_{k=0}^{K} (k/K) * x_i_k
weights = [sum((k / K) * x[i, k] for k in range(K + 1)) for i in range(num_assets)]

# 1. Varianza del portafolio: w^T * Sigma * w
portfolio_variance = 0
for i in range(num_assets):
    for j in range(num_assets):
        portfolio_variance += weights[i] * cov_matrix_np[i, j] * weights[j]

# Coeficientes de penalización (deben ser ajustados cuidadosamente)
# Estos valores deben ser lo suficientemente grandes para asegurar que las restricciones se cumplan.
# Un punto de partida común es que sean de magnitud similar o mayor que los coeficientes
# de la función objetivo principal (la varianza).
lambda_penalty = 150.0 # Penalización para sum(w_i) = 1
mu_penalty = 150.0     # Penalización para que cada activo elija un solo nivel de peso

# 2. Restricción de la suma de pesos: sum(w_i) = 1
# Penalización: lambda * (sum(w_i) - 1)^2
total_selected_weight = sum(weights[i] for i in range(num_assets))
sum_weights_constraint = (total_selected_weight - 1)**2

# 3. Restricción de selección única de nivel de peso para cada activo: sum_k(x_i_k) = 1 para cada i
# Penalización: mu * sum_i (sum_k(x_i_k) - 1)^2
one_level_constraint = 0
for i in range(num_assets):
    one_level_constraint += (sum(x[i, k] for k in range(K + 1)) - 1)**2

# Hamiltoniano (Función Objetivo H a minimizar)
H = portfolio_variance + lambda_penalty * sum_weights_constraint + mu_penalty * one_level_constraint

# Compilar el Hamiltoniano a un modelo QUBO
model_pyqubo = H.compile() 
pyqubo_dict, pyqubo_offset = model_pyqubo.to_qubo()

print(f"PyQUBO generated dictionary with {len(pyqubo_dict)} terms.")
print(f"PyQUBO offset: {pyqubo_offset}")

# --- Store the mapping from PyQUBO variable label to (asset_index, level_k) ---
pyqubo_label_to_coords = {}
for i in range(num_assets):
    for k in range(K + 1):
        label = f'x[{i}][{k}]'
        pyqubo_label_to_coords[label] = (i, k)


# --- PASO 4: Configuración y Resolución con Quantagonia ---
API_KEY = "API_KEY" 

# Crear una instancia del modelo QUBO de Quantagonia
q_model = QuboModel()

# Diccionario para mantener un registro de las variables de Quantagonia
quantagonia_vars = {}

# Añadir variables y términos del objetivo al modelo de Quantagonia
for (var_label_i, var_label_j), coeff in pyqubo_dict.items():
    # Asegurar que las variables existen en el modelo de Quantagonia
    if var_label_i not in quantagonia_vars:
        quantagonia_vars[var_label_i] = q_model.add_variable(name=var_label_i) # PyQUBO labels son strings
    if var_label_j not in quantagonia_vars:
        quantagonia_vars[var_label_j] = q_model.add_variable(name=var_label_j)

    q_var_i = quantagonia_vars[var_label_i]
    q_var_j = quantagonia_vars[var_label_j]

    if var_label_i == var_label_j: # Término lineal Q_ii * x_i (ya que x_i^2 = x_i para binarias)
        q_model.objective += coeff * q_var_i
    else: # Término cuadrático Q_ij * x_i * x_j
        q_model.objective += coeff * q_var_i * q_var_j

# Establecer el sentido de la optimización (minimizar para la varianza del portafolio)
q_model.sense = HybridSolverOptSenses.MINIMIZE

# Configurar el solver híbrido de Quantagonia
hybrid_solver = HybridSolver(API_KEY)

# Establecer parámetros del solver (opcional, pero recomendado)
params = HybridSolverParameters()
params.set_time_limit(30) # Establecer el límite de tiempo en segundos

print("\nEnviando el problema al solver de Quantagonia...")

# Resolver el QUBO
try:
    # MODIFICACIÓN CLAVE AQUÍ: The solve() method returns a solution object.
    # We need to access its 'objective' attribute.
    solution_result = q_model.solve(hybrid_solver, params)
    solution_obj_val = solution_result['objective'] # Access the objective from the solution object

    # El valor real del Hamiltoniano H es solution_obj_val + pyqubo_offset
    hamiltonian_value = solution_obj_val + pyqubo_offset

    print(f"\n--- Resultados del Solver de Quantagonia ---")
    print(f"Valor objetivo del QUBO (devuelto por Quantagonia): {solution_obj_val}")
    print(f"Offset de PyQUBO: {pyqubo_offset}")
    print(f"Valor total del Hamiltoniano (H): {hamiltonian_value}")

    # --- PASO 5: Interpretación y Presentación de los Resultados ---
    # Definimos una función para obtener el índice a partir de la etiqueta 'x[i][j]'
    def get_index(label):
        # Extraemos i y j de la cadena 'x[i][j]'
        i = int(label.split('[')[1].split(']')[0])  # Obtiene el número entre el primer []
        j = int(label.split('[')[2].split(']')[0])  # Obtiene el número entre el segundo []
        return i * num_assets + j

    # Interpretación y presentación de los resultados
    solved_variables_dict = {}
    print("\nValores de las variables binarias de la solución (variables que son '1'):")
    for var_label, q_var_instance in quantagonia_vars.items():
        # Calculamos el índice de la variable y obtenemos su valor de la solución
        index = get_index(var_label)
        value = solution_result['solution'][str(index)]
        solved_variables_dict[var_label] = value
        if value == 1:
            print(f"  {var_label}: {value}")

    if not solved_variables_dict:
        print("ADVERTENCIA: No se pudieron recuperar los valores de las variables de la solución de Quantagonia.")
        print("La decodificación de pesos será incorrecta.")
        for i in range(num_assets):
            label = f'x[{i}][0]'
            solved_variables_dict[label] = 1
            for k_dummy in range(1, K + 1):
                label_dummy = f'x[{i}][{k_dummy}]'
                solved_variables_dict[label_dummy] = 0

    print("\n--- ¡Aquí tienes el portafolio optimizado! ---")
    print("-" * 40)

    final_weights = np.zeros(num_assets)
    portfolio_summary = []

    for i in range(num_assets):
        selected_k = -1
        for k in range(K + 1):
            var_label = f'x[{i}][{k}]'
            if solved_variables_dict.get(var_label, 0) == 1:
                final_weights[i] = k / K
                selected_k = k
                break

        if selected_k != -1:
            portfolio_summary.append({
                "Empresa": tickers[i],
                "Peso en el Portafolio": f"{final_weights[i]:.4f}",
                "Nivel de Discretización": f"{selected_k}/{K}"
            })
        else:
            portfolio_summary.append({
                "Empresa": tickers[i],
                "Peso en el Portafolio": "0.0000",
                "Nivel de Discretización": "Ninguno (error o 0)"
            })
            print(f"ADVERTENCIA: No se asignó ningún nivel de peso válido para {tickers[i]}. Puede que una de las restricciones no se haya cumplido perfectamente o el solver no encontró una solución óptima para esta variable.")

    portfolio_df = pd.DataFrame(portfolio_summary)
    print(portfolio_df.to_string(index=False))

    print("-" * 40)
    print(f"\nSuma total de los pesos asignados: {np.sum(final_weights):.4f} (debería ser cercano a 1.0)")

except Exception as e:
    print(f"\nOcurrió un error durante la ejecución con Quantagonia: {e}")
