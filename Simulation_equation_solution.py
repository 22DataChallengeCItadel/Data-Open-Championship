import numpy as np
import pandas as pd
from extrapolate_production_data import ProductionData
from Carbon_tariffs import TariffData
from sklearn.model_selection import ParameterGrid
import seaborn as sns

# Import data
## Trade data
trade_flows = pd.read_csv("Data/trade_flow_data.csv")

## Carbon tariff data
R = TariffData().get_mean_price()

## Emission data
C_DE = ProductionData().get_gwp(product_type="Plastic").to_list()[0]
C1_DE = ProductionData().get_gwp(product_type="Alternative").to_list()[0]

# Baseline model
## Six unknowns: P_EXP, P1_EXP, P_DOM, P1_DOM, Q, Q1
## Eight parameters: P_US, P1_US, Q_US, Q1_US, R, C_US, C1_US, e

# We use Trade flows from Germany to the US
Q_DE = trade_flows.loc[
    (trade_flows["Country"] == "Germany")
    & (trade_flows["Year"] == 2020)
    & (trade_flows["product"] == "plastic bag"),
    "qty_thousands",
].to_list()[0]

Q_ALT_DE = trade_flows.loc[
    (trade_flows["Country"] == "Germany")
    & (trade_flows["Year"] == 2020)
    & (trade_flows["product"].isin(["textile bag", "paper bag"])),
    "qty_thousands",
]

P_DE = trade_flows.loc[
    (trade_flows["Country"] == "Germany")
    & (trade_flows["Year"] == 2020)
    & (trade_flows["product"] == "plastic bag"),
    "price_per_thousands",
].to_list()[0]

P_ALT_DE = trade_flows.loc[
    (trade_flows["Country"] == "Germany")
    & (trade_flows["Year"] == 2020)
    & (trade_flows["product"].isin(["textile bag", "paper bag"])),
    "price_per_thousands",
]

P1_DE = np.average(P_ALT_DE, weights=Q_ALT_DE)
Q1_DE = Q_ALT_DE.sum()

P_DE = 0.01 * 1000
P1_DE = 0.02 * 1000

# Set elasticity parameters

param_grid = {
    "e1": [-0.1, -0.15, -0.2, -0.25, -0.3, -0.35, -0.4],
    "e2": [-0.1, -0.15, -0.2, -0.25, -0.3, -0.35, -0.4],
    "eta1": [0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2],
    "eta2": [0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2],
}

params = pd.DataFrame(list(ParameterGrid(param_grid)))

e1 = -0.2
e2 = -0.2
eta1 = 0.02
eta2 = 0.05

## Solution
def simulation_baseline(e1, e2, eta1, eta2):
    a = np.array(
        [
            [1, 0, 0, 0, -P_DE / Q_DE, 0],
            [0, 1, 0, 0, 0, -P1_DE / Q1_DE],
            [-1, 0, 1, 0, 0, 0],
            [0, -1, 0, 1, 0, 0],
            [0, 0, e1 * Q_DE / P_DE, eta1 * Q_DE / P1_DE, -1, 0],
            [0, 0, eta2 * Q1_DE / P_DE, e2 * Q1_DE / P1_DE, 0, -1],
        ]
    )
    b = np.array(
        [
            0,
            0,
            R * C_DE,
            R * C1_DE,
            (e1 + eta1 - 1) * Q_DE,
            (e2 + eta2 - 1) * Q1_DE,
        ]
    )
    x = np.linalg.solve(a, b)
    return x


results_P_EXP = params.apply(
    lambda row: simulation_baseline(
        row["e1"], row["e2"], row["eta1"], row["eta2"]
    )[0],
    axis=1,
)
results_P1_EXP = params.apply(
    lambda row: simulation_baseline(
        row["e1"], row["e2"], row["eta1"], row["eta2"]
    )[1],
    axis=1,
)
results_P_DOM = params.apply(
    lambda row: simulation_baseline(
        row["e1"], row["e2"], row["eta1"], row["eta2"]
    )[2],
    axis=1,
)
results_P1_DOM = params.apply(
    lambda row: simulation_baseline(
        row["e1"], row["e2"], row["eta1"], row["eta2"]
    )[3],
    axis=1,
)
results_Q = params.apply(
    lambda row: simulation_baseline(
        row["e1"], row["e2"], row["eta1"], row["eta2"]
    )[4],
    axis=1,
)
results_Q1 = params.apply(
    lambda row: simulation_baseline(
        row["e1"], row["e2"], row["eta1"], row["eta2"]
    )[5],
    axis=1,
)

results = pd.DataFrame(
    {
        "P_EXP": results_P_EXP,
        "P1_EXP": results_P1_EXP,
        "P_DOM": results_P_DOM,
        "P1_DOM": results_P1_DOM,
    }
)
# Review inputs
print([P_DE, P1_DE, Q_DE, Q1_DE])
# Equations and solution
