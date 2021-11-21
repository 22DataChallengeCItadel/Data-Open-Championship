import numpy as np
from numpy import random
import pandas as pd
from extrapolate_production_data import ProductionData
from Carbon_tariffs import TariffData
from sklearn.model_selection import ParameterGrid
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Import data
## Trade data
trade_flows_country = pd.read_csv("Data/trade_flow_data_all_country.csv")
trade_flows_total = pd.read_csv("Data/trade_flow_data_total.csv")
trade_flows_country = trade_flows_country.rename(
    columns={"qty_in_thousands": "qty_thousands"}
)
trade_flows_total = trade_flows_total.rename(
    columns={"qty_in_thousands": "qty_thousands"}
)

## Carbon tariff data
R = TariffData().get_mean_price()

## Emission data
C_US = ProductionData().get_gwp(product_type="Plastic").to_list()[0]
C_world = C_US * ProductionData().get_ghg_index("world")
C1_US = ProductionData().get_gwp(product_type="Alternative").to_list()[0]
C1_world = C1_US * ProductionData().get_ghg_index("world")

C_IN = C_US * ProductionData().get_ghg_index("india")
C_CN = C_US * ProductionData().get_ghg_index("china")
C_DE = C_US * ProductionData().get_ghg_index("germany")

C1_IN = C1_US * ProductionData().get_ghg_index("india")
C1_CN = C1_US * ProductionData().get_ghg_index("china")
C1_DE = C1_US * ProductionData().get_ghg_index("germany")

P_US = 0.01 * 1000
P_PAP_US = 0.02 * 1000
P_TEX_US = 0.1 * 1000

# Baseline model
## Six unknowns: P_EXP, P1_EXP, P_DOM, P1_DOM, Q, Q1
## Eight parameters: P_US, P1_US, Q_US, Q1_US, R, C_US, C1_US, e1, e2, eta1, eta2

Q_US = trade_flows_total.loc[
    (trade_flows_total["product"] == "plastic bag"),
    "qty_thousands",
].to_list()

Q_PAP_US = trade_flows_total.loc[
    (trade_flows_total["product"] == "paper bag"),
    "qty_thousands",
].to_list()

Q_TEX_US = trade_flows_total.loc[
    (trade_flows_total["product"] == "textile bag"),
    "qty_thousands",
].to_list()

# Set elasticity parameters

N_ITERATION = 20000

params = pd.DataFrame(
    {
        "P_US": P_US
        + random.uniform(
            low=-0.2 * P_US, high=0.2 * P_US, size=N_ITERATION * 10
        ),
        "P_PAP_US": P_PAP_US
        + random.uniform(
            low=-0.2 * P_PAP_US, high=0.2 * P_PAP_US, size=N_ITERATION * 10
        ),
        "P_TEX_US": P_TEX_US
        + random.uniform(
            low=-0.2 * P_TEX_US, high=0.2 * P_TEX_US, size=N_ITERATION * 10
        ),
        "Q_US": random.choice(Q_US, size=N_ITERATION * 10),
        "Q_PAP_US": random.choice(Q_PAP_US, size=N_ITERATION * 10),
        "Q_TEX_US": random.choice(Q_TEX_US, size=N_ITERATION * 10),
        "e1": random.uniform(low=-0.4, high=-0.15, size=N_ITERATION * 10),
        "e2": random.uniform(low=-0.4, high=-0.15, size=N_ITERATION * 10),
        "eta1": random.uniform(low=0.075, high=0.2, size=N_ITERATION * 10),
        "eta2": random.uniform(low=0.075, high=0.2, size=N_ITERATION * 10),
        "Rpt": random.uniform(low=0, high=200, size=N_ITERATION * 10),
    }
)

params = params.loc[
    (-params["e1"] > params["eta1"]) & (-params["e2"] > params["eta2"])
].sample(N_ITERATION)

## Solution: baseline
def simulation_baseline(
    P_US,
    P_PAP_US,
    P_TEX_US,
    Q_US,
    Q_PAP_US,
    Q_TEX_US,
    e1,
    e2,
    eta1,
    eta2,
    Rpt,
    R=R,
):
    Q1_US = Q_PAP_US + Q_TEX_US
    P1_US = np.average([P_PAP_US, P_TEX_US], weights=[Q_PAP_US, Q_TEX_US])
    a = np.array(
        [
            [1, 0, 0, 0, -P_US / Q_US, 0],
            [0, 1, 0, 0, 0, -P1_US / Q1_US],
            [-1, 0, 1, 0, 0, 0],
            [0, -1, 0, 1, 0, 0],
            [0, 0, e1 * Q_US / P_US, eta1 * Q_US / P1_US, -1, 0],
            [0, 0, eta2 * Q1_US / P_US, e2 * Q1_US / P1_US, 0, -1],
        ]
    )
    b = np.array(
        [
            0,
            0,
            R * C_world + Rpt,
            R * C1_world,
            (e1 + eta1 - 1) * Q_US,
            (e2 + eta2 - 1) * Q1_US,
        ]
    )
    x = np.linalg.solve(a, b)
    return np.concatenate((x, np.array([Q1_US, P1_US])), axis=0)


results = params.apply(
    lambda row: simulation_baseline(
        row["P_US"],
        row["P_PAP_US"],
        row["P_TEX_US"],
        row["Q_US"],
        row["Q_PAP_US"],
        row["Q_TEX_US"],
        row["e1"],
        row["e2"],
        row["eta1"],
        row["eta2"],
        row["Rpt"],
    ),
    axis=1,
)

results = pd.DataFrame(results.tolist())
results.columns = [
    "P_EXP",
    "P1_EXP",
    "P_DOM",
    "P1_DOM",
    "Q_DOM",
    "Q1_DOM",
    "Q1_US",
    "P1_US",
]

results = pd.concat(
    [params.reset_index(drop=True), results],
    axis=1,
)

# The ones we are interested in

results["plastic_change"] = results["Q_DOM"] - results["Q_US"]
results["plastic_proportion_before"] = results["Q_US"] / (
    results["Q_US"] + results["Q1_US"]
)
results["plastic_proportion_after"] = results["Q_DOM"] / (
    results["Q_DOM"] + results["Q1_DOM"]
)
results["plastic_percentage_change"] = (
    results["plastic_proportion_after"] - results["plastic_proportion_before"]
) * 100

results["non_plastic_change"] = results["Q1_DOM"] - results["Q1_US"]

results["CO2_change"] = (
    results["non_plastic_change"] * C1_world
    + results["plastic_change"] * C_world
)

results["current_CO2"] = (
    results["Q1_US"] * C1_world + results["Q_US"] * C_world
)

# Regressions

rlm_model = sm.RLM(
    results.plastic_change,
    sm.add_constant(results.Rpt),
    M=sm.robust.norms.HuberT(),
)
rlm_results = rlm_model.fit()

print(rlm_results.summary())

x_pred = -rlm_results.params[0] / rlm_results.params[1]
x_pred