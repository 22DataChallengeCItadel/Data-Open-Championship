import numpy as np
from numpy import random
import pandas as pd
from extrapolate_production_data import ProductionData
from Carbon_tariffs import TariffData
from sklearn.model_selection import ParameterGrid
import seaborn as sns
import matplotlib.pyplot as plt

# Import data
## Trade data
trade_flows_country = pd.read_csv("Data/trade_flow_data.csv")
trade_flows_total = pd.read_csv("Data/trade_flow_data_total.csv")

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

N_ITERATION = 10000

params = pd.DataFrame(
    {
        "Q_US": random.choice(Q_US, size=N_ITERATION * 10),
        "Q_PAP_US": random.choice(Q_PAP_US, size=N_ITERATION * 10),
        "Q_TEX_US": random.choice(Q_TEX_US, size=N_ITERATION * 10),
        "e1": random.uniform(low=-0.4, high=-0.15, size=N_ITERATION * 10),
        "e2": random.uniform(low=-0.4, high=-0.15, size=N_ITERATION * 10),
        "eta1": random.uniform(low=0.075, high=0.2, size=N_ITERATION * 10),
        "eta2": random.uniform(low=0.075, high=0.2, size=N_ITERATION * 10),
    }
)

params = params.loc[
    (-params["e1"] > params["eta1"]) & (-params["e2"] > params["eta2"])
].sample(N_ITERATION)

## Solution: baseline
def simulation_baseline(Q_US, Q_PAP_US, Q_TEX_US, e1, e2, eta1, eta2, R=R):
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
            R * C_world,
            R * C1_world,
            (e1 + eta1 - 1) * Q_US,
            (e2 + eta2 - 1) * Q1_US,
        ]
    )
    x = np.linalg.solve(a, b)
    return np.concatenate((x, np.array([Q_US, Q1_US, P1_US])), axis=0)


results_P_EXP = params.apply(
    lambda row: simulation_baseline(
        row["Q_US"],
        row["Q_PAP_US"],
        row["Q_TEX_US"],
        row["e1"],
        row["e2"],
        row["eta1"],
        row["eta2"],
    )[0],
    axis=1,
)
results_P1_EXP = params.apply(
    lambda row: simulation_baseline(
        row["Q_US"],
        row["Q_PAP_US"],
        row["Q_TEX_US"],
        row["e1"],
        row["e2"],
        row["eta1"],
        row["eta2"],
    )[1],
    axis=1,
)
results_P_DOM = params.apply(
    lambda row: simulation_baseline(
        row["Q_US"],
        row["Q_PAP_US"],
        row["Q_TEX_US"],
        row["e1"],
        row["e2"],
        row["eta1"],
        row["eta2"],
    )[2],
    axis=1,
)
results_P1_DOM = params.apply(
    lambda row: simulation_baseline(
        row["Q_US"],
        row["Q_PAP_US"],
        row["Q_TEX_US"],
        row["e1"],
        row["e2"],
        row["eta1"],
        row["eta2"],
    )[3],
    axis=1,
)
results_Q_DOM = params.apply(
    lambda row: simulation_baseline(
        row["Q_US"],
        row["Q_PAP_US"],
        row["Q_TEX_US"],
        row["e1"],
        row["e2"],
        row["eta1"],
        row["eta2"],
    )[4],
    axis=1,
)
results_Q1_DOM = params.apply(
    lambda row: simulation_baseline(
        row["Q_US"],
        row["Q_PAP_US"],
        row["Q_TEX_US"],
        row["e1"],
        row["e2"],
        row["eta1"],
        row["eta2"],
    )[5],
    axis=1,
)
results_Q_US = params.apply(
    lambda row: simulation_baseline(
        row["Q_US"],
        row["Q_PAP_US"],
        row["Q_TEX_US"],
        row["e1"],
        row["e2"],
        row["eta1"],
        row["eta2"],
    )[6],
    axis=1,
)
results_Q1_US = params.apply(
    lambda row: simulation_baseline(
        row["Q_US"],
        row["Q_PAP_US"],
        row["Q_TEX_US"],
        row["e1"],
        row["e2"],
        row["eta1"],
        row["eta2"],
    )[7],
    axis=1,
)
results_P1_US = params.apply(
    lambda row: simulation_baseline(
        row["Q_US"],
        row["Q_PAP_US"],
        row["Q_TEX_US"],
        row["e1"],
        row["e2"],
        row["eta1"],
        row["eta2"],
    )[8],
    axis=1,
)

results = pd.concat(
    [
        params,
        pd.DataFrame(
            {
                "P_EXP": results_P_EXP,
                "P1_EXP": results_P1_EXP,
                "P_DOM": results_P_DOM,
                "P1_DOM": results_P1_DOM,
                "Q_DOM": results_Q_DOM,
                "Q1_DOM": results_Q1_DOM,
                "Q1_US": results_Q1_US,
                "P1_US:": results_P1_US,
            }
        ),
    ],
    axis=1,
)

# Review inputs
print(
    [
        P_US,
        P_PAP_US,
        P_TEX_US,
        np.median(Q_US),
        np.median(Q_PAP_US),
        np.median(Q_TEX_US),
    ]
)

# The ones we are interested in
results.loc[(results["Q_DOM"] > results["Q_US"])]

results.loc[
    (results["Q_DOM"] > results["Q_US"])
    & (results["Q1_DOM"] < results["Q1_US"])
]

results["plastic_increase"] = results["Q_DOM"] - results["Q_US"]
results["plastic_proportion_before"] = results["Q_US"] / (
    results["Q_US"] + results["Q1_US"]
)
results["plastic_proportion_after"] = results["Q_DOM"] / (
    results["Q_DOM"] + results["Q1_DOM"]
)
results["plastic_percentage_change"] = (
    results["plastic_proportion_after"] - results["plastic_proportion_before"]
) * 100

results["non_plastic_change"] = results["Q1_DOM"] - results["Q1_us"]
# Visualizations
print(results["plastic_increase"].median())
print(results["plastic_increase"].median() / results["Q_US"].mean())
print((results["plastic_increase"] > 0).mean())
sns.distplot(results["plastic_increase"])
plt.axvline(results["plastic_increase"].median(), 0, 2, color="red")
plt.savefig("figs/simulations/plastic_increase_amount.png", dpi=400)

sns.distplot(results["plastic_percentage_change"])
plt.savefig("figs/simulations/plastic_percentage_change.png", dpi=400)


sns.scatterplot(x=results["e1"], y=results["plastic_increase"], alpha=0.1)
sns.scatterplot(x=results["e2"], y=results["plastic_increase"], alpha=0.1)

sns.scatterplot(x=results["eta1"], y=results["plastic_increase"], alpha=0.1)
sns.scatterplot(x=results["eta2"], y=results["plastic_increase"], alpha=0.1)

# Three-country model
## 14 unknowns: P_LIC, P1_LIC, P_MIC, P1_MIC, P_HIC, P1_HIC, P_DOM, P1_DOM, Q_LIC, Q1_LIC, Q_MIC, Q1_MIC, Q_HIC, Q1_HIC
## Eight parameters: P_US, P1_US, Q_IN, Q1_IN, Q_CN, Q1_CN, Q_DE, Q1_DE, R, C_IN, C1_IN, C_CN, C1_CN, C_DE, C1_DE, e1, e2, eta1, eta2

# Stuff below needs to change to include all countries
Q_IN = trade_flows_country.loc[
    (
        (trade_flows_country["Country"] == "India")
        & (trade_flows_country["product"] == "plastic bag")
    ),
    "qty_thousands",
].to_list()

Q_PAP_IN = trade_flows_country.loc[
    (
        (trade_flows_country["Country"] == "India")
        & (trade_flows_country["product"] == "plastic bag")
    ),
    "qty_thousands",
].to_list()

Q_TEX_IN = trade_flows_country.loc[
    (
        (trade_flows_country["Country"] == "India")
        & (trade_flows_country["product"] == "textile bag")
    ),
    "qty_thousands",
].to_list()

Q_CN = trade_flows_country.loc[
    (
        (trade_flows_country["Country"] == "China")
        & (trade_flows_country["product"] == "plastic bag")
    ),
    "qty_thousands",
].to_list()

Q_PAP_CN = trade_flows_country.loc[
    (
        (trade_flows_country["Country"] == "China")
        & (trade_flows_country["product"] == "plastic bag")
    ),
    "qty_thousands",
].to_list()

Q_TEX_CN = trade_flows_country.loc[
    (
        (trade_flows_country["Country"] == "China")
        & (trade_flows_country["product"] == "textile bag")
    ),
    "qty_thousands",
].to_list()

Q_DE = trade_flows_country.loc[
    (
        (trade_flows_country["Country"] == "Germany")
        & (trade_flows_country["product"] == "plastic bag")
    ),
    "qty_thousands",
].to_list()

Q_PAP_DE = trade_flows_country.loc[
    (
        (trade_flows_country["Country"] == "Germany")
        & (trade_flows_country["product"] == "plastic bag")
    ),
    "qty_thousands",
].to_list()

Q_TEX_DE = trade_flows_country.loc[
    (
        (trade_flows_country["Country"] == "Germany")
        & (trade_flows_country["product"] == "textile bag")
    ),
    "qty_thousands",
].to_list()

## Solution: Three country
def simulation_threecountry(
    Q_US,
    Q_PAP_US,
    Q_TEX_US,
    Q_IN,
    Q_PAP_IN,
    Q_TEX_IN,
    Q_CN,
    Q_PAP_CN,
    Q_TEX_CN,
    Q_DE,
    Q_PAP_DE,
    Q_TEX_DE,
    e1,
    e2,
    eta1,
    eta2,
):
    Q1_US = Q_PAP_US + Q_TEX_US
    Q1_IN = Q_PAP_IN + Q_TEX_IN
    Q1_CN = Q_PAP_CN + Q_TEX_CN
    Q1_DE = Q_PAP_DE + Q_TEX_DE
    P1_US = np.average([P_PAP_US, P_TEX_US], weights=[Q_PAP_US, Q_TEX_US])
    a = np.array(
        [
            [1, 0, 0, 0, 0, 0, 0, 0, -P_US / Q_IN, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, -P1_US / Q1_IN, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, -P_US / Q_CN, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, -P1_US / Q1_CN, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, -P_US / Q_DE, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, -P1_US / Q1_DE],
            [-1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, -1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, -1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, -1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, -1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, -1, 0, 1, 0, 0, 0, 0, 0, 0],
            [
                0,
                0,
                0,
                0,
                0,
                0,
                e1 * Q_US / P_US,
                eta1 * Q_US / P1_US,
                -1,
                0,
                -1,
                0,
                -1,
                0,
            ],
            [
                0,
                0,
                0,
                0,
                0,
                0,
                eta2 * Q1_US / P_US,
                e2 * Q1_US / P1_US,
                0,
                -1,
                0,
                -1,
                0,
                -1,
            ],
        ]
    )
    b = np.array(
        [
            0,
            0,
            0,
            0,
            0,
            0,
            R * C_IN,
            R * C1_IN,
            R * C_CN,
            R * C1_CN,
            R * C_DE,
            R * C1_DE,
            (e1 + eta1 - 1) * Q_US,
            (e2 + eta2 - 1) * Q1_US,
        ]
    )
    x = np.linalg.solve(a, b)
    return np.concatenate((x, np.array([Q_US, Q1_US, P1_US])), axis=0)


simulation_threecountry(
    Q_US=Q_US[0],
    Q_PAP_US=Q_PAP_US[0],
    Q_TEX_US=Q_TEX_US[0],
    Q_IN=Q_IN[0],
    Q_PAP_IN=Q_PAP_IN[0],
    Q_TEX_IN=Q_TEX_IN[0],
    Q_CN=Q_CN[0],
    Q_PAP_CN=Q_PAP_CN[0],
    Q_TEX_CN=Q_TEX_CN[0],
    Q_DE=Q_DE[0],
    Q_PAP_DE=Q_PAP_DE[0],
    Q_TEX_DE=Q_TEX_DE[0],
    e1=-0.4,
    e2=-0.4,
    eta1=0.2,
    eta2=0.2,
)
