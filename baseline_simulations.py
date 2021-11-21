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
trade_flows_country = pd.read_csv("Data/trade_flow_data_all_country.csv")
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
    "qty_in_thousands",
].to_list()

Q_PAP_US = trade_flows_total.loc[
    (trade_flows_total["product"] == "paper bag"),
    "qty_in_thousands",
].to_list()

Q_TEX_US = trade_flows_total.loc[
    (trade_flows_total["product"] == "textile bag"),
    "qty_in_thousands",
].to_list()

# Set elasticity parameters

N_ITERATION = 10000

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
    }
)

params = params.loc[
    (-params["e1"] > params["eta1"]) & (-params["e2"] > params["eta2"])
].sample(N_ITERATION)

## Solution: baseline
def simulation_baseline(
    P_US, P_PAP_US, P_TEX_US, Q_US, Q_PAP_US, Q_TEX_US, e1, e2, eta1, eta2, R=R
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
            R * C_world,
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
results.loc[(results["Q_DOM"] > results["Q_US"])]

results.loc[
    (results["Q_DOM"] > results["Q_US"])
    & (results["Q1_DOM"] < results["Q1_US"])
]

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

# Visualizations
## Change in plastic quantity
print(results["plastic_change"].median())
print(results["plastic_change"].median() / results["Q_US"].mean())
print((results["plastic_change"] > 0).mean())

sns.distplot(results["plastic_change"])
plt.axvline(results["plastic_change"].median(), 0, 2, color="red")
plt.savefig("figs/simulations/plastic_increase_amount.png", dpi=400)
# plt.show()
plt.clf()


# Change in non-plastic quantity
print(results["non_plastic_change"].median())
print(results["non_plastic_change"].median() / results["Q1_US"].mean())
print((results["non_plastic_change"] > 0).mean())

sns.distplot(results["non_plastic_change"])
plt.axvline(results["non_plastic_change"].median(), 0, 10, color="red")
plt.xticks([-4*10**6,-3.5*10**6, -3*10**6,-2.5*10**6, -2*10**6,-1.5*10**6, -1*10**6, -0.5*10**6],\
           ['-4M', '-3.5M','-3M','-2.5M', '-2M', '-1.5M','-1M','-0.5M'])
plt.savefig("figs/simulations/non_plastic_increase_amount.png", dpi=400)
# plt.show()
plt.clf()

# Change in plastic bag price
print(results["P_DOM"].median())
print(results["P_EXP"].median())
print(results["P_US"].median())


print(results["P1_DOM"].median())
print(results["P1_EXP"].median())
print(results["P1_US"].median())

np.average(
    [results["P_PAP_US"], results["P_TEX_US"]],
    weights=[results["Q_PAP_US"], results["Q_TEX_US"]],
)

ax = sns.scatterplot(x=results["e1"], y=results["plastic_change"], alpha=0.1)
ax.set(xlabel="e", ylabel="Change in plastic bag import")
plt.yticks([-5*10**6,0,5*10**6,10*10**6,15*10**6, 20*10**6], ['-5M', '0M','5M','10M','15M', '20M'])
# plt.show()
plt.savefig("figs/simulations/plastic_change_e1.png", dpi=400)
plt.clf()


ax = sns.scatterplot(x=results["e2"], y=results["plastic_change"], alpha=0.1)
ax.set(xlabel="e'", ylabel="Change in plastic bag import")
plt.yticks([-5*10**6,0,5*10**6,10*10**6,15*10**6, 20*10**6], ['-5M', '0M','5M','10M','15M', '20M'])
# plt.show()
plt.savefig("figs/simulations/plastic_change_e2.png", dpi=400)
plt.clf()

ax = sns.scatterplot(x=results["eta1"], y=results["plastic_change"], alpha=0.1)
ax.set(xlabel="eta", ylabel="Change in plastic bag import")
plt.yticks([-5*10**6,0,5*10**6,10*10**6,15*10**6, 20*10**6], ['-5M', '0M','5M','10M','15M', '20M'])
# plt.show()
plt.savefig("figs/simulations/plastic_change_eta1.png", dpi=400)
plt.clf()

# New simulation: change in R

N_ITERATION = 10000

params_new_R = pd.DataFrame(
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
        "R": np.linspace(R, 2 * R, num=N_ITERATION * 10),
    }
)

params_new_R = params_new_R.loc[
    (-params_new_R["e1"] > params_new_R["eta1"])
    & (-params_new_R["e2"] > params_new_R["eta2"])
].sample(N_ITERATION)

results_Q_DOM_new_R = params_new_R.apply(
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
        row["R"],
    )[4],
    axis=1,
)

results_new_R = pd.concat(
    [
        params_new_R,
        pd.DataFrame({"Q_DOM": results_Q_DOM_new_R}),
    ],
    axis=1,
)

results_new_R["plastic_change"] = (
    results_new_R["Q_DOM"] - results_new_R["Q_US"]
)

ax = sns.regplot(
    x=results_new_R["R"],
    y=results_new_R["plastic_change"],
    scatter_kws={"alpha": 0.1},
    line_kws={"color": "red"},
)
plt.yticks([-1*10**7,0,1*10**7,2*10**7,3*10**7, 4*10**7], ['-10M', '0M','10M','20M','30M', '40M'])
plt.xticks([70,80,90,100,110,120], ['70%','80%','90%','100%','110%','120%'])
# plt.show()
plt.savefig("figs/simulations/plastic_change_R.png", dpi=400)
plt.clf()