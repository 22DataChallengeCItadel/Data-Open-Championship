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

### Get income levels of countries
income_level_Data = pd.read_csv("Data/country_level.csv")[
    ["country_name", "income_id"]
]
income_level_Data.loc[
    income_level_Data["country_name"] == "Bosnia and Herzegovina",
    "country_name",
] = "Bosnia-Herzegov"
income_level_Data.loc[
    income_level_Data["country_name"] == "Myanmar", "country_name"
] = "Burma"
income_level_Data.loc[
    income_level_Data["country_name"] == "Dominican Republic", "country_name"
] = "Dominican Rep"
income_level_Data.loc[
    income_level_Data["country_name"] == "Egypt, Arab Rep.", "country_name"
] = "Egypt"
income_level_Data.loc[
    income_level_Data["country_name"] == "Hong Kong SAR, China", "country_name"
] = "Hong Kong"
income_level_Data.loc[
    income_level_Data["country_name"] == "Lao PDR", "country_name"
] = "Laos"
income_level_Data.loc[
    income_level_Data["country_name"] == "Russian Federation", "country_name"
] = "Russia"
income_level_Data.loc[
    income_level_Data["country_name"] == "Slovak Republic", "country_name"
] = "Slovakia"
income_level_Data.loc[
    income_level_Data["country_name"] == "Korea, Rep.", "country_name"
] = "South Korea"
income_level_Data.loc[
    income_level_Data["country_name"] == "United Arab Emirates", "country_name"
] = "United Arab Em"
income_level_Data.loc[
    income_level_Data["country_name"] == "Venezuela, RB", "country_name"
] = "Venezuela"

trade_flows_country = pd.merge(
    trade_flows_country,
    income_level_Data,
    how="left",
    left_on="Country",
    right_on="country_name",
)
trade_flows_country.loc[
    trade_flows_country["Country"] == "Taiwan", "income_id"
] = "HIC"

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

trade_flows_country = trade_flows_country.rename(
    columns={"qty_in_thousands": "qty_thousands"}
)
trade_flows_total = trade_flows_total.rename(
    columns={"qty_in_thousands": "qty_thousands"}
)

# Aggregate data by income levels
pd.DataFrame(
    trade_flows_country.loc[
        (trade_flows_country["product"] == "plastic bag")
        & (trade_flows_country["income_id"].isin(["LMC", "LIC"])),
    ]
    .groupby(["Year"])
    .agg(sum_qty_thousands=pd.NamedAgg(column="qty_thousands", aggfunc=sum))
)
# Three-country model
## 14 unknowns: P_LIC, P1_LIC, P_MIC, P1_MIC, P_HIC, P1_HIC, P_DOM, P1_DOM, Q_LIC, Q1_LIC, Q_MIC, Q1_MIC, Q_HIC, Q1_HIC
## Eight parameters: P_US, P1_US, Q_IN, Q1_IN, Q_CN, Q1_CN, Q_DE, Q1_DE, R, C_IN, C1_IN, C_CN, C1_CN, C_DE, C1_DE, e1, e2, eta1, eta2

# Stuff below needs to change to include all countries
Q_US = trade_flows_total.loc[
    (trade_flows_total["product"] == "plastic bag"),
    "qty_thousands",
].to_list()

Q_IN = pd.DataFrame(
    trade_flows_country.loc[
        (trade_flows_country["product"] == "plastic bag")
        & (trade_flows_country["income_id"].isin(["LMC", "LIC"])),
    ]
    .groupby(["Year"])
    .agg(sum_qty_thousands=pd.NamedAgg(column="qty_thousands", aggfunc=sum))
)["sum_qty_thousands"].tolist()

Q_PAP_IN = pd.DataFrame(
    trade_flows_country.loc[
        (trade_flows_country["product"] == "paper bag")
        & (trade_flows_country["income_id"].isin(["LMC", "LIC"])),
    ]
    .groupby(["Year"])
    .agg(sum_qty_thousands=pd.NamedAgg(column="qty_thousands", aggfunc=sum))
)["sum_qty_thousands"].tolist()

Q_TEX_IN = pd.DataFrame(
    trade_flows_country.loc[
        (trade_flows_country["product"] == "textile bag")
        & (trade_flows_country["income_id"].isin(["LMC", "LIC"])),
    ]
    .groupby(["Year"])
    .agg(sum_qty_thousands=pd.NamedAgg(column="qty_thousands", aggfunc=sum))
)["sum_qty_thousands"].tolist()

Q_CN = pd.DataFrame(
    trade_flows_country.loc[
        (trade_flows_country["product"] == "plastic bag")
        & (trade_flows_country["income_id"] == "UMC"),
    ]
    .groupby(["Year"])
    .agg(sum_qty_thousands=pd.NamedAgg(column="qty_thousands", aggfunc=sum))
)["sum_qty_thousands"].tolist()

Q_PAP_CN = pd.DataFrame(
    trade_flows_country.loc[
        (trade_flows_country["product"] == "paper bag")
        & (trade_flows_country["income_id"] == "UMC"),
    ]
    .groupby(["Year"])
    .agg(sum_qty_thousands=pd.NamedAgg(column="qty_thousands", aggfunc=sum))
)["sum_qty_thousands"].tolist()

Q_TEX_CN = pd.DataFrame(
    trade_flows_country.loc[
        (trade_flows_country["product"] == "textile bag")
        & (trade_flows_country["income_id"] == "UMC"),
    ]
    .groupby(["Year"])
    .agg(sum_qty_thousands=pd.NamedAgg(column="qty_thousands", aggfunc=sum))
)["sum_qty_thousands"].tolist()

Q_DE = pd.DataFrame(
    trade_flows_country.loc[
        (trade_flows_country["product"] == "plastic bag")
        & (trade_flows_country["income_id"] == "HIC"),
    ]
    .groupby(["Year"])
    .agg(sum_qty_thousands=pd.NamedAgg(column="qty_thousands", aggfunc=sum))
)["sum_qty_thousands"].tolist()

Q_PAP_DE = pd.DataFrame(
    trade_flows_country.loc[
        (trade_flows_country["product"] == "paper bag")
        & (trade_flows_country["income_id"] == "HIC"),
    ]
    .groupby(["Year"])
    .agg(sum_qty_thousands=pd.NamedAgg(column="qty_thousands", aggfunc=sum))
)["sum_qty_thousands"].tolist()

Q_TEX_DE = pd.DataFrame(
    trade_flows_country.loc[
        (trade_flows_country["product"] == "textile bag")
        & (trade_flows_country["income_id"] == "HIC"),
    ]
    .groupby(["Year"])
    .agg(sum_qty_thousands=pd.NamedAgg(column="qty_thousands", aggfunc=sum))
)["sum_qty_thousands"].tolist()

## Solution: Three country
def simulation_threecountry(
    P_US,
    P_PAP_US,
    P_TEX_US,
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
    Q_US = Q_IN + Q_CN + Q_DE
    Q_PAP_US = Q_PAP_IN + Q_PAP_CN + Q_PAP_DE
    Q_TEX_US = Q_TEX_IN + Q_TEX_CN + Q_TEX_DE
    Q1_IN = Q_PAP_IN + Q_TEX_IN
    Q1_CN = Q_PAP_CN + Q_TEX_CN
    Q1_DE = Q_PAP_DE + Q_TEX_DE
    Q1_US = Q1_IN + Q1_CN + Q1_DE
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
        "Q_IN": random.choice(Q_IN, size=N_ITERATION * 10),
        "Q_PAP_IN": random.choice(Q_PAP_IN, size=N_ITERATION * 10),
        "Q_TEX_IN": random.choice(Q_TEX_IN, size=N_ITERATION * 10),
        "Q_CN": random.choice(Q_CN, size=N_ITERATION * 10),
        "Q_PAP_CN": random.choice(Q_PAP_CN, size=N_ITERATION * 10),
        "Q_TEX_CN": random.choice(Q_TEX_CN, size=N_ITERATION * 10),
        "Q_DE": random.choice(Q_DE, size=N_ITERATION * 10),
        "Q_PAP_DE": random.choice(Q_PAP_DE, size=N_ITERATION * 10),
        "Q_TEX_DE": random.choice(Q_TEX_DE, size=N_ITERATION * 10),
        "e1": random.uniform(low=-0.4, high=-0.15, size=N_ITERATION * 10),
        "e2": random.uniform(low=-0.4, high=-0.15, size=N_ITERATION * 10),
        "eta1": random.uniform(low=0.075, high=0.2, size=N_ITERATION * 10),
        "eta2": random.uniform(low=0.075, high=0.2, size=N_ITERATION * 10),
    }
)

params = params.loc[
    (-params["e1"] > params["eta1"]) & (-params["e2"] > params["eta2"])
].sample(N_ITERATION)

simulation_threecountry(
    P_US=P_US,
    P_PAP_US=P_PAP_US,
    P_TEX_US=P_TEX_US,
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

results = params.apply(
    lambda row: simulation_threecountry(
        P_US=row["P_US"],
        P_PAP_US=row["P_PAP_US"],
        P_TEX_US=row["P_TEX_US"],
        Q_IN=row["Q_IN"],
        Q_PAP_IN=row["Q_PAP_IN"],
        Q_TEX_IN=row["Q_TEX_IN"],
        Q_CN=row["Q_CN"],
        Q_PAP_CN=row["Q_PAP_CN"],
        Q_TEX_CN=row["Q_TEX_CN"],
        Q_DE=row["Q_DE"],
        Q_PAP_DE=row["Q_PAP_DE"],
        Q_TEX_DE=row["Q_TEX_DE"],
        e1=row["e1"],
        e2=row["e2"],
        eta1=row["eta1"],
        eta2=row["eta2"],
    ),
    axis=1,
)

results = pd.DataFrame(results.tolist())
results.columns = [
    "P_LIC",
    "P1_LIC",
    "P_MIC",
    "P1_MIC",
    "P_HIC",
    "P1_HIC",
    "P_DOM",
    "P1_DOM",
    "Q_LIC",
    "Q1_LIC",
    "Q_MIC",
    "Q1_MIC",
    "Q_HIC",
    "Q1_HIC",
    "Q_US",
    "Q1_US",
    "P1_US",
]

results = pd.concat(
    [params.reset_index(drop=True), results],
    axis=1,
)

results["Q_DOM"] = results["Q_LIC"] + results["Q_MIC"] + results["Q_HIC"]
results["Q1_DOM"] = results["Q1_LIC"] + results["Q1_MIC"] + results["Q1_HIC"]

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
plt.xticks(
    [
        -10 * 10 ** 6,
        -5 * 10 ** 6,
        0,
        5 * 10 ** 6,
        10 * 10 ** 6,
        15 * 10 ** 6,
        20 * 10 ** 6,
        25 * 10 ** 6,
    ],
    ["-10M", "-5M", "0M", "5M", "10M", "15M", "20M", "25M"],
)
# plt.show()
plt.savefig("figs/simulations/plastic_increase_amount_3c.png", dpi=400)
plt.clf()

# Change in non-plastic quantity
print(results["non_plastic_change"].median())
print(results["non_plastic_change"].median() / results["Q1_US"].mean())
print((results["non_plastic_change"] > 0).mean())

sns.distplot(results["non_plastic_change"])
plt.axvline(results["non_plastic_change"].median(), 0, 10, color="red")
plt.savefig("figs/simulations/non_plastic_increase_amount_3c.png", dpi=400)
plt.clf()

# Change in plastic bag price
print(results["P_DOM"].median())
print(results["P_US"].median())

# Change in non-plastic bag price
print(results["P1_DOM"].median())
print(results["P1_US"].median())

## Change in plastic quantity: LIC
results["plastic_change_LIC"] = results["Q_LIC"] - results["Q_IN"]
print(results["plastic_change_LIC"].median())
print(results["plastic_change_LIC"].median() / results["Q_IN"].mean())
print((results["plastic_change_LIC"] > 0).mean())

sns.distplot(results["plastic_change_LIC"])
plt.axvline(results["plastic_change_LIC"].median(), 0, 2, color="red")
plt.savefig("figs/simulations/plastic_increase_amount_LIC.png", dpi=400)
plt.clf()

results["P_change_LIC"] = results["P_LIC"] - results["P_US"]
print(results["P_change_LIC"].median())
print(results["P_change_LIC"].median() / results["P_US"].mean())
print((results["P_change_LIC"] > 0).mean())

sns.distplot(results["P_change_LIC"])
plt.axvline(results["P_change_LIC"].median(), 0, 2, color="red")
plt.xlabel("Price change (LIC)")
plt.savefig("figs/simulations/P_increase_amount_LIC.png", dpi=400)
plt.clf()

## Change in plastic quantity: MIC
results["plastic_change_MIC"] = results["Q_MIC"] - results["Q_CN"]
print(results["plastic_change_MIC"].median())
print(results["plastic_change_MIC"].median() / results["Q_IN"].mean())
print((results["plastic_change_MIC"] > 0).mean())

sns.distplot(results["plastic_change_MIC"])
plt.axvline(results["plastic_change_MIC"].median(), 0, 2, color="red")
plt.savefig("figs/simulations/plastic_increase_amount_MIC.png", dpi=400)
plt.clf()

results["P_change_MIC"] = results["P_MIC"] - results["P_US"]
print(results["P_change_MIC"].median())
print(results["P_change_MIC"].median() / results["P_US"].mean())
print((results["P_change_MIC"] > 0).mean())

sns.distplot(results["P_change_MIC"])
plt.axvline(results["P_change_MIC"].median(), 0, 2, color="red")
plt.xlabel("Price change (MIC)")
plt.savefig("figs/simulations/P_increase_amount_MIC.png", dpi=400)
plt.clf()

## Change in plastic quantity: HIC
results["plastic_change_HIC"] = results["Q_HIC"] - results["Q_DE"]
print(results["plastic_change_HIC"].median())
print(results["plastic_change_HIC"].median() / results["Q_IN"].mean())
print((results["plastic_change_HIC"] > 0).mean())

sns.distplot(results["plastic_change_HIC"])
plt.axvline(results["plastic_change_HIC"].median(), 0, 2, color="red")
plt.savefig("figs/simulations/plastic_increase_amount_HIC.png", dpi=400)
plt.clf()

results["P_change_HIC"] = results["P_HIC"] - results["P_US"]
print(results["P_change_HIC"].median())
print(results["P_change_HIC"].median() / results["P_US"].mean())
print((results["P_change_HIC"] > 0).mean())

sns.distplot(results["P_change_HIC"])
plt.axvline(results["P_change_HIC"].median(), 0, 2, color="red")
plt.xlabel("Price change (HIC)")
plt.savefig("figs/simulations/P_increase_amount_HIC.png", dpi=400)
plt.clf()

## Change in non-plastic quantity: LIC
results["non_plastic_change_LIC"] = (
    results["Q1_LIC"] - results["Q_PAP_IN"] - results["Q_TEX_IN"]
)
print(results["non_plastic_change_LIC"].median())
print(
    results["non_plastic_change_LIC"].median()
    / (results["Q_PAP_IN"].mean() + results["Q_TEX_IN"].mean())
)
print((results["non_plastic_change_LIC"] > 0).mean())

sns.distplot(results["non_plastic_change_LIC"])
plt.axvline(results["non_plastic_change_LIC"].median(), 0, 2, color="red")
plt.savefig("figs/simulations/non_plastic_increase_amount_LIC.png", dpi=400)
plt.clf()

results["P1_change_LIC"] = results["P1_LIC"] - results["P1_US"]
print(results["P1_change_LIC"].median())
print(results["P1_change_LIC"].median() / results["P_US"].mean())
print((results["P1_change_LIC"] > 0).mean())

sns.distplot(results["P1_change_LIC"])
plt.axvline(results["P1_change_LIC"].median(), 0, 2, color="red")
plt.xlabel("Price change (LIC)")
plt.savefig("figs/simulations/P1_increase_amount_LIC.png", dpi=400)
plt.clf()

## Change in non-plastic quantity: MIC
results["non_plastic_change_MIC"] = (
    results["Q1_MIC"] - results["Q_PAP_CN"] - results["Q_TEX_CN"]
)
print(results["non_plastic_change_MIC"].median())
print(
    results["non_plastic_change_MIC"].median()
    / (results["Q_PAP_CN"].mean() + results["Q_TEX_CN"].mean())
)
print((results["non_plastic_change_MIC"] > 0).mean())

sns.distplot(results["non_plastic_change_MIC"])
plt.axvline(results["non_plastic_change_MIC"].median(), 0, 2, color="red")
plt.savefig("figs/simulations/non_plastic_increase_amount_MIC.png", dpi=400)
plt.clf()

results["P1_change_MIC"] = results["P1_MIC"] - results["P1_US"]
print(results["P1_change_MIC"].median())
print(results["P1_change_MIC"].median() / results["P_US"].mean())
print((results["P1_change_MIC"] > 0).mean())

sns.distplot(results["P1_change_MIC"])
plt.axvline(results["P1_change_MIC"].median(), 0, 2, color="red")
plt.xlabel("Price change (MIC)")
plt.savefig("figs/simulations/P1_increase_amount_MIC.png", dpi=400)
plt.clf()

## Change in non-plastic quantity: HIC
results["non_plastic_change_HIC"] = (
    results["Q1_HIC"] - results["Q_PAP_DE"] - results["Q_TEX_DE"]
)
print(results["non_plastic_change_HIC"].median())
print(
    results["non_plastic_change_HIC"].median()
    / (results["Q_PAP_DE"].mean() + results["Q_TEX_DE"].mean())
)
print((results["non_plastic_change_HIC"] > 0).mean())

sns.distplot(results["non_plastic_change_HIC"])
plt.axvline(results["non_plastic_change_HIC"].median(), 0, 2, color="red")
plt.savefig("figs/simulations/non_plastic_increase_amount_HIC.png", dpi=400)
plt.clf()

results["P1_change_HIC"] = results["P1_HIC"] - results["P1_US"]
print(results["P1_change_HIC"].median())
print(results["P1_change_HIC"].median() / results["P_US"].mean())
print((results["P1_change_HIC"] > 0).mean())

sns.distplot(results["P1_change_HIC"])
plt.axvline(results["P1_change_HIC"].median(), 0, 2, color="red")
plt.xlabel("Price change (HIC)")
plt.savefig("figs/simulations/P1_increase_amount_HIC.png", dpi=400)
plt.clf()
