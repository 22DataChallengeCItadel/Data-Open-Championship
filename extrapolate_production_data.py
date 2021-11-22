import pandas as pd
import numpy as np



class ProductionData:

    def __init__(self):
        self.us_baseline_df = pd.read_csv("Data/ProductionData/production_metric_comparison_us.csv")
        self.ghg_emissions = pd.read_csv("Data/EnergySources/ghg_emissions.csv")

        self.emissions = {}
        self.emissions["us"] = pd.read_csv("Data/EnergySources/us_energy_production_by_source.csv")
        self.emissions["china"] = pd.read_csv("Data/EnergySources/china_energy_production_by_source.csv")
        self.emissions["india"] = pd.read_csv("Data/EnergySources/india_energy_production_by_source.csv")
        self.emissions["world"] = pd.read_csv("Data/EnergySources/world_energy_production_by_source.csv")
        self.emissions["germany"] = pd.read_csv("Data/EnergySources/germany_energy_production_by_source.csv")

        self.ghg_raw = {}
        self.ghg_raw["us"] = self.calculate_country_ghg_per_mwh("us")
        self.ghg_raw["china"] = self.calculate_country_ghg_per_mwh("china")
        self.ghg_raw["india"] = self.calculate_country_ghg_per_mwh("india")
        self.ghg_raw["world"] = self.calculate_country_ghg_per_mwh("world")
        self.ghg_raw["germany"] = self.calculate_country_ghg_per_mwh("germany")

        #print(self.ghg_raw)

        self.ghg_index = {}
        self.ghg_index["us"] = 1
        self.ghg_index["china"] = round(self.ghg_raw["china"]/self.ghg_raw["us"],2)
        self.ghg_index["india"] = round(self.ghg_raw["india"]/self.ghg_raw["us"],2)
        self.ghg_index["world"] = round(self.ghg_raw["world"]/self.ghg_raw["us"],2)
        self.ghg_index["germany"] = round(self.ghg_raw["germany"]/self.ghg_raw["us"],2)

        #print(self.ghg_index)

        #print(self.ghg_emissions)
        #print(list(self.ghg_emissions["Source"])

    # Outputs units [tonne CO2e/mWh]
    def calculate_country_ghg_per_mwh(self, country):

        country_emissions = self.emissions[country]
        country_emissions = country_emissions.loc[country_emissions["Year"] == 2018].drop(columns=["Year"])

        total_energy = country_emissions.sum(axis=1)

        country_emissions_pct = country_emissions*(1/float(total_energy))

        net_ghg_per_mwh = 0
        for energy_type in list(self.ghg_emissions["Source"]):
            # Units: [g CO2e/kWh]*pct (unitless)*[tonne/1000000 g]*[1000 kWh/mWh]
            net_ghg_per_mwh += self.ghg_emissions.loc[self.ghg_emissions["Source"] == energy_type,"Median_Emissions"].values[0]*country_emissions_pct[energy_type].values[0]/1000

        return round(net_ghg_per_mwh,5)

    
    '''
    Returns ghg index of country specified
    '''
    def get_ghg_index(self, country):
        return self.ghg_index[country]

    '''
    Returns GWP in units of [tonnes CO2/1000 bags]
    '''
    def get_gwp(self, country = "us", product_type = "Plastic"):
        kg_per_plastic_bag = 0.00643

        kg_per_paper_bag = 0.023
        kg_per_textile_bag = 0.15

        kg_per_alternative_bag = 0.701*kg_per_paper_bag + 0.299*kg_per_textile_bag

        carrier_bags = self.us_baseline_df.loc[self.us_baseline_df["Category"] == "Carrier_bags"]

        if(product_type == "Plastic"):
            gwp = (carrier_bags["Plastic_warming_potential"]/carrier_bags["Plastic_weight"])*kg_per_plastic_bag*1000
        elif(product_type == "Alternative"):
            gwp = (carrier_bags["Alternative_warming_potential"]/carrier_bags["Alternative_weight"])*kg_per_alternative_bag*1000
        else:
            raise Exception("Invalid product type. Please select Plastic or Alternative.")

        return gwp



if __name__ == "__main__":
    prodData = ProductionData()
    print(prodData.get_gwp(product_type = "Plastic")*1000*prodData.get_ghg_index("germany"))
    print(prodData.get_gwp(product_type = "Alternative")*1000*prodData.get_ghg_index("germany"))









