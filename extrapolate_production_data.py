import pandas as pd
import numpy as np



class ProductionData:

    def __init__(self):
        self.us_baseline_df = pd.read_csv("Data/ProductionData/production_metric_comparison_us.csv")
        self.ghg_emissions = pd.read_csv("Data/ghg_emissions.csv")

        self.emissions = {}
        self.emissions["us"] = pd.read_csv("Data/us_energy_production_by_source.csv")
        self.emissions["china"] = pd.read_csv("Data/china_energy_production_by_source.csv")
        self.emissions["india"] = pd.read_csv("Data/india_energy_production_by_source.csv")

        self.ghg_index = {}

        self.ghg_index["us"] = 1

        self.ghg_index["china"] = self.calculate_country_ghg_index("china")
        self.ghg_index["india"] = self.calculate_country_ghg_index("india")

        print(self.us_baseline_df)


    def calculate_country_ghg_index(self, country):

        country_emissions = self.emissions[country]

        for energy_type in self.ghg_emissions.columns:
            pass
    
        return

    '''
    Returns GWP in units of [tonnes CO2/1000 bags]
    '''
    def get_gwp(self, product_type = "Plastic"):
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
    print(prodData.get_gwp(product_type = "Plastic"))
    print(prodData.get_gwp(product_type = "Alternative"))





