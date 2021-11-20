import pandas as pd
import numpy as np



class ProductionData:

    def __init__(self):
        self.us_baseline_df = pd.read_csv("Data/production_metric_comparison_us.csv")
        self.ghg_emissions = pd.read_csv("Data/ghg_emissions.csv")

        self.emissions = {}
        self.emissions["us"] = pd.read_csv("Data/us_energy_production_by_source.csv").multiply(1000)
        self.emissions["china"] = pd.read_csv("Data/china_energy_production_by_source.csv")
        self.emissions["india"] = pd.read_csv("Data/india_energy_production_by_source.csv")

        self.ghg_index = {}

        self.ghg_index["us"] = 1

        self.ghg_index["china"] = self.calculate_country_ghg_index("china")
        self.ghg_index["india"] = self.calculate_country_ghg_index("india")

        print(self.emissions["us"].head())

    def calculate_country_ghg_index(self, country):

        country_emissions = self.emissions[country]

        for energy_type in self.ghg_emissions.columns:
            pass
    
        return



if __name__ == "__main__":
    ProductionData()





