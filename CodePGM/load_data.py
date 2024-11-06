import pandas as pd

# Load all CSV files
Demand_Data_Normalized = pd.read_csv('../Data/ModelData/Demand_YearlyDemandUtilityProfile_Normalized.csv', delimiter=',')
Fuel_Cost_Data_Normalized = pd.read_csv('../Data\ModelData\FuelCost_PriceDevelopment50years.csv', delimiter=',')
Generation_Data_Normalized = pd.read_csv('../Data\ModelData\VRE_YearlyGenerationProfile_Normalized.csv', delimiter=',')
Generation_Asset_Data_Existing = pd.read_csv('../Data\ModelData\Generators_AssetData_Existing.csv', delimiter=',')
Demand_Unit_Data = pd.read_csv('../Data\ModelData\Demand_UnitSpecificData.csv', delimiter=',')
System_Demand = pd.read_csv('../Data\ModelData\System_Demand.csv', delimiter=',')
Transmission_Capacity = pd.read_csv('../Data\ModelData\Transmission Capacity.csv', delimiter=',')