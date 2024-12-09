import pandas as pd
import numpy as np


## PREPARING THE DATA

# Load all CSV files
Dem_Data = pd.read_excel('../Data/LoadProfile.xlsx', sheet_name='Python_Dem_Data')
Uti_Data = pd.read_excel('../Data/LoadProfile.xlsx', sheet_name='Python_Uti_Data')
Load_Data = pd.read_excel('../Data/LoadProfile.xlsx', sheet_name='Python_Load_Data')
Gen_E_Data = pd.read_excel('../Data/Generators_Existing.xlsx', sheet_name='Python_Gen_E_Data')
Gen_N_Data = pd.read_excel('../Data/Generators_New.xlsx', sheet_name='Python_Gen_N_Data')
Gen_E_Z_Data = pd.read_excel('../Data/Generators_Existing.xlsx', sheet_name='Python_Gen_E_Z_Data')
Gen_N_Z_Data = pd.read_excel('../Data/Generators_New.xlsx', sheet_name='Python_Gen_N_Z_Data')
Gen_E_OpCap_Data = pd.read_excel('../Data/GenerationProfile.xlsx', sheet_name='Python_Gen_E_OpCap_Data')
Gen_N_OpCap_Data = pd.read_excel('../Data/GenerationProfile.xlsx', sheet_name='Python_Gen_N_OpCap_Data')
Trans_Data = pd.read_excel('../Data/Transmission.xlsx', sheet_name='Python_Trans_Data')
Trans_Line_From_Z = pd.read_excel('../Data/Transmission.xlsx', sheet_name='Python_Line_From_Z_Data')
Trans_Line_To_Z = pd.read_excel('../Data/Transmission.xlsx', sheet_name='Python_Line_To_Z_Data')
Trans_Z_Connected_To_Z = pd.read_excel('../Data/Transmission.xlsx', sheet_name='Python_Z_Connected_To_Z_Data')


# Export the needed matrices
Dem = np.array(Dem_Data)    # Demand profile
Uti = np.transpose(np.array(Uti_Data))   # Utility profile
Load_Z = np.array(Load_Data)   # Load Zone
Gen_E_OpCost = np.array(Gen_E_Data['Cost'])   # Existing Generators Operational Cost
Gen_N_OpCost = np.array(Gen_N_Data['Cost'])  # New Generators Operational Cost
Gen_E_Cap = np.array(Gen_E_Data['Capacity'])   # Existing Generators Maximum Capacity
Gen_N_MaxInvCap = np.array(Gen_N_Data['MaxInv (MW)'])  # Maximum New Generators Capacity Investment (MW)
Gen_N_InvCost = np.array(Gen_N_Data['C_CapInv ($/MW)'])  # New Generators Investment Cost ($/MW)
Gen_E_Tech = np.array(Gen_E_Data['Technology'])  # Existing Generators Technology
Gen_N_Tech = np.array(Gen_N_Data['Technology'])  # New Generators Technology
Gen_E_Z = np.array(Gen_E_Z_Data)  # Existing Generators Zone
Gen_N_Z = np.array(Gen_N_Z_Data)  # New Generators Zone
Gen_E_OpCap = np.array(Gen_E_OpCap_Data)  # Maximum Capacity of Existing Generators (Hourly profile if RES, Max capacity otherwise)
Gen_N_OpCap = np.array(Gen_N_OpCap_Data)  # Maximum Capacity of New Generators (Hourly profile if RES, Max capacity otherwise)
Trans_React = np.array(Trans_Data['Reactance'])  # Transmission Reactance
Trans_Cap = np.array(Trans_Data['Capacity [MW]'])  # Transmission Capacity
Trans_Line_From_Z = np.array(Trans_Line_From_Z)  # Mapping the origine zone for each transmission line
Trans_Line_To_Z = np.array(Trans_Line_To_Z)  # Mapping the destination zone for each transmission line
Trans_Z_Connected_To_Z = np.array(Trans_Z_Connected_To_Z)  # Mapping the connected zones for each zone

# Fix the shape of matrices with only one column   
Gen_E_OpCost = Gen_E_OpCost.reshape((Gen_E_OpCost.shape[0], 1))
Gen_N_OpCost = Gen_N_OpCost.reshape((Gen_N_OpCost.shape[0], 1))
Gen_E_Cap = Gen_E_Cap.reshape((Gen_N_OpCost.shape[0], 1))
Gen_N_MaxInvCap = Gen_N_MaxInvCap.reshape((Gen_N_MaxInvCap.shape[0], 1))
Gen_N_InvCost = Gen_N_InvCost.reshape((Gen_N_InvCost.shape[0], 1))
Gen_E_Tech = Gen_E_Tech.reshape((Gen_E_Tech.shape[0], 1))
Gen_N_Tech = Gen_N_Tech.reshape((Gen_N_Tech.shape[0], 1))
Trans_React = Trans_React.reshape((Trans_React.shape[0], 1))
Trans_Cap = Trans_Cap.reshape((Trans_Cap.shape[0], 1))

# Create random scenarios for the investment cost of new generators
N_S = 5 # Number of scenarios for main models
N_S_train = 100 # Number of scenarios for training
N_S_test = 100 # Number of scenarios for testing
max_deviation = 0.8 # 80% of maximum/ min deviation
Gen_N_Data_scenarios = np.zeros((len(Gen_N_Data),N_S))
Gen_N_OpCost_scenarios = np.zeros((len(Gen_N_OpCost),N_S))
Gen_N_Data_scenarios_train = np.zeros((len(Gen_N_Data),N_S_train))
Gen_N_OpCost_scenarios_train = np.zeros((len(Gen_N_OpCost),N_S_train))
Gen_N_Data_scenarios_test = np.zeros((len(Gen_N_Data),N_S_test))
Gen_N_OpCost_scenarios_test = np.zeros((len(Gen_N_OpCost),N_S_test))

# Set the seed for reproducibility
np.random.seed(69)

def create_scenarios(N_S, Gen_N_Data_scenarios, Gen_N_OpCost_scenarios):
    for i in range(N_S):
        # Generate random variations in the range [-x%, x%]
        random_variation = np.random.uniform(-max_deviation, max_deviation, size=len(Gen_N_Data))
        
        # Apply the variations to the original costs
        Gen_N_Data_scenarios[:,i] = Gen_N_Data.loc[:,'C_CapInv ($/MW)'] * (1 + random_variation)

    for i in range(N_S):
        # Generate random variations in the range [-x%, x%]
        random_variation = np.random.uniform(-max_deviation, max_deviation, size=len(Gen_N_OpCost))
        
        # Apply the variations to the original costs
        Gen_N_OpCost_scenarios[:,i] = Gen_N_OpCost[:,0] * (1 + random_variation)
    return Gen_N_Data_scenarios, Gen_N_OpCost_scenarios

create_scenarios(N_S, Gen_N_Data_scenarios, Gen_N_OpCost_scenarios)
create_scenarios(N_S_train, Gen_N_Data_scenarios_train, Gen_N_OpCost_scenarios_train)
create_scenarios(N_S_test, Gen_N_Data_scenarios_test, Gen_N_OpCost_scenarios_test)

# Export the scenarios to CSV files
Gen_N_Data_scenarios_df = pd.DataFrame(Gen_N_Data_scenarios)
Gen_N_OpCost_scenarios_df = pd.DataFrame(Gen_N_OpCost_scenarios)
Gen_N_Data_scenarios_train_df = pd.DataFrame(Gen_N_Data_scenarios_train)
Gen_N_OpCost_scenarios_train_df = pd.DataFrame(Gen_N_OpCost_scenarios_train)
Gen_N_Data_scenarios_test_df = pd.DataFrame(Gen_N_Data_scenarios_test)
Gen_N_OpCost_scenarios_test_df = pd.DataFrame(Gen_N_OpCost_scenarios_test)

# Gen_N_Data_scenarios_df.to_csv('../Data/Scenarios/Main/Generators_New_Scenarios.csv', index=False)
# Gen_N_OpCost_scenarios_df.to_csv('../Data/Scenarios/Main/Generators_New_OpCost_Scenarios.csv', index=False)
# Gen_N_Data_scenarios_train_df.to_csv('../Data/Scenarios/Train/Generators_New_Scenarios_train.csv', index=False)
# Gen_N_OpCost_scenarios_train_df.to_csv('../Data/Scenarios/Train/Generators_New_OpCost_Scenarios_train.csv', index=False)
# Gen_N_Data_scenarios_test_df.to_csv('../Data/Scenarios/Test/Generators_New_Scenarios_test.csv', index=False)
# Gen_N_OpCost_scenarios_test_df.to_csv('../Data/Scenarios/Test/Generators_New_OpCost_Scenarios_test.csv', index=False)

## DATA INDEX

# Create a Dataframe to store the name of each vector/matrix we will use, their size and their content
Data_df = pd.DataFrame(columns=['Name', 'Size', 'Content'])
Data_df['Name'] = ['Dem', 'Uti', 'Load_Z', 'Gen_E_OpCost', 'Gen_N_OpCost','Gen_E_Cap', 'Gen_N_MaxInvCap', 'Gen_N_InvCost', 'Gen_E_Tech', 'Gen_N_Tech', 'Gen_E_Z', 'Gen_N_Z', 'Gen_E_OpCap', 'Gen_N_OpCap', 'Trans_React', 'Trans_Cap', 'Trans_Line_From_Z', 'Trans_Line_To_Z', 'Trans_Z_Connected_To_Z']
Data_df['Size'] = [Dem.shape, Uti.shape, Load_Z.shape, Gen_E_OpCost.shape, Gen_E_Cap.shape, Gen_N_OpCost.shape, Gen_N_MaxInvCap.shape, Gen_N_InvCost.shape, Gen_E_Tech.shape, Gen_N_Tech.shape, Gen_E_Z.shape, Gen_N_Z.shape, Gen_E_OpCap.shape, Gen_N_OpCap.shape, Trans_React.shape, Trans_Cap.shape, Trans_Line_From_Z.shape, Trans_Line_To_Z.shape, Trans_Z_Connected_To_Z.shape ]
Data_df['Content'] = ['Demand for each load for each hour of the investment problem',
                     'Utility for each load for one hour',
                     'Zone of each load',
                     'Operationnal cost of each existing generator',
                     'Operationnal cost of each new generator',
                     'Maximum capacity for existing units',
                     'Maximum capacity investment of each new generator',
                     'Unit Investment cost of each new generator',
                     'Technology of each existing generator',
                     'Technology of each new generator',
                     'Zone of each existing generator',
                     'Zone of each new generator',
                     'Maximum operationnal capacity of each existing energy source for each hour of the investment problem (Hourly profile if RES, Max capacity otherwise)',
                     'Maximum operationnal capacity of each new energy source for each hour of the investment problem (Hourly profile if RES, Max capacity otherwise)',
                     'Transmission Reactance',
                     'Transmission Capacity',
                     'Origine zone of each transmission line',
                     'Destination zone of each transmission line',
                     'Connected zones for each zone']