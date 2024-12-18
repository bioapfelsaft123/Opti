# Loading all the needed Packages
import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB

# Import classes from Class.py
from Data_handling import *
from Class import *
from Class_no_zone import *


## PARAMETERS DEFINITION

# Time
H = 24          # Hours in a day
D = 5           # Typical days in a year
Y = 30          # Years of the investment timeline
N = H*D*Y       # Number of hours in the investment timeline    

epsilon= 0.1
alpha = 0.95
beta = 0
# Number of loads and generators
N_dem = len(Dem[0,:])       # Number of loads
N_gen_E = len(Gen_E_OpCost)   # Number of existing generators
N_gen_N = len(Gen_N_OpCost)   # Number of new generators
N_zone = len(Trans_Z_Connected_To_Z)     # Number of zones
N_line = len(Trans_Line_From_Z)   # Number of transmission lines
N_S = len(Gen_N_OpCost_scenarios[0]) # Number of scenarios


# Hyperparameters
B = 1000000000   # Budget for the investment problem
R = 73 # Conversion rate


## CREATE THE PARAMETERS AND DATA OBJECTS
## CREATE THE PARAMETERS AND DATA OBJECTS
ParametersObj = Parameters(H, D, Y, N, N_dem, N_gen_E, N_gen_N, N_zone, N_line, B, R, N_S, N_S_test, max_deviation)
DataObj = InputData(Dem, Uti, Load_Z, Gen_E_OpCost, Gen_N_OpCost, Gen_N_MaxInvCap, Gen_E_Cap, Gen_N_InvCost, Gen_E_Tech, Gen_N_Tech, Gen_E_Z, Gen_N_Z, Gen_E_OpCap, Gen_N_OpCap, Trans_React, Trans_Cap, Trans_Line_From_Z, Trans_Line_To_Z, Trans_Z_Connected_To_Z,Gen_N_Data_scenarios,Gen_N_OpCost_scenarios, Gen_N_Data_scenarios_train, Gen_N_OpCost_scenarios_train, Gen_N_Data_scenarios_test, Gen_N_OpCost_scenarios_test)
# Run the Market Clearing Problem
MarketClearing1 = MarketClearingModel1(ParametersObj, DataObj)

DA_Price = MarketClearing1.res.DA_price


m = gp.Model('Stochastic problem')


P_N = m.addMVar((N_gen_N), lb=0) # Invested capacity in every new generator
p_N = m.addMVar((N, N_gen_N, N_S), lb=0) # Power output per hour for every new generator


# Capacity investment constraint
cap_inv = m.addConstr(P_N <= Gen_N_MaxInvCap, name='Maximum capacity investment')

# Max production constraint
for s in range(N_S):
    max_p_N = m.addConstr(p_N[:,:,s] <= Gen_N_OpCap * P_N, name='Maximum RES production') 

# Budget constraint
for s in range(N_S):
    budget = m.addConstr(gp.quicksum(P_N[g] * Gen_N_Data_scenarios[g,s] for g in range(N_gen_N)) <= B, name='Budget constraint')


revenues = (gp.quicksum((p_N[:,:,s] @ Gen_N_Z.T for s in range(N_S))) *  DA_Price).sum()
op_costs = (gp.quicksum(p_N[:,:,s] @ Gen_N_OpCost_scenarios[:,s] for s in range(N_S))).sum()
invest_costs = gp.quicksum(P_N @ Gen_N_Data_scenarios[:,s] for s in range(N_S))
objective = (1/N_S) * (R*(revenues - op_costs) - invest_costs)
m.setObjective(objective, GRB.MAXIMIZE)


m.optimize()
        
# Display the generators the model invested in, in a dataframe
P_N = P_N.X
P_N = P_N.reshape((N_gen_N,1))
df = pd.DataFrame(Gen_N_Tech, columns = ['Technology'])
df['Invested capacity (MW)'] = P_N  
print(df)
