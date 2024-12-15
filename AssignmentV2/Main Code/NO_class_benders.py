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
ParametersObj = Parameters(H, D, Y, N, N_dem, N_gen_E, N_gen_N, N_zone, N_line, B, R, N_S, N_S_test, max_deviation)
DataObj = InputData(Dem, Uti, Load_Z, Gen_E_OpCost, Gen_N_OpCost, Gen_N_MaxInvCap, Gen_E_Cap, Gen_N_InvCost, Gen_E_Tech, Gen_N_Tech, Gen_E_Z, Gen_N_Z, Gen_E_OpCap, Gen_N_OpCap, Trans_React, Trans_Cap, Trans_Line_From_Z, Trans_Line_To_Z, Trans_Z_Connected_To_Z,Gen_N_Data_scenarios,Gen_N_OpCost_scenarios, Gen_N_Data_scenarios_train, Gen_N_OpCost_scenarios_train, Gen_N_Data_scenarios_test, Gen_N_OpCost_scenarios_test)

# Run the Market Clearing Problem
MarketClearing1 = MarketClearingModel1(ParametersObj, DataObj)

DA_Price = MarketClearing1.res.DA_price


# %%

rev = DA_Price @ Gen_N_Z
## BENDERS

mas = gp.Model('Benders')

P_N = mas.addMVar((N_gen_N), lb=0) # Invested capacity in every new generator
q = mas.addMVar(1, lb=0) # For subproblem

# Capacity investment constraint
cap_inv = mas.addConstr(P_N <= Gen_N_MaxInvCap, name='Maximum capacity investment')

# Budget constraint
for s in range(N_S):
    budget = mas.addConstr(P_N @ Gen_N_Data_scenarios[:,s] <= B, name='Budget constraint')

objective = - gp.quicksum(P_N @ Gen_N_Data_scenarios[:,s] for s in range(N_S))*(1/N_S) + q
mas.setObjective(objective, GRB.MAXIMIZE)

def solve_master(nu):
    tmp = Gen_N_OpCap * P_N
    mas.addConstr(gp.quicksum(nu[:,g,s] @ (tmp)[:,g] for g in range(N_gen_N) for s in range(N_S)) >= q)

    mas.optimize()

    print(q.x)

    return mas.objVal, P_N.X
        

def solve_sub(P_N_c):

    sub = gp.Model('Benders_sub')
    
    nu = sub.addMVar((N, N_gen_N, N_S), lb=0) # Power output per hour for every new generator

    # Max production constraint
    for h in range(N):
        for s in range(N_S):
                sub.addConstr(nu[h,:,s] >= (rev[h,:] - Gen_N_OpCost_scenarios[:,s])*R/N_S)
    Prod = Gen_N_OpCap * P_N_c
    objective = gp.quicksum(nu[:,g,s] @ Prod[:,g] for g in range(N_gen_N) for s in range(N_S))

    sub.setObjective(objective, GRB.MINIMIZE)

    sub.optimize()

    return sub.objVal, nu.x

import math

# Main code
def main():
    UB = math.inf
    LB = -math.inf
    Delta = 10
    Pbar = np.zeros(N_gen_N)
    it = 1

    while (UB - LB > Delta):
        sub_obj, nu = solve_sub(Pbar)  # Replace solve_sub with the corresponding function

        LB = max(LB, sub_obj - sum(Pbar @ Gen_N_Data_scenarios[:,s] for s in range(N_S))/N_S) 

        mas_obj, P_N = solve_master(nu) 

        Pbar = P_N
        UB = mas_obj

        print(f"It: {it} UB: {UB} LB: {LB} Sub: {sub_obj}")
        it += 1

    print("Correct Ending")

    return Pbar

main()

# %%
