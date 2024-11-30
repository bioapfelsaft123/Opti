import numpy as np
import gurobipy as gp
from gurobipy import GRB
import pandas as pd
from Data_handling import *
from Class import *



## CLASS FOR THE SECOND PROBLEM

class Model_2_no_zone():
    def __init__(self, ParametersObj, DataObj, Model_results = 1, Guroby_results = 1):
        self.P = ParametersObj # Parameters
        self.D = DataObj # Data
        self.Model_results = Model_results
        self.Guroby_results = Guroby_results
        self.var = Expando()  # Variables
        self.con = Expando()  # Constraints
        self.res = Expando()  # Results
        self._build_model() 


    def _build_variables(self):
        self.var.P_N = self.m.addMVar((self.P.N_gen_N), lb=0) # Invested capacity in every new generator
        self.var.d = self.m.addMVar((self.P.N, self.P.N_dem), lb=0)  # demand per hour for every load
        self.var.p_E = self.m.addMVar((self.P.N, self.P.N_gen_E), lb=0)  # power output per hour for every existing generator
        self.var.p_N = self.m.addMVar((self.P.N, self.P.N_gen_N), lb=0) # Power output per hour for every new generator
        
        # Dual variables
        self.var.DA_Price = self.m.addMVar((self.P.N, 1), lb=-GRB.INFINITY)  # Day ahead price per hour
        self.var.mu_E_up = self.m.addMVar((self.P.N, self.P.N_gen_E), lb=0)  # Dual 1
        self.var.mu_E_down = self.m.addMVar((self.P.N, self.P.N_gen_E), lb=0)  # Dual 1
        self.var.mu_N_up = self.m.addMVar((self.P.N, self.P.N_gen_N), lb=0)  # Dual 2
        self.var.mu_N_down = self.m.addMVar((self.P.N, self.P.N_gen_N), lb=0)  # Dual 2
        self.var.nu_up = self.m.addMVar((self.P.N, self.P.N_dem), lb=0)  # Dual 3
        self.var.nu_down = self.m.addMVar((self.P.N, self.P.N_dem), lb=0)  # Dual 3
        
        

    def _build_constraints(self):
        # Capacity investment constraint
        self.con.cap_inv = self.m.addConstr(self.var.P_N <= self.D.Gen_N_MaxInvCap, name='Maximum capacity investment')

        # Budget constraint
        self.con.budget = self.m.addConstr(self.var.P_N.T @ self.D.Gen_N_InvCost <= self.P.B, name='Budget constraint')

        ## PRIMAL CONSTRAINTS
        # Max production constraint existing
        self.con.max_p_E = self.m.addConstr(self.var.p_E <= self.D.Gen_E_OpCap * (self.P.Sum_over_hours @ self.D.Gen_E_Cap.T), name='Maximum production of existing generators')

        # Max production constraint new, in a different shape because we can't transpose varaibles
        self.con.max_p_N = (self.m.addConstr(self.var.p_N[h] <= self.D.Gen_N_OpCap[h] *  self.var.P_N.T, name='Maximum New production') for h in range(self.P.N))

        # Max demand constraint
        self.con.max_dem = self.m.addConstr(self.var.d <= self.D.Dem, name='Maximum demand')        

        # Balance constraint
        prod_E = self.var.p_E @ self.P.Sum_over_gen_E  
        prod_N = self.var.p_N @ self.P.Sum_over_gen_N  
        dem = self.var.d @ self.P.Sum_over_dem
        self.con.balance = self.m.addConstr(dem == prod_E + prod_N, name='Power balance') 

        ## FIRST ORDER CONDITIONS

        self.con.L_p_EC = self.m.addConstr(self.P.Sum_over_hours @ self.D.Gen_E_OpCost.T - self.var.DA_Price @ self.P.Sum_over_gen_E.T - self.var.mu_E_up + self.var.mu_E_down == 0, name='L_p_EC')
        self.con.L_p_NC = self.m.addConstr(self.P.Sum_over_hours @ self.D.Gen_N_OpCost.T - self.var.DA_Price @ self.P.Sum_over_gen_N.T - self.var.mu_E_up + self.var.mu_E_down == 0, name='L_p_NC')
        self.con.L_d = self.m.addConstr(- self.P.Sum_over_hours @ self.D.Uti.T + self.var.DA_Price @ self.P.Sum_over_dem.T - self.var.nu_up + self.var.nu_down == 0, name='L_d')
        

        ## COMPLMEENTARY CONDITIONS

        
        for h in range(self.P.N):

             # Equation 3.29: μEC_g,t · pEC_g,t = 0, μEC_g,t · (P_EC_g − pEC_g,t) = 0
             for g in range(self.P.N_gen_E):

                 self.m.addSOS(1, [self.var.mu_E_down[h,g], self.var.p_E[h,g]])  

                 aux = self.m.addVar(lb=-gp.GRB.INFINITY)

                 aux == self.D.Gen_E_OpCap[h,g] * self.D.Gen_E_Cap[g] - self.var.p_E[h,g]

                 self.m.addSOS(1, [self.var.mu_E_up[h,g], aux])


             # Equation 3.30: μNC_g,t · pNC_g,t = 0, μNC_g,t · (P_NC_g − pNC_g,t) = 0
             for g in range(self.P.N_gen_N):

                 self.m.addSOS(1, [self.var.mu_N_down[h,g], self.var.p_N[h,g]])  

                 aux = self.m.addVar(lb=-gp.GRB.INFINITY)

                 aux == self.D.Gen_N_OpCap[h,g] * self.var.P_N[g] - self.var.p_N[h,g]

                 self.m.addSOS(1, [self.var.mu_N_up[h,g], aux])

             # Equation 3.31: νd,t · dd,t = 0, νd,t · (DC_d − dd,t) = 0
             for d in range(self.P.N_dem):

                 self.m.addSOS(1, [self.var.nu_down[h,d], self.var.d[h,d]])  

                 aux = self.m.addVar(lb=-gp.GRB.INFINITY)

                 aux == self.D.Dem[h,d] - self.var.d[h,d]

                 self.m.addSOS(1, [self.var.nu_up[h,d], aux])



    def _build_objective(self):
        
        objective = 1
        self.m.setObjective(objective, GRB.MAXIMIZE)


    def _display_guropby_results(self):
        self.m.setParam('OutputFlag', self.Guroby_results)
        self.m.setParam('OutputFlag', self.Guroby_results)
        self.m.setParam('Method', 2)  # Use barrier method
        self.m.setParam('Crossover', 0)  # Skip crossover for speed
        self.m.setParam('Heuristics', 0.2)  # Enable aggressive heuristics
        self.m.setParam('MIPFocus', 1)  # Focus on finding feasible solutions
    

    def _build_model(self):
        self.m = gp.Model('Model 2')
        self._build_variables()  
        self._build_constraints()
        self._build_objective()
        self._display_guropby_results()
        self.m.optimize()
        if self.Model_results == 1:
            self._extract_results()

    def _extract_results(self):
        # Display the objective value
        print('Objective value: ', self.m.objVal)
        
        # Display the generators the model invested in, in a dataframe
        self.res.P_N = self.var.P_N.X
        self.res.P_N = self.res.P_N.reshape((self.P.N_gen_N,1))
        self.res.df = pd.DataFrame(self.D.Gen_N_Tech, columns = ['Technology'])
        self.res.df['Invested capacity (MW)'] = self.res.P_N
            

        