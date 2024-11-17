import numpy as np
import gurobipy as gp
from gurobipy import GRB
import pandas as pd



## CLASS FOR THE INPUT DATA

class InputData():
    def __init__(self, Dem, Uti, Gen_E_Cost, Gen_N_Cost, Gen_E_MaxC, Gen_N_MaxC, Gen_N_MaxInvCap, Gen_N_InvCost, Gen_E_Tech, Gen_N_Tech):
        self.Dem = Dem
        self.Uti = Uti
        self.Gen_E_Cost = Gen_E_Cost
        self.Gen_N_Cost = Gen_N_Cost
        self.Gen_E_MaxC = Gen_E_MaxC
        self.Gen_N_MaxC = Gen_N_MaxC
        self.Gen_N_MaxInvCap = Gen_N_MaxInvCap
        self.Gen_N_InvCost = Gen_N_InvCost
        self.Gen_E_Tech = Gen_E_Tech
        self.Gen_N_Tech = Gen_N_Tech




## CLASS FOR PARAMETERS

class Parameters():
    def __init__(self, H, D, Y, N, N_dem, N_gen_E, N_gen_N, B):
        self.H = H
        self.D = D
        self.Y = Y
        self.N = N
        self.N_dem = N_dem
        self.N_gen_E = N_gen_E
        self.N_gen_N = N_gen_N
        self.B = B
        self.Sum_over_dem = np.ones((N_dem,1)) # Vector of ones to sum the demands over hours
        self.Sum_over_gen_E = np.ones((N_gen_E,1)) # Vector of ones to sum the generation over hours
        self.Sum_over_gen_N = np.ones((N_gen_N,1)) # Vector of ones to sum the generation over hours
        self.Sum_over_hours = np.ones((N,1)) # Vector of ones to sum over hours
        self.Sum_over_hours_gen_N = np.ones((N, N_gen_N)) # Vector of ones to sum over hours and generators




## CLASS WHICH CAN HAVE ATTRIBUTES SET

class Expando(object):
    '''
        A small class which can have attributes set
    '''
    pass




## CREATE A CLASS FOR THE MARKET CLEARING PROBLEM

class MarketClearingProblem():
    def __init__(self, Parameters, Data, Model_results = 1, Guroby_results = 0):
        self.P = Parameters # Parameters
        self.D = Data # Data
        self.Model_results = Model_results
        self.Guroby_results = Guroby_results
        self.var = Expando()  # Variables
        self.con = Expando()  # Constraints
        self.res = Expando()  # Results
        self._build_model() 


    def _build_variables(self):
        # Create the variables
        self.var.d = self.m.addMVar((self.P.N, self.P.N_dem), lb=0)  # demand per hour for every load
        self.var.p_E = self.m.addMVar((self.P.N, self.P.N_gen_E), lb=0)  # power output per hour for every existing generator


    def _build_constraints(self):
        # Max demand constraint
        self.con.max_dem = self.m.addConstr(self.var.d <= self.D.Dem, name='Maximum demand')

        # Max production constraint
        self.con.max_p_E = self.m.addConstr(self.var.p_E <= self.D.Gen_E_MaxC, name='Maximum production of existing generators')

        # Balance constraint
        self.con.balance = self.m.addConstr(self.var.d @ self.P.Sum_over_dem == self.var.p_E @ self.P.Sum_over_gen_E, name='Demand balance')

    
    def _build_objective(self):
        # Objective function
        objective = gp.quicksum(self.var.d @ self.D.Uti - self.var.p_E @ self.D.Gen_E_Cost)
        self.m.setObjective(objective, GRB.MAXIMIZE)


    def _display_guropby_results(self):
        self.m.setParam('OutputFlag', self.Guroby_results)

    
    def _build_model(self):
        self.m = gp.Model('Market Clearing Problem')
        self._build_variables()
        self._build_constraints()
        self._build_objective()
        self._display_guropby_results()
        self.m.optimize()
        self._results()
        if self.Model_results == 1:
            self._extract_results()

    
    def _results(self):
        self.res.obj = self.m.objVal
        self.res.d = self.var.d.X
        self.res.p_E = self.var.p_E.X
        self.res.DA_price = self.con.balance.Pi

    def _extract_results(self):
        # Display the objective value
        print('Objective value: ', self.m.objVal)
        
        # Display the optimal values of the decision variables for the 5 first hours in a df
        n_test = 24
        self.res.df = pd.DataFrame(columns=['Hour', 'Load', 'Existing generators', 'Price'])
        self.res.df['Hour'] = np.arange(1,n_test+1)
        self.res.df['Load'] = self.var.d.X[0:n_test] @ self.P.Sum_over_dem
        self.res.df['Existing generators'] = self.var.p_E.X[0:n_test] @ self.P.Sum_over_gen_E
        self.res.df['Price'] = self.con.balance.Pi[0:n_test]
        print(self.res.df)

       


## CLASS FOR THE INVESTMENT PROBLEM

class InvestmentProblem():
    def __init__(self, Parameters, Data, DA_Price, Model_results = 1, Guroby_results = 0):
        self.D = Data  # Data
        self.P = Parameters  # Parameters
        self.DA_Price = DA_Price  # Day-ahead price
        self.Model_results = Model_results  # Display results
        self.Guroby_results = Guroby_results  # Display guroby results
        self.var = Expando()  # Variables
        self.con = Expando()  # Constraints
        self.res = Expando()  # Results
        self._build_model() 


    def _build_variables(self):
        self.var.P_N = self.m.addMVar((self.P.N_gen_N, 1), lb=0) # Invested capacity in every new generator
        self.var.p_N = self.m.addMVar((self.P.N, self.P.N_gen_N), lb=0) # Power output per hour for every new generator


    def _build_constraints(self):
        # Capacity investment constraint
        self.con.cap_inv = self.m.addConstr(self.var.P_N <= self.D.Gen_N_MaxInvCap, name='Maximum capacity investment')

        # Max production constraint
        ratio_invest = (self.var.P_N.T / self.D.Gen_N_MaxInvCap.T) # % of the maximum investment capacity invested in each new generator, size (1, N_gen_N)
        self.ratio_invest_hourly = self.P.Sum_over_hours_gen_N * ratio_invest # Create a matrix of size (N, N_gen_N) with the % of the maximum investment capacity invested in each new generator for each hour
        self.con.max_p_N = self.m.addConstr(self.var.p_N <= self.D.Gen_N_MaxC * self.ratio_invest_hourly , name='Maximum RES production')

        # Budget constraint
        self.con.budget = self.m.addConstr(self.var.P_N.T @ self.D.Gen_N_InvCost <= self.P.B, name='Budget constraint')


    def _build_objective(self):
        objective = gp.quicksum (((self.var.p_N @ self.P.Sum_over_gen_N) * self.DA_Price - self.var.p_N @ self.D.Gen_N_Cost)) + self.P.B - self.var.P_N.T @ self.D.Gen_N_InvCost
        self.m.setObjective(objective, GRB.MAXIMIZE)


    def _display_guropby_results(self):
        self.m.setParam('OutputFlag', self.Guroby_results)
    

    def _build_model(self):
        self.m = gp.Model('Investment problem')
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
        print(self.res.df)


    

        