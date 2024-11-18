import numpy as np
import gurobipy as gp
from gurobipy import GRB
import pandas as pd



## CLASS FOR THE INPUT DATA

class InputData():
    def __init__(self, Dem, Uti, Load_Z, Gen_E_OpCost, Gen_N_OpCost, Gen_N_MaxInvCap, Gen_E_Cap, Gen_N_InvCost, Gen_E_Tech, Gen_N_Tech, Gen_E_Z, Gen_N_Z, Gen_E_OpCap, Gen_N_OpCap, Trans_React, Trans_Cap, Trans_Line_From_Z, Trans_Line_To_Z, Trans_Z_Connected_To_Z):
        self.Dem = Dem
        self.Uti = Uti
        self.Load_Z = Load_Z
        self.Gen_E_OpCost = Gen_E_OpCost
        self.Gen_N_OpCost = Gen_N_OpCost
        self.Gen_E_Cap = Gen_E_Cap
        self.Gen_N_MaxInvCap = Gen_N_MaxInvCap
        self.Gen_N_InvCost = Gen_N_InvCost
        self.Gen_E_Tech = Gen_E_Tech
        self.Gen_N_Tech = Gen_N_Tech
        self.Gen_E_Z = Gen_E_Z
        self.Gen_N_Z = Gen_N_Z
        self.Gen_E_OpCap = Gen_E_OpCap
        self.Gen_N_OpCap = Gen_N_OpCap
        self.Trans_React = Trans_React
        self.Trans_Cap = Trans_Cap
        self.Trans_Line_From_Z = Trans_Line_From_Z
        self.Trans_Line_To_Z = Trans_Line_To_Z
        self.Trans_Z_Connected_To_Z = Trans_Z_Connected_To_Z




## CLASS FOR PARAMETERS

class Parameters():
    def __init__(self, H, D, Y, N, N_dem, N_gen_E, N_gen_N, N_zone, N_line, B, R):
        self.H = H
        self.D = D
        self.Y = Y
        self.N = N
        self.N_dem = N_dem
        self.N_gen_E = N_gen_E
        self.N_gen_N = N_gen_N
        self.N_zone = N_zone
        self.N_line = N_line
        self.B = B
        self.R = R

        # Create useful vectors
        self.Sum_over_dem = np.ones((N_dem,1)) # Vector of ones to sum the demands over hours
        self.Sum_over_gen_E = np.ones((N_gen_E,1)) # Vector of ones to sum the generation over hours
        self.Sum_over_gen_N = np.ones((N_gen_N,1)) # Vector of ones to sum the generation over hours
        self.Sum_over_hours = np.ones((N,1)) # Vector of ones to sum over hours
        self.Sum_over_hours_gen_N = np.ones((N, N_gen_N)) # Vector of ones to sum over hours and generators

        # Create a matrix to access the first zone
        self.first_zone = np.zeros((N_zone, N_zone))
        self.first_zone[0, 0] = 1

        # Create a matrix to set the initial voltage angle to 0
        self.voltage_angle_0 = np.zeros((N, N_zone))

        



## CLASS WHICH CAN HAVE ATTRIBUTES SET

class Expando(object):
    '''
        A small class which can have attributes set
    '''
    pass



## CLASS FOR THE MARKET CLEARING PROBLEM OF MODEL 1

class MarketClearingModel1():
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
        # Create the variables
        self.var.d = self.m.addMVar((self.P.N, self.P.N_dem), lb=0)  # demand per hour for every load
        self.var.p_E = self.m.addMVar((self.P.N, self.P.N_gen_E), lb=0)  # power output per hour for every existing generator
        self.var.theta = self.m.addMVar((self.P.N,self.P.N_zone), lb= -1000)  # power flow per hour for every transmission line


    def _build_constraints(self):
        # Power flow constraints, one per transmission line
        self.con.power_flow_0 = self.m.addConstr(self.var.theta @ self.P.first_zone == self.P.voltage_angle_0, name='Initial voltage angle')
        self.Inv_Trans_React = 1/self.D.Trans_React
        Delta_theta = self.var.theta @ self.D.Trans_Line_From_Z - self.var.theta @ self.D.Trans_Line_To_Z
        self.con.power_flow = self.m.addConstr(self.P.Sum_over_hours @ self.Inv_Trans_React.T * Delta_theta <= self.P.Sum_over_hours @ self.D.Trans_Cap.T, name='Power flow constraint')

        # Max demand constraint
        self.con.max_dem = self.m.addConstr(self.var.d <= self.D.Dem, name='Maximum demand')

        # Max production constraint
        self.con.max_p_E = self.m.addConstr(self.var.p_E <= self.D.Gen_E_OpCap * self.D.Gen_E_Cap, name='Maximum production of existing generators')

        # Balance constraint
        prod_zone = self.var.p_E @ self.D.Gen_E_Z.T
        dem_zone = self.var.d @ self.D.Load_Z.T
        trans_zone = self.P.Sum_over_hours @ self.Inv_Trans_React.T * (self.var.theta - self.var.theta @ self.D.Trans_Z_Connected_To_Z.T)
        self.con.balance = self.m.addConstr(dem_zone - prod_zone == -trans_zone, name='Demand balance')
        
    
    def _build_objective(self):
        # Objective function
        objective = gp.quicksum(self.var.d @ self.D.Uti - self.var.p_E @ self.D.Gen_E_OpCost)
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
        
        # Display the optimal values of the decision variables for the 24 first hours in a df, looping through the zones
        n_test = 3600
        self.res.df = pd.DataFrame(columns=['Hour'])
        self.res.df['Hour'] = np.arange(1, n_test + 1)
        Load = self.var.d[0:n_test].X @ self.D.Load_Z.T
        Existing_gen = self.var.p_E[0:n_test].X @ self.D.Gen_E_Z.T
        Price = self.con.balance.Pi[0:n_test]
        Trans = self.P.Sum_over_hours[0:n_test] @ self.Inv_Trans_React.T * (self.var.theta[0:n_test].X @ self.D.Trans_Line_From_Z - self.var.theta[0:n_test].X @ self.D.Trans_Line_To_Z)
        for zone in range(self.P.N_zone):
            self.res.df['Load Zone ' + str(zone)] = Load[:, zone]
            self.res.df['Existing generators Zone ' + str(zone)] = Existing_gen[:, zone]
            self.res.df['Price Zone ' + str(zone)] = Price[:, zone]
        for line in range(self.P.N_line):
            self.res.df['Power flow line ' + str(line + 1)] = Trans[:, line]


       

## CLASS FOR THE INVESTMENT PROBLEM OF MODEL 1

class InvestmentModel1():
    def __init__(self, Parameters, Data, DA_Price, Model_results = 1, Guroby_results = 1):
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
        self.con.max_p_N = self.m.addConstr(self.var.p_N <= self.D.Gen_N_OpCap @ self.var.P_N, name='Maximum RES production')

        # Budget constraint
        self.con.budget = self.m.addConstr(self.var.P_N.T @ self.D.Gen_N_InvCost <= self.P.B, name='Budget constraint')


    def _build_objective(self):
        revenues = ((self.var.p_N @ self.D.Gen_N_Z.T) * self.DA_Price).sum()  # don't use quicksum here because it's a <MLinExpr (3600, N_zone)>
        op_costs = gp.quicksum(self.var.p_N @ self.D.Gen_N_OpCost)
        budget_init = self.P.B
        invest_costs = self.var.P_N.T @ self.D.Gen_N_InvCost
        objective = revenues - op_costs + budget_init - invest_costs
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
        


## CLASS FOR THE SECOND PROBLEM

class Model_2():
    def __init__(self, ParametersObj, DataObj, Model_results = 1, Guroby_results = 0):
        self.P = ParametersObj # Parameters
        self.D = DataObj # Data
        self.Model_results = Model_results
        self.Guroby_results = Guroby_results
        self.var = Expando()  # Variables
        self.con = Expando()  # Constraints
        self.res = Expando()  # Results
        self._build_model() 


    def _build_variables(self):
        self.var.d = self.m.addMVar((self.P.N, self.P.N_dem), lb=0)  # demand per hour for every load
        self.var.P_N = self.m.addMVar((self.P.N_gen_N, 1), lb=0) # Invested capacity in every new generator
        self.var.p_E = self.m.addMVar((self.P.N, self.P.N_gen_E), lb=0)  # power output per hour for every existing generator
        self.var.p_N = self.m.addMVar((self.P.N, self.P.N_gen_N), lb=0) # Power output per hour for every new generator
        self.var.theta = self.m.addMVar((self.P.N,self.P.N_zone))  # power flow per hour for every transmission line
        self.var
        


    def _build_constraints(self):
        # Capacity investment constraint
        self.con.cap_inv = self.m.addConstr(self.var.P_N <= self.D.Gen_N_MaxInvCap, name='Maximum capacity investment')

        # Max production constraint
        ratio_invest = (self.var.P_N.T / self.D.Gen_N_MaxInvCap.T) # % of the maximum investment capacity invested in each new generator, size (1, N_gen_N)
        self.ratio_invest_hourly = self.P.Sum_over_hours_gen_N * ratio_invest # Create a matrix of size (N, N_gen_N) with the % of the maximum investment capacity invested in each new generator for each hour
        self.con.max_p_N = self.m.addConstr(self.var.p_N <= self.D.Gen_N_OpCap * self.ratio_invest_hourly , name='Maximum RES production')

        # Budget constraint
        self.con.budget = self.m.addConstr(self.var.P_N.T @ self.D.Gen_N_InvCost <= self.P.B, name='Budget constraint')


    def _build_objective(self):
        revenues = ((self.var.p_N @ self.D.Gen_N_Z.T) * self.DA_Price).sum()  # don't use quicksum here because it's a <MLinExpr (3600, N_zone)>
        op_costs = gp.quicksum(self.var.p_N @ self.D.Gen_N_OpCost)
        budget_init = self.P.B
        invest_costs = self.var.P_N.T @ self.D.Gen_N_InvCost
        objective = self.P.R*(revenues - op_costs) + budget_init - invest_costs
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
            

        