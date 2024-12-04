import numpy as np
import gurobipy as gp
from gurobipy import GRB
import pandas as pd



## CLASS FOR THE INPUT DATA

class InputData():
    def __init__(self, Dem, Uti, Load_Z, Gen_E_OpCost, Gen_N_OpCost, Gen_N_MaxInvCap, Gen_E_Cap, Gen_N_InvCost, Gen_E_Tech, Gen_N_Tech, Gen_E_Z, Gen_N_Z, Gen_E_OpCap, Gen_N_OpCap, Trans_React, Trans_Cap, Trans_Line_From_Z, Trans_Line_To_Z, Trans_Z_Connected_To_Z,Gen_N_Data_scenarios,Gen_N_OpCost_scenarios):
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
        self.Gen_N_Data_scenarios = Gen_N_Data_scenarios
        self.Gen_N_OpCost_scenarios = Gen_N_OpCost_scenarios




## CLASS FOR PARAMETERS

class Parameters():
    def __init__(self, H, D, Y, N, N_dem, N_gen_E, N_gen_N, N_zone, N_line, B,R, N_S,max_deviation,epsilon, alpha, beta):
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
        self.N_S = N_S
        self.Big_M = B*(1+max_deviation)
        self.epsilon = epsilon
        self.alpha = alpha
        self.beta = beta

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



## CLASS FOR THE MARKET CLEARING PROBLEM OF MODEL1

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
        self.con.max_p_E = self.m.addConstr(self.var.p_E <= self.D.Gen_E_OpCap * (self.P.Sum_over_hours @ self.D.Gen_E_Cap.T), name='Maximum production of existing generators')

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
        n_test = self.P.N
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
        self.var.P_N = self.m.addMVar((self.P.N_gen_N), lb=0) # Invested capacity in every new generator
        self.var.p_N = self.m.addMVar((self.P.N, self.P.N_gen_N), lb=0) # Power output per hour for every new generator


    def _build_constraints(self):
        # Capacity investment constraint
        self.con.cap_inv = self.m.addConstr(self.var.P_N <= self.D.Gen_N_MaxInvCap, name='Maximum capacity investment')

        # Max production constraint
        self.con.max_p_N = self.m.addConstr(self.var.p_N <= self.D.Gen_N_OpCap * self.var.P_N, name='Maximum RES production') 

        # Budget constraint
        self.con.budget = self.m.addConstr(self.var.P_N @ self.D.Gen_N_InvCost <= self.P.B, name='Budget constraint')


    def _build_objective(self):
        revenues = ((self.var.p_N @ self.D.Gen_N_Z.T) * self.DA_Price).sum()  # don't use quicksum here because it's a <MLinExpr (3600, N_zone)>
        op_costs = gp.quicksum(self.var.p_N @ self.D.Gen_N_OpCost)
        invest_costs = self.var.P_N.T @ self.D.Gen_N_InvCost
        objective = self.P.R*(revenues - op_costs) - invest_costs
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
        
        
# Class for the robust stochastic model
class InvestmentModel_Robust():
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
        self.var.P_N = self.m.addMVar((self.P.N_gen_N), lb=0) # Invested capacity in every new generator
        self.var.p_N = self.m.addMVar((self.P.N, self.P.N_gen_N), lb=0) # Power output per hour for every new generator


    def _build_constraints(self):
        # Capacity investment constraint
        self.con.cap_inv = self.m.addConstr(self.var.P_N <= self.D.Gen_N_MaxInvCap, name='Maximum capacity investment')

        # Max production constraint
        self.con.max_p_N = self.m.addConstr(self.var.p_N <= self.D.Gen_N_OpCap * self.var.P_N, name='Maximum RES production') 

        # Budget constraint
        for s in range(self.P.N_S):
            self.con.budget = self.m.addConstr(gp.quicksum(self.var.P_N [g] * self.D.Gen_N_Data_scenarios[g,s] for g in range(self.P.N_gen_N)) <= self.P.B, name='Budget constraint')


    def _build_objective(self):
        revenues = ((self.var.p_N @ self.D.Gen_N_Z.T) * self.DA_Price).sum()  # don't use quicksum here because it's a <MLinExpr (3600, N_zone)>
        op_costs = gp.quicksum(self.var.p_N @ self.D.Gen_N_OpCost)
        invest_costs = gp.quicksum((1/self.P.N_S)*self.var.P_N[g] * self.D.Gen_N_Data_scenarios[g,s] for g in range(self.P.N_gen_N) for s in range(self.P.N_S))
        objective = self.P.R*(revenues - op_costs) - invest_costs
        self.m.setObjective(objective, GRB.MAXIMIZE)


    def _display_guropby_results(self):
        self.m.setParam('OutputFlag', self.Guroby_results)
    

    def _build_model(self):
        self.m = gp.Model('Investment problem')
        self._build_variables()  
        self._build_constraints()
        self._build_objective()
        #self._display_guropby_results()
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


#Class for the robust stochastic model
class InvestmentModel_Stochastic():
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
        self.var.P_N = self.m.addMVar((self.P.N_gen_N), lb=0) # Invested capacity in every new generator
        self.var.p_N = self.m.addMVar((self.P.N, self.P.N_gen_N,self.P.N_S), lb=0) # Power output per hour for every new generator


    def _build_constraints(self):
        # Capacity investment constraint
        self.con.cap_inv = self.m.addConstr(self.var.P_N <= self.D.Gen_N_MaxInvCap, name='Maximum capacity investment')

        # Max production constraint
        for s in range(self.P.N_S):
            self.con.max_p_N = self.m.addConstr(self.var.p_N[:,:,s] <= self.D.Gen_N_OpCap * self.var.P_N, name='Maximum RES production') 

        # Budget constraint
        for s in range(self.P.N_S):
            self.con.budget = self.m.addConstr(gp.quicksum(self.var.P_N [g] * self.D.Gen_N_Data_scenarios[g,s] for g in range(self.P.N_gen_N)) <= self.P.B, name='Budget constraint')


    def _build_objective(self):
        revenues = gp.quicksum(self.var.p_N[h,g,s] * self.D.Gen_N_Z[z,g] * self.DA_Price[h,z] for h in range(self.P.N) for g in range(self.P.N_gen_N) for s in range(self.P.N_S) for z in range(self.P.N_zone))  
        op_costs = gp.quicksum(self.var.p_N[h,g,s] * self.D.Gen_N_OpCost_scenarios[g,s] for h in range(self.P.N) for g in range(self.P.N_gen_N) for s in range(self.P.N_S))
        invest_costs = gp.quicksum(self.var.P_N[g] * self.D.Gen_N_Data_scenarios[g,s] for g in range(self.P.N_gen_N) for s in range(self.P.N_S))
        objective = (1/self.P.N_S)*self.P.R*(revenues - op_costs) - invest_costs*(1/self.P.N_S)

        self.m.setObjective(objective, GRB.MAXIMIZE)


    def _display_guropby_results(self):
        self.m.setParam('OutputFlag', self.Guroby_results)
    

    def _build_model(self):
        self.m = gp.Model('Investment problem')
        self._build_variables()  
        self._build_constraints()
        self._build_objective()
        #self._display_guropby_results()
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


#Class for chance constraint model
class InvestmentModel_ChanceConstraint():
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
        self.var.P_N = self.m.addMVar((self.P.N_gen_N), lb=0) # Invested capacity in every new generator
        self.var.p_N = self.m.addMVar((self.P.N, self.P.N_gen_N), lb=0) # Power output per hour for every new generator
        self.var.u = self.m.addMVar((self.P.N_S), vtype=GRB.BINARY) # Binary variable for each scenario


    def _build_constraints(self):
        # Capacity investment constraint
        self.con.cap_inv = self.m.addConstr(self.var.P_N <= self.D.Gen_N_MaxInvCap, name='Maximum capacity investment')

        # Max production constraint
        self.con.max_p_N = self.m.addConstr(self.var.p_N <= self.D.Gen_N_OpCap * self.var.P_N, name='Maximum RES production') 

        # Budget constraint
        for s in range(self.P.N_S):
            self.con.budget = self.m.addConstr(gp.quicksum(self.var.P_N [g] * self.D.Gen_N_Data_scenarios[g,s] for g in range(self.P.N_gen_N)) - self.P.B <= (1-self.var.u[s])*self.P.Big_M, name='Budget constraint')

        # Chance constraint
        self.con.chance = self.m.addConstr(gp.quicksum(self.var.u[s] for s in range(self.P.N_S))/self.P.N_S >= (1-self.P.epsilon), name='Chance constraint')


    def _build_objective(self):
        revenues = ((self.var.p_N @ self.D.Gen_N_Z.T) * self.DA_Price).sum()  # don't use quicksum here because it's a <MLinExpr (3600, N_zone)>
        op_costs = gp.quicksum(self.var.p_N @ self.D.Gen_N_OpCost)
        invest_costs = gp.quicksum((1/self.P.N_S)*self.var.P_N[g] * self.D.Gen_N_Data_scenarios[g,s] for g in range(self.P.N_gen_N) for s in range(self.P.N_S))
        objective = self.P.R*(revenues - op_costs) - invest_costs
        self.m.setObjective(objective, GRB.MAXIMIZE)


    def _display_guropby_results(self):
        self.m.setParam('OutputFlag', self.Guroby_results)
    

    def _build_model(self):
        self.m = gp.Model('Investment problem')
        self._build_variables()  
        self._build_constraints()
        self._build_objective()
        #self._display_guropby_results()
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


#Class for CVaR model
class InvestmentModel_CVaR():
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
        self.var.P_N = self.m.addMVar((self.P.N_gen_N), lb=0) # Invested capacity in every new generator
        self.var.p_N = self.m.addMVar((self.P.N, self.P.N_gen_N), lb=0) # Power output per hour for every new generator
        self.var.eta = self.m.addMVar((self.P.N_S), lb=0)# Eta for each scenario
        self.var.zeta = self.m.addMVar((1), lb=0) # Zeta for the CVaR


    def _build_constraints(self):
        # Capacity investment constraint
        self.con.cap_inv = self.m.addConstr(self.var.P_N <= self.D.Gen_N_MaxInvCap, name='Maximum capacity investment')

        # Max production constraint
        self.con.max_p_N = self.m.addConstr(self.var.p_N <= self.D.Gen_N_OpCap * self.var.P_N, name='Maximum RES production') 

        # Budget constraint
        for s in range(self.P.N_S):
            self.con.budget = self.m.addConstr(gp.quicksum(self.var.P_N [g] * self.D.Gen_N_Data_scenarios[g,s] for g in range(self.P.N_gen_N)) <= self.P.B, name='Budget constraint')
        # CVaR constraint
        for s in range(self.P.N_S):
            self.con.CVaR = self.m.addConstr(self.var.eta[s] >= self.var.zeta 
                                             - (((self.var.p_N @ self.D.Gen_N_Z.T) * self.DA_Price).sum() # Revenues
                             - gp.quicksum(self.var.p_N @ self.D.Gen_N_OpCost)  # Operating Costs
                             - gp.quicksum((1/self.P.N_S)*self.var.P_N[g] * self.D.Gen_N_Data_scenarios[g,s] for g in range(self.P.N_gen_N) for s in range(self.P.N_S))) #Uncertain Capex
                                                         , name='CVaR constraint')

    def _build_objective(self):
        revenues = ((self.var.p_N @ self.D.Gen_N_Z.T) * self.DA_Price).sum()  # don't use quicksum here because it's a <MLinExpr (3600, N_zone)>
        op_costs = gp.quicksum(self.var.p_N @ self.D.Gen_N_OpCost)
        invest_costs = gp.quicksum((1/self.P.N_S)*self.var.P_N[g] * self.D.Gen_N_Data_scenarios[g,s] for g in range(self.P.N_gen_N) for s in range(self.P.N_S))
        CVaR = (self.var.zeta - (1/(1-self.P.alpha))*gp.quicksum(1/self.P.N_S*self.var.eta[s] for s in range(self.P.N_S)))
        objective = (1-self.P.beta)*(self.P.R*(revenues - op_costs) - invest_costs) + self.P.beta*CVaR
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
        self.var.theta = self.m.addMVar((self.P.N,self.P.N_zone), lb= -GRB.INFINITY)  # power flow per hour for every transmission line
        # Dual variables
        self.var.DA_Price = self.m.addMVar((self.P.N, self.P.N_zone), lb=-GRB.INFINITY)  # Day ahead price per hour for every zone
        self.var.mu_E_up = self.m.addMVar((self.P.N, self.P.N_gen_E), lb=-GRB.INFINITY)  # Dual 1
        self.var.mu_E_down = self.m.addMVar((self.P.N, self.P.N_gen_E), lb=-GRB.INFINITY)  # Dual 1
        self.var.mu_N_up = self.m.addMVar((self.P.N, self.P.N_gen_N), lb=-GRB.INFINITY)  # Dual 2
        self.var.mu_N_down = self.m.addMVar((self.P.N, self.P.N_gen_N), lb=-GRB.INFINITY)  # Dual 2
        self.var.nu_up = self.m.addMVar((self.P.N, self.P.N_dem), lb=-GRB.INFINITY)  # Dual 3
        self.var.nu_down = self.m.addMVar((self.P.N, self.P.N_dem), lb=-GRB.INFINITY)  # Dual 3
        self.var.ome_up = self.m.addMVar((self.P.N, self.P.N_zone), lb=-GRB.INFINITY)  # Dual 4
        self.var.ome_down = self.m.addMVar((self.P.N, self.P.N_zone), lb=-GRB.INFINITY)  # Dual 4
        self.var.rho = self.m.addMVar((self.P.N), lb=-GRB.INFINITY)  # Dual 5
        

    def _build_constraints(self):
        # Capacity investment constraint
        self.con.cap_inv = self.m.addConstr(self.var.P_N <= self.D.Gen_N_MaxInvCap, name='Maximum capacity investment')

        # Budget constraint
        self.con.budget = self.m.addConstr(self.var.P_N.T @ self.D.Gen_N_InvCost <= self.P.B, name='Budget constraint')

        ## Primal constraints
        # Max production constraint existing
        self.con.max_p_E = self.m.addConstr(self.var.p_E <= self.D.Gen_E_OpCap * self.D.Gen_E_Cap, name='Maximum production of existing generators')

        # Max production constraint new
        self.con.max_p_N = self.m.addConstr(self.var.p_N <= self.D.Gen_N_OpCap * self.var.P_N, name='Maximum New production')

        # Max demand constraint
        self.con.max_dem = self.m.addConstr(self.var.d <= self.D.Dem, name='Maximum demand')        

        # Power flow constraints, one per transmission line
        self.con.power_flow_0 = self.m.addConstr(self.var.theta @ self.P.first_zone == self.P.voltage_angle_0, name='Initial voltage angle')
        self.Inv_Trans_React = 1/self.D.Trans_React
        Delta_theta = self.var.theta @ self.D.Trans_Line_From_Z - self.var.theta @ self.D.Trans_Line_To_Z
        self.con.power_flow = self.m.addConstr(self.P.Sum_over_hours @ self.Inv_Trans_React.T * Delta_theta <= self.P.Sum_over_hours @ self.D.Trans_Cap.T, name='Power flow constraint')

        # Balance constraint
        prod_zone = self.var.p_E @ self.D.Gen_E_Z.T
        dem_zone = self.var.d @ self.D.Load_Z.T
        trans_zone = self.P.Sum_over_hours @ self.Inv_Trans_React.T * (self.var.theta - self.var.theta @ self.D.Trans_Z_Connected_To_Z.T)
        self.con.balance = self.m.addConstr(dem_zone - prod_zone == -trans_zone, name='Power balance') 

        ## FIRST Order Conditions

        self.con.L_p_EC = (self.m.addConstr(self.D.Gen_E_OpCost - self.var.DA_Price[h] * self.D.Gen_E_Z - self.var.mu_E_up[h] + self.var.mu_E_down[h] == 0, name='L_p_EC') for h in range(self.P.N))
        self.con.L_p_NC = (self.m.addConstr(self.D.Gen_N_OpCost - self.var.DA_Price[h] * self.D.Gen_N_Z - self.var.mu_N_up[h] + self.var.mu_N_down[h] == 0, name='L_p_NC') for h in range(self.P.N))
        self.con.L_d = (self.m.addConstr(- self.D.Uti - self.var.DA_Price[h] * self.D.Load_Z - self.var.nu_up[h] + self.var.nu_down[h] == 0, name='L_d') for h in range(self.P.N))
        self.con.L_O = (self.m.addConstr(self.Inv_Trans_React*(self.var.ome_up[h] - self.var.ome_down[h]) @ self.D.Trans_Line_From_Z[0] 
                                - self.Inv_Trans_React*(self.var.ome_up[h] - self.var.ome_down[h]) @ self.D.Trans_Line_To_Z[0]
                                - self.Inv_Trans_React*self.var.DA_Price[h] * self.D.Trans_Line_From_Z[0]
                                + self.Inv_Trans_React*self.var.DA_Price[h] * self.D.Trans_Line_To_Z[0]
                                + self.var.rho[h]
                                == 0, name='L_O') for h in range(self.P.N) for z in range(self.P.N_zone))

        ## Complementary Conditions

        # # Auxiliary variables for SOS1
        # aux1 = self.m.addMVar((self.P.N, self.P.N_zone),lb=-gp.GRB.INFINITY)
        # aux2 = self.m.addMVar((self.P.N, self.P.N_zone),lb=-gp.GRB.INFINITY)
        # aux1 == - self.P.Sum_over_hours @ self.D.Trans_Cap.T - self.P.Sum_over_hours @ self.Inv_Trans_React.T * Delta_theta
        # aux2 == self.P.Sum_over_hours @ self.Inv_Trans_React.T * Delta_theta - self.P.Sum_over_hours @ self.D.Trans_Cap.T

        # for h in range(self.P.N):

        #     # Equation 3.29: μEC_g,t · pEC_g,t = 0, μEC_g,t · (P_EC_g − pEC_g,t) = 0
        #     for g in range(self.P.N_gen_E):

        #         self.m.addSOS(1, [self.var.mu_E_down[h,g], self.var.p_E[h,g]])  

        #         aux = self.m.addVar(lb=-gp.GRB.INFINITY)

        #         aux == self.D.Gen_E_OpCap[h,g] * self.D.Gen_E_Cap[g] - self.var.p_E[h,g]

        #         self.m.addSOS(1, [self.var.mu_E_up[h,g], aux])


        #     # Equation 3.30: μNC_g,t · pNC_g,t = 0, μNC_g,t · (P_NC_g − pNC_g,t) = 0
        #     for g in range(self.P.N_gen_N):

        #         self.m.addSOS(1, [self.var.mu_N_down[h,g], self.var.p_N[h,g]])  

        #         aux = self.m.addVar(lb=-gp.GRB.INFINITY)

        #         aux == self.D.Gen_N_OpCap[h,g] * self.var.P_N[g] - self.var.p_N[h,g]

        #         self.m.addSOS(1, [self.var.mu_N_up[h,g], aux])

        #     # Equation 3.31: νd,t · dd,t = 0, νd,t · (DC_d − dd,t) = 0
        #     for d in range(self.P.N_dem):

        #         self.m.addSOS(1, [self.var.nu_down[h,d], self.var.d[h,d]])  

        #         aux = self.m.addVar(lb=-gp.GRB.INFINITY)

        #         aux == self.D.Dem[h,d] - self.var.d[h,d]

        #         self.m.addSOS(1, [self.var.nu_up[h,d], aux])

        #     # Equation 3.32: ωz,m,t · linear terms for power flow

        #     for z in range(self.P.N_zone):  

        #         self.m.addSOS(1, [self.var.ome_down[h,z], aux1[h,z]])

        #         self.m.addSOS(1, [self.var.ome_up[h,z], aux2[h,z]])

        #     # Equation 3.33: ρt · θ1,t = 0
        #     self.m.addSOS(1, [self.var.rho[h]  , self.var.theta[h,0]], [1, 1])


    def _build_objective(self):
        el_1 = gp.quicksum(self.var.p_E @ self.D.Gen_E_OpCost)
        el_2 = gp.quicksum(self.D.Gen_E_OpCap[h,g]* self.D.Gen_E_Cap[g] * self.var.mu_E_up[h,g] for h in range(self.P.N) for g in range(self.P.N_gen_E))
        el_3 = gp.quicksum(self.var.d @ self.D.Uti)
        el_4 = gp.quicksum(self.D.Dem[h,d] * self.var.nu_up[h,d]  for h in range(self.P.N) for d in range(self.P.N_dem))
        el_5 = gp.quicksum(gp.quicksum((self.P.Sum_over_hours @ self.D.Trans_Cap.T) @ self.var.ome_down.T))
        el_6 = gp.quicksum(gp.quicksum((self.P.Sum_over_hours @ self.D.Trans_Cap.T) @ self.var.ome_up.T))
        op_costs = gp.quicksum(self.var.p_N @ self.D.Gen_N_OpCost)
        invest_costs = self.var.P_N.T @ self.D.Gen_N_InvCost
        objective = self.P.R*(- el_1 - el_2 + el_3 - el_4 - el_5 + el_6 - op_costs) - invest_costs
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
            

        