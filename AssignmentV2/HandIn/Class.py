import numpy as np
import gurobipy as gp
from gurobipy import GRB
import pandas as pd



## CLASS FOR THE INPUT DATA

class InputData():
    def __init__(self, Dem, Uti, Load_Z, Gen_E_OpCost, Gen_N_OpCost, Gen_N_MaxInvCap, Gen_E_Cap, Gen_N_InvCost, Gen_E_Tech, Gen_N_Tech, Gen_E_Z, Gen_N_Z, Gen_E_OpCap, Gen_N_OpCap, Trans_React, Trans_Cap, Trans_Line_From_Z, Trans_Line_To_Z, Trans_Z_Connected_To_Z,Gen_N_Data_scenarios,Gen_N_OpCost_scenarios, Gen_N_Data_scenarios_train, Gen_N_OpCost_scenarios_train, Gen_N_Data_scenarios_test, Gen_N_OpCost_scenarios_test):
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
        self.Gen_N_Data_scenarios_train = Gen_N_Data_scenarios_train
        self.Gen_N_OpCost_scenarios_train = Gen_N_OpCost_scenarios_train
        self.Gen_N_Data_scenarios_test = Gen_N_Data_scenarios_test
        self.Gen_N_OpCost_scenarios_test = Gen_N_OpCost_scenarios_test




## CLASS FOR PARAMETERS

class Parameters():
    def __init__(self, H, D, Y, N, N_dem, N_gen_E, N_gen_N, N_zone, N_line, B,R, N_S,N_S_train, N_S_test, max_deviation):
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
        self.B_test_train = B # Budget for the training and testing to see more investments
        self.R = R
        self.N_S = N_S
        self.N_S_train = N_S_train
        self.N_S_test = N_S_test
        self.Big_M = B*(1+max_deviation)
        

        # Create useful vectors
        self.Sum_over_dem = np.ones((N_dem,1)) # Vector of ones to sum the demands over hours
        self.Sum_over_gen_E = np.ones((N_gen_E,1)) # Vector of ones to sum the generation over hours
        self.Sum_over_gen_N = np.ones((N_gen_N,1)) # Vector of ones to sum the generation over hours
        self.Sum_over_hours = np.ones((N,1)) # Vector of ones to sum over hours
        self.Sum_over_zones = np.ones((N_zone,1))
        self.Sum_over_hours_gen_N = np.ones((N, N_gen_N)) # Vector of ones to sum over hours and generators
        self.Sum_over_scenarios = np.ones((N_S,1)) # Vector of ones to sum over scenarios
        self.Sum_over_scenarios_test = np.ones((N_S_test,1)) # Vector of ones to sum over scenarios
        

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


       

# Combined Class for stochastic robust model, chance constraint model and CVaR model

class Test_and_Train_model():
    def __init__(self, Parameters, Data, DA_Price, Mode = 'Robust', Type = 'Train', Targets = np.ones((16,1)) ,epsilon = 0.1, Beta=0.5, Alpha=0.95, Model_results=True, Guroby_results=False):
        self.D = Data  # Data
        self.P = Parameters  # Parameters
        # self.DA_Price_reduced = DA_Price  # Day-ahead price
        self.Mode = Mode
        self.Type = Type
        self.Targets = Targets
        self.epsilon = epsilon
        self.beta = Beta
        self.alpha = Alpha
        self.Model_results = Model_results  # Display results
        self.Guroby_results = Guroby_results  # Display Gurobi results
        self.var = Expando()  # Variables
        self.con = Expando()  # Constraints
        self.res = Expando()  # Results

        # Reduce the computationnal time
        self.P.N_reduced = self.P.N // 30 # Number of hours in 1 year
        self.D.Gen_N_OpCap_reduced = self.D.Gen_N_OpCap[0:120] # Gen_N_OpCap for 1 year
        self.P.R_year = self.P.R * 30 # Conversion rate for 1 year
        self.DA_Price_reduced = DA_Price[0:120] # DA_Price for 1 year
        if self.Type == 'Train':
            self.N_S = self.P.N_S_train
            self.OPEX_Cost = self.D.Gen_N_OpCost_scenarios_train
            self.CAPEX_Cost = self.D.Gen_N_Data_scenarios_train
        if self.Type == 'Test':
            self.N_S = self.P.N_S_test
            self.OPEX_Cost = self.D.Gen_N_OpCost_scenarios_test
            self.CAPEX_Cost = self.D.Gen_N_Data_scenarios_test
        if self.Type == 'Train' and self.Mode == 'Deterministic':
            self.N_S = 1
            self.OPEX_Cost = self.D.Gen_N_OpCost_scenarios_train.mean(axis=1).reshape((self.P.N_gen_N,1))
            self.CAPEX_Cost = self.D.Gen_N_Data_scenarios_train.mean(axis=1).reshape((self.P.N_gen_N,1))
            
        self._build_model()
        

    def _build_variables(self):
        self.var.P_N = self.m.addMVar((self.P.N_gen_N), lb=0, name="Invested_Capacity")  
        self.var.p_N = self.m.addMVar((self.P.N_reduced, self.P.N_gen_N, self.N_S), lb=0, name="Power_Output") 

        if self.Mode == 'Chance':
            self.var.u = self.m.addMVar((self.N_S), vtype=GRB.BINARY) # Binary variable for each scenario

        if self.Mode == 'CVaR' and self.Type == 'Train':
            self.var.eta = self.m.addMVar((self.N_S), lb=0, name="Eta")  
            self.var.zeta = self.m.addVar(lb=-GRB.INFINITY, name="Zeta")  

    def _build_constraints(self):
        # Capacity investment constraint
        self.con.cap_inv = self.m.addConstr(self.var.P_N <= self.D.Gen_N_MaxInvCap, name='Max_Investment_Capacity')
        
        # Max production constraints for all scenarios
        self.con.max_p_N = self.m.addConstrs(
            (self.var.p_N[:, :, s] <= self.D.Gen_N_OpCap_reduced * self.var.P_N for s in range(self.N_S)), name="Max_Production")
        
        # Budget constraint 
        if self.Mode == 'Robust' or self.Mode == 'CVaR' or self.Mode == 'Deterministic':
            if self.Type == 'Train':
                for s in range(self.N_S):
                    self.con.budget = self.m.addConstr(self.var.P_N @ self.CAPEX_Cost[:,s] <= self.P.B_test_train, name='Budget constraint')
        
        if self.Mode == 'Chance':
            if self.Type == 'Test':
                self.epsilon = 1
            if self.Type == 'Train':
                for s in range(self.N_S):
                    self.con.budget = self.m.addConstr(self.var.P_N @ self.CAPEX_Cost[:,s] - self.P.B_test_train <= (1-self.var.u[s])*self.P.Big_M, name='Budget constraint')
                self.con.chance = self.m.addConstr( (self.var.u).sum() /self.N_S >= (1-self.epsilon), name='Chance constraint')
                
        if self.Mode == 'CVaR' and self.Type == 'Train':
            # CVaR constraints 
            for s in range(self.N_S):
                revenues = (self.var.p_N[:, :, s] @ self.D.Gen_N_Z.T * self.DA_Price_reduced).sum()
                op_costs = (self.var.p_N[:, :, s] @ self.OPEX_Cost[:, s]).sum()
                invest_costs = self.var.P_N @ self.CAPEX_Cost[:, s]
                remaining_budget = self.P.B_test_train -self.var.P_N @ self.CAPEX_Cost.mean(axis=1) 
                
                self.con.CVaR = self.m.addConstr(
                    self.var.eta[s] >= self.var.zeta - (self.P.R_year * (revenues - op_costs) + remaining_budget),
                    name=f"CVaR_{s}")
        
        if self.Type == 'Test':
            self.con.targets = self.m.addConstr(self.var.P_N == self.Targets, name='Targets')
            
            
    def _build_objective(self):
        # Calculate components of the objective
        revenues = sum((self.var.p_N[:, :, s] @ self.D.Gen_N_Z.T * self.DA_Price_reduced).sum() for s in range(self.N_S))
        op_costs = sum((self.var.p_N[:, :, s] @ self.OPEX_Cost[:, s]).sum() for s in range(self.N_S))
        invest_costs = self.var.P_N @ self.CAPEX_Cost.mean(axis=1)  # Average costs
        remaining_budget = self.P.B_test_train -self.var.P_N @ self.CAPEX_Cost.mean(axis=1) 
        
        if self.Mode == 'CVaR' and self.Type == 'Train':
            self.CVaR = self.var.zeta - (1 / (1 - self.alpha)) * gp.quicksum(self.var.eta[s] / self.N_S for s in range(self.N_S))
            self.expected_result = (1 / self.N_S) * (self.P.R_year * (revenues - op_costs) + remaining_budget) 
            objective = (1 - self.beta) * self.expected_result + self.beta * self.CVaR

        else:
            objective = self.P.R_year * (revenues - op_costs) + remaining_budget


        self.m.setObjective(objective, GRB.MAXIMIZE)

    def _set_solver_parameters(self):
        self.m.setParam('Presolve', 2)  # Moderate presolve to speed up model setup
        self.m.setParam('Threads', 4)  # Use 4 threads (adjust for your system)

    def _build_model(self):
        self.m = gp.Model('Optimized_Investment_Model')
        self._build_variables()
        self._build_constraints()
        self._build_objective()
        self._set_solver_parameters()
        self.m.optimize()
        if self.Model_results:
            self._extract_results()

    def _extract_results(self):
        print(f'Objective Value: {self.m.objVal}')

        if self.Mode == 'CVaR' and self.Type == 'Train':
            self.res.CVaR = self.CVaR.getValue()
            self.res.Expected_Result = self.expected_result.getValue()
        
        self.res.P_N = self.var.P_N.X
        self.res.df = pd.DataFrame(self.D.Gen_N_Tech, columns=['Technology'])
        self.res.df['Invested Capacity (MW)'] = self.res.P_N
        self.res.rem_budget = self.P.B_test_train - (self.var.P_N.X @ self.CAPEX_Cost.mean(axis=1)).sum()

        if self.Mode == 'Deterministic' and self.Type == 'Train':
            self.res.CAPEX_Cost = self.CAPEX_Cost
            self.res.OPEX_Cost = self.OPEX_Cost

        #Assumed Budget
        self.res.budget = self.P.B_test_train 

        #Assumed demand and generation
        self.res.Demand = self.D.Dem
        self.res.Generation = self.D.Gen_N_OpCap

        self.res.objective_values = []
        self.res.violated_budget = []
        for s in range(self.N_S):
            revenues = (self.var.p_N[:, :, s].X @ self.D.Gen_N_Z.T * self.DA_Price_reduced).sum()
            op_costs = (self.var.p_N[:, :, s].X @ self.OPEX_Cost[:, s]).sum()
            invest_costs = (self.var.P_N.X @ self.CAPEX_Cost[:, s]).sum()
            remaining_budget = self.P.B_test_train - (self.var.P_N.X @ self.CAPEX_Cost[:, s]).sum()
            if remaining_budget < 0:
                self.res.violated_budget.append(1)
            if remaining_budget >= 0:
                self.res.violated_budget.append(0)
            objective = (self.P.R_year * (revenues - op_costs) + remaining_budget)/1000000000
            self.res.objective_values.append(objective)
            self.res.standard_deviation = np.std(self.res.objective_values)
        
       
        
# Class for bilevel model implementation
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
        self.var.P_N = self.m.addMVar((self.P.N_gen_N, 1), lb=0) # Invested capacity in every new generator
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
        self.con.max_p_N = self.m.addConstrs(self.var.p_N[h, g] <= (self.D.Gen_N_OpCap[h, g] *  self.var.P_N[g].T) for h in range(self.P.N) for g in range(self.P.N_gen_N))

        # Max demand constraint
        self.con.max_dem = self.m.addConstr(self.var.d <= self.D.Dem, name='Maximum demand')        

        # Balance constraint
        prod_E = self.var.p_E @ self.P.Sum_over_gen_E  
        prod_N = self.var.p_N @ self.P.Sum_over_gen_N  
        dem = self.var.d @ self.P.Sum_over_dem
        self.con.balance = self.m.addConstr(dem == prod_E + prod_N, name='Power balance') 

        # ## FIRST ORDER CONDITIONS

        self.con.L_p_EC = self.m.addConstr(self.P.Sum_over_hours @ self.D.Gen_E_OpCost.T + (self.var.DA_Price @ self.P.Sum_over_gen_E.T)/self.P.N_gen_E - self.var.mu_E_down + self.var.mu_E_up == 0, name='L_p_EC')
        self.con.L_p_NC = self.m.addConstr(self.P.Sum_over_hours @ self.D.Gen_N_OpCost.T + (self.var.DA_Price @ self.P.Sum_over_gen_N.T)/self.P.N_gen_N - self.var.mu_N_down + self.var.mu_N_up == 0, name='L_p_NC')
        self.con.L_d = self.m.addConstr(- self.P.Sum_over_hours @ self.D.Uti.T - (self.var.DA_Price @ self.P.Sum_over_dem.T)/self.P.N_dem - self.var.nu_down + self.var.nu_up == 0, name='L_d')
        

        # ## COMPLMEENTARY CONDITIONS

        # # Define the Big M matrixes

        Big_M_p = np.full((self.P.N, self.P.N_gen_E), 1e9)  # Big M for the production constraints
        Big_M_d = np.full((self.P.N, self.P.N_dem), 1e9)  # Big M for the demand constraints

        # # Define the binary variables

        b_E_down = self.m.addVars(self.P.N, self.P.N_gen_E, vtype=gp.GRB.BINARY, name="b_E_down")
        b_E_up = self.m.addVars(self.P.N, self.P.N_gen_E, vtype=gp.GRB.BINARY, name="b_E_up")

        b_N_down = self.m.addVars(self.P.N, self.P.N_gen_N, vtype=gp.GRB.BINARY, name="b_N_down")
        b_N_up = self.m.addVars(self.P.N, self.P.N_gen_N, vtype=gp.GRB.BINARY, name="b_N_up")

        b_d_down = self.m.addVars(self.P.N, self.P.N_dem, vtype=gp.GRB.BINARY, name="b_d_down")
        b_d_up = self.m.addVars(self.P.N, self.P.N_dem, vtype=gp.GRB.BINARY, name="b_d_up")

        # Define the complementary constraints
        # Existing generators
        self.con.compl_E_down_mu = self.m.addConstrs((self.var.mu_E_down[h, g] <= Big_M_p[h, g] * b_E_down[h, g] for h in range(self.P.N) for g in range(self.P.N_gen_E)), name='compl_E_down_mu')
        self.con.compl_E_down_p = self.m.addConstrs((self.var.p_E[h, g] <= Big_M_p[h, g] * (1 - b_E_down[h, g]) for h in range(self.P.N) for g in range(self.P.N_gen_E)), name='compl_E_down_p')
        self.con.compl_E_up_mu = self.m.addConstrs((self.var.mu_E_up[h, g] <= Big_M_p[h, g] * b_E_up[h, g] for h in range(self.P.N) for g in range(self.P.N_gen_E)), name='compl_E_up_mu')
        self.con.compl_E_up_p = self.m.addConstrs(((self.D.Gen_E_OpCap[h, g] * self.D.Gen_E_Cap[g,0] - self.var.p_E[h, g]) <= Big_M_p[h, g] * (1 - b_E_up[h, g]) for h in range(self.P.N) for g in range(self.P.N_gen_E)), name='compl_E_up_p')

        # New generators
        self.con.compl_N_down_mu = self.m.addConstrs((self.var.mu_N_down[h, g] <= Big_M_p[h, g] * b_N_down[h, g] for h in range(self.P.N) for g in range(self.P.N_gen_N)), name='compl_N_down_mu')
        self.con.compl_N_down_p = self.m.addConstrs((self.var.p_N[h, g] <= Big_M_p[h, g] * (1 - b_N_down[h, g]) for h in range(self.P.N) for g in range(self.P.N_gen_N)), name='compl_N_down_p')
        self.con.compl_N_up_mu = self.m.addConstrs((self.var.mu_N_up[h, g] <= Big_M_p[h, g] * b_N_up[h, g] for h in range(self.P.N) for g in range(self.P.N_gen_N)), name='compl_N_up_mu')
        self.con.compl_N_up_p = self.m.addConstrs(((self.D.Gen_N_OpCap[h, g] *  self.var.P_N[g].T - self.var.p_N[h, g]) <= Big_M_p[h, g] * (1 - b_N_up[h, g]) for h in range(self.P.N) for g in range(self.P.N_gen_N)), name='compl_N_up_p')

        # Demand
        self.con.compl_d_down_mu = self.m.addConstrs((self.var.nu_down[h, d] <= Big_M_d[h, d] * b_d_down[h, d] for h in range(self.P.N) for d in range(self.P.N_dem)), name='compl_d_down_mu')
        self.con.compl_d_down_p = self.m.addConstrs((self.var.d[h, d] <= Big_M_d[h, d] * (1 - b_d_down[h, d]) for h in range(self.P.N) for d in range(self.P.N_dem)), name='compl_d_down_p')
        self.con.compl_d_up_mu = self.m.addConstrs((self.var.nu_up[h, d] <= Big_M_d[h, d] * b_d_up[h, d] for h in range(self.P.N) for d in range(self.P.N_dem)), name='compl_d_up_mu')
        self.con.compl_d_up_p = self.m.addConstrs(((self.D.Dem[h, d] - self.var.d[h, d]) <= Big_M_d[h, d] * (1 - b_d_up[h, d]) for h in range(self.P.N) for d in range(self.P.N_dem)), name='compl_d_up_p')


    # def _build_objective(self): # without linearisation
    #     self.revenues = (self.var.p_N * (self.var.DA_Price @ self.P.Sum_over_gen_N.T)).sum()  # don't use quicksum here because it's a <MLinExpr (3600, N_zone)>
    #     self.op_costs = gp.quicksum(self.var.p_N @ self.D.Gen_N_OpCost)
    #     self.invest_costs = self.var.P_N.T @ self.D.Gen_N_InvCost
    #     objective = self.P.R*(self.revenues - self.op_costs) - self.invest_costs
    #     self.m.setObjective(objective, GRB.MAXIMIZE)

    def _build_objective(self): # with linearisation

        # 1. Costs related to existing generators (G_EC)
        cost_EC_p = (self.var.p_E @ self.D.Gen_E_OpCost).sum()
        cost_EC_mu = ((self.D.Gen_E_OpCap * (self.P.Sum_over_hours @ self.D.Gen_E_Cap.T)) * self.var.mu_E_up).sum()

        # 2. Benefits and costs related to demand (D)
        benefit_d = (self.var.d @ self.D.Uti).sum()
        cost_nu_d = (self.D.Dem * self.var.nu_up).sum()

        # 3. Costs related to new generators (G_NC)
        cost_NC_p = (self.var.p_N @ self.D.Gen_N_OpCost).sum()
        cost_NC_P = (self.var.P_N.T @ self.D.Gen_N_InvCost).sum()

        # Full objective function
        objective = self.P.R * (- cost_EC_p - cost_EC_mu  # Existing generators
            + benefit_d - cost_nu_d   # Demand
            - cost_NC_p) - cost_NC_P
        
        self.m.setObjective(objective, gp.GRB.MAXIMIZE)

        


    def _display_guropby_results(self):
        self.m.setParam('OutputFlag', self.Guroby_results)
        #self.m.setParam('Method', 2)  # Use barrier method
        #self.m.setParam('Crossover', 0)  # Skip crossover for speed
        #self.m.setParam('Heuristics', 0.2)  # Enable aggressive heuristics
        #self.m.setParam('MIPFocus', 1)  # Focus on finding feasible solutions
    

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
        self.res.P_N = np.round(self.var.P_N.X, 2)
        self.res.P_N = self.res.P_N.reshape((self.P.N_gen_N,1))
        self.res.df = pd.DataFrame(self.D.Gen_N_Tech, columns = ['Technology'])
        self.res.df['Invested capacity (MW)'] = self.res.P_N
        remaining_budget = self.P.B - (self.var.P_N.T.X @ self.D.Gen_N_InvCost).sum()

        # 1. Costs related to existing generators (G_EC)
        cost_EC_p = (self.var.p_E.X @ self.D.Gen_E_OpCost).sum()
        cost_EC_mu = ((self.D.Gen_E_OpCap * (self.P.Sum_over_hours @ self.D.Gen_E_Cap.T)) * self.var.mu_E_up.X).sum()
        

        # 2. Benefits and costs related to demand (D)
        benefit_d = (self.var.d.X @ self.D.Uti).sum()
        cost_nu_d = (self.D.Dem * self.var.nu_up.X).sum()

        # 3. Costs related to new generators (G_NC)
        cost_NC_p = (self.var.p_N.X @ self.D.Gen_N_OpCost).sum()
        cost_NC_P = (self.var.P_N.T.X @ self.D.Gen_N_InvCost).sum()

        # Full objective function
        objective = self.P.R * (- cost_EC_p - cost_EC_mu  # Existing generators
            + benefit_d - cost_nu_d   # Demand
            - cost_NC_p) - cost_NC_P


        objective_val = (objective + remaining_budget)/1000000000
        self.res.objective_values = objective_val
        self.res.remaining_budget = remaining_budget
        
            

        