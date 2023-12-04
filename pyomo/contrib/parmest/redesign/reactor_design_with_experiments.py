#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2022
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
"""
Continuously stirred tank reactor model, based on
pyomo/examples/doc/pyomobook/nonlinear-ch/react_design/ReactorDesign.py
"""
from os.path import join, abspath, dirname
from itertools import product
import pandas as pd

import pyomo.environ as pyo
import pyomo.contrib.parmest.parmest_redesign as parmest

def reactor_design_model():
    # Create the concrete model
    model = pyo.ConcreteModel()

    # Rate constants
    model.k1 = pyo.Param(initialize=5.0 / 6.0, within=pyo.PositiveReals, mutable=True)  # min^-1
    model.k2 = pyo.Param(initialize=5.0 / 3.0, within=pyo.PositiveReals, mutable=True)  # min^-1
    model.k3 = pyo.Param(initialize=1.0 / 6000.0, within=pyo.PositiveReals, mutable=True)  # m^3/(gmol min)

    # Inlet concentration of A, gmol/m^3
    model.caf = pyo.Param(initialize=10000, within=pyo.PositiveReals, mutable=True)
     
    # Space velocity (flowrate/volume)
    model.sv = pyo.Param(initialize=1.0, within=pyo.PositiveReals, mutable=True)

    # Outlet concentration of each component
    model.ca = pyo.Var(initialize=5000.0, within=pyo.PositiveReals)
    model.cb = pyo.Var(initialize=2000.0, within=pyo.PositiveReals)
    model.cc = pyo.Var(initialize=2000.0, within=pyo.PositiveReals)
    model.cd = pyo.Var(initialize=1000.0, within=pyo.PositiveReals)

    # Objective
    model.obj = pyo.Objective(expr=model.cb, sense=pyo.maximize)

    # Constraints
    model.ca_bal = pyo.Constraint(
        expr=(
            0
            == model.sv * model.caf
            - model.sv * model.ca
            - model.k1 * model.ca
            - 2.0 * model.k3 * model.ca**2.0
        )
    )

    model.cb_bal = pyo.Constraint(
        expr=(0 == -model.sv * model.cb + model.k1 * model.ca - model.k2 * model.cb)
    )

    model.cc_bal = pyo.Constraint(expr=(0 == -model.sv * model.cc + model.k2 * model.cb))

    model.cd_bal = pyo.Constraint(
        expr=(0 == -model.sv * model.cd + model.k3 * model.ca**2.0)
    )

    return model
            
class Experiment(object):
    
    def __init__(self, data, experiment_number):    
        self.data = data
        self.experiment_number = experiment_number
        self.data_i = data.loc[experiment_number,:]
        self.model = None
    
    def create_model(self):
        self.model = m = reactor_design_model()
        return m
    
    def finalize_model(self):
        m = self.model
        
        # Experiment inputs values
        m.sv = self.data_i['sv']
        m.caf = self.data_i['caf']
        
        # Experiment output values
        m.ca = self.data_i['ca']
        m.cb = self.data_i['cb']
        m.cc = self.data_i['cc']
        m.cd = self.data_i['cd']
        
        return m

    def label_model(self):
        m = self.model
        
        m.experiment_outputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.experiment_outputs.update([(m.ca, self.data_i['ca'])])
        m.experiment_outputs.update([(m.cb, self.data_i['cb'])])
        m.experiment_outputs.update([(m.cc, self.data_i['cc'])])
        m.experiment_outputs.update([(m.cd, self.data_i['cd'])])
        
        m.unknown_parameters = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.unknown_parameters.update((k, pyo.ComponentUID(k)) 
                                    for k in [m.k1, m.k2, m.k3])

        return m
    
    def get_labeled_model(self):
        m = self.create_model()
        m = self.finalize_model()
        m = self.label_model()
        
        return m

if __name__ == "__main__":
    
    # Read in data
    file_dirname = dirname(abspath(str(__file__)))
    file_name = abspath(join(file_dirname, "reactor_data.csv"))
    data = pd.read_csv(file_name)
    
    # Create an experiment list
    exp_list= []
    for i in range(data.shape[0]):
        exp_list.append(Experiment(data, i))
    
    # View one model
    #exp0_model = exp_list[0].get_labeled_model()
    #print(exp0_model.pprint())

    pest = parmest.Estimator(exp_list, 'SSE')

    # Parameter estimation with covariance
    obj, theta, cov = pest.theta_est(calc_cov=True, cov_n=17)
    print(obj)
    print(theta)

    # Bootstrapping
    bootstrap_theta = pest.theta_est_bootstrap(10)
    print(bootstrap_theta)
    
    # Leave N out
    lNo_theta = pest.theta_est_leaveNout(1)
    print(lNo_theta.head())  
    
    # Leave N out with bootstrapping
    lNo = 7
    lNo_samples = 5
    bootstrap_samples = 10
    dist = "MVN"
    alphas = [0.7, 0.8, 0.9]
    results = pest.leaveNout_bootstrap_test(
        lNo, lNo_samples, bootstrap_samples, dist, alphas, seed=524
    )
    print(results)

    # Calculate the objective at known theta values
    k1 = [0.8, 0.85, 0.9]
    k2 = [1.6, 1.7]
    k3 = [0.00016, 0.000165, 0.00017]
    theta_vals = pd.DataFrame(list(product(k1, k2, k3)), 
                              columns=["k1", "k2", "k3"])
    obj_at_theta = pest.objective_at_theta(theta_vals)
    print(obj_at_theta)
    
    # Likelhood ration test
    LR = pest.likelihood_ratio_test(obj_at_theta, obj, [0.8, 0.85, 0.9, 0.95])
    print(LR)
    
    # Confidence region test
    CR = pest.confidence_region_test(bootstrap_theta, "MVN", [0.5, 0.75, 1.0])
    print(CR)
    