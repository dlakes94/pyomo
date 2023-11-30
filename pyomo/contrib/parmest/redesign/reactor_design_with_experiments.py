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
import pandas as pd
from pyomo.environ import (
    ConcreteModel,
    Param,
    Var,
    PositiveReals,
    Objective,
    Constraint,
    maximize,
    SolverFactory,
    Suffix
)

def reactor_design_model():
    # Create the concrete model
    model = ConcreteModel()

    # Rate constants
    model.k1 = Param(initialize=5.0 / 6.0, within=PositiveReals, mutable=True)  # min^-1
    model.k2 = Param(initialize=5.0 / 3.0, within=PositiveReals, mutable=True)  # min^-1
    model.k3 = Param(initialize=1.0 / 6000.0, within=PositiveReals, mutable=True)  # m^3/(gmol min)

    # Inlet concentration of A, gmol/m^3
    model.caf = Param(initialize=10000, within=PositiveReals, mutable=True)
    
    # Space velocity (flowrate/volume)
    model.sv = Param(initialize=1.0, within=PositiveReals, mutable=True)

    # Outlet concentration of each component
    model.ca = Var(initialize=5000.0, within=PositiveReals)
    model.cb = Var(initialize=2000.0, within=PositiveReals)
    model.cc = Var(initialize=2000.0, within=PositiveReals)
    model.cd = Var(initialize=1000.0, within=PositiveReals)

    # Objective
    model.obj = Objective(expr=model.cb, sense=maximize)

    # Constraints
    model.ca_bal = Constraint(
        expr=(
            0
            == model.sv * model.caf
            - model.sv * model.ca
            - model.k1 * model.ca
            - 2.0 * model.k3 * model.ca**2.0
        )
    )

    model.cb_bal = Constraint(
        expr=(0 == -model.sv * model.cb + model.k1 * model.ca - model.k2 * model.cb)
    )

    model.cc_bal = Constraint(expr=(0 == -model.sv * model.cc + model.k2 * model.cb))

    model.cd_bal = Constraint(
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
        
        # initialization
        # discretize
        
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
        
        """
        TODO add labels, 
        change to suffix, component maps
        experiment output[expn] = float/pint.quantity (needed for DR, DOE, Parmest)
        experiment input[var/param] = UID (needed for DOE)
        unknown parameters[var/param] = UID (needed for Parmest)
        observed_input[var/param] = float (needed for DR)
        
        experiment_outputs = {m['ca']: self.data_i['ca'],
                              m['cb']: self.data_i['cb'],
                              m['cc']: self.data_i['cc'],
                              m['cd']: self.data_i['cd']}
    
        unknown_parameters = [m['k1'], m['k2'], m['k3']]
        """         
        #m.experiment_outputs = Suffix...{model variable: (float(data), unique identifier)}
        #m.unknown_parameters = Suffix...
        
        # BUILDING THE OBJECTIVE, in parmest
        #    sum(m.x.value - m.x)^2 --> current value - initial value
        
        return m
    
    def get_labeled_model(self):
        m = self.create_model()
        m = self.finalize_model()
        m = self.label_model()
        
        return m

def create_data():
    # For a range of sv values, return ca, cb, cc, and cd
    results = []
    sv_values = [1.0 + v * 0.05 for v in range(1, 20)]
    caf = 10000
    for sv in sv_values:
        model = reactor_design_model()
        model.sv = sv
        model.caf = caf
        solver = SolverFactory("ipopt")
        solver.solve(model)
        results.append([sv, caf, model.ca(), model.cb(), model.cc(), model.cd()])

    results = pd.DataFrame(results, columns=["sv", "caf", "ca", "cb", "cc", "cd"])
    print(results)
    return results


if __name__ == "__main__":
    
    data = create_data()
    
    exp_list= []
    for i in range(data.shape[0]):
        exp_list.append(Experiment(data, i))
        
    exp0_model = exp_list[0].get_labeled_model()
    print(exp0_model)
    
    import pyomo.contrib.parmest.parmest2 as parmest
    pest = parmest.Estimator(exp_list, ['k1', 'k2', 'k3'])
    obj, theta = pest.theta_est()
    