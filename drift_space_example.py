"""
Example of finding a second-order approximation of a drift space
"""

from accelerator_hamiltonians import positions, momenta, M_ds, a_y, a_s, ds
from sympy import Symbol, print_latex

# Define vector potentials for drift space 
A_y = 0
A_s = 0

# Integrator parameters
L = Symbol('L') # length
N = 1 # Integration steps. Doesn't matter for drift spaces since they are analytical
delta_sigma = L/N
transfer_function = [*positions, *momenta]

for i in range(N):
    transfer_function = M_ds.subs({a_y: A_y, a_s: A_s, ds: delta_sigma}) * transfer_function
    
#print(transfer_function)
print_latex(transfer_function)