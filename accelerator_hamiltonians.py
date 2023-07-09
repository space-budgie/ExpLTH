from sympy import Symbol, symbols, Function, diff, Matrix, integrate
#from sympy.physics.units.quantities import Quantity
from ExpLTH import ExpLTH

# Dependent variables of Hamiltonian
positions = Matrix(symbols('x y z s'))
momenta = Matrix(symbols('p_x p_y \\delta p_s'))
x, y, z, s = positions
p_x, p_y, delta, p_s = momenta

# Independent variable of Hamiltonians
ds = Symbol('\\Delta\\sigma')

# Constants placeholders
beta_0, gamma_0 = symbols('\\beta_0 \\gamma_0', constant=True)
#beta_0 = Quantity('beta_0', latex_repr="\\beta_0")
#gamma_0 = Quantity('gamma_0', latex_repr="\\gamma_0")

# Vector potentials placeholders
a_y = Function('a_y')(x, y, s)
a_s = Function('a_s')(x, y, s)

# Hamiltonians
H1 = -(1/beta_0 + delta) + 1/(2*beta_0**2*gamma_0**2)*(1/beta_0+delta)**(-1)+delta/beta_0 + p_x**2/(2*(1/beta_0 + delta)) + p_s
H2_bar = p_y**2/(2*(1/beta_0+delta))
H2_Iy = integrate(a_y, (y, 0, y))
H3 = -1*a_s

# Only used for testing, depends on both momenta and position and therefore not integrable
H2 = (p_y-a_y)**2/(2*(1/beta_0+delta))

# The full second-order integrator
M_ds = \
    ExpLTH(-ds/4, H1, positions, momenta) * ExpLTH(-ds/2, H3, positions, momenta) * ExpLTH(-ds/4, H1, positions, momenta) \
    * ExpLTH(1, H2_Iy, positions, momenta) * ExpLTH(-ds, H2_bar, positions, momenta) * ExpLTH(-1, H2_Iy, positions, momenta) \
    * ExpLTH(-ds/4, H1, positions, momenta) * ExpLTH(-ds/2, H3, positions, momenta) * ExpLTH(-ds/4, H1, positions, momenta)


def gauge_transform(A_x: Function, A_y: Function, A_s: Function):
    "Transform into a gauge where x-component of vector potential is 0"
    gauge = integrate(A_x, (x, 0, x))
    # x_component is always 0 here by fundamental theorem of calculus
    return A_y - diff(gauge, y), A_s - diff(gauge, s)


def get_transfer_function(vector_potential: tuple, length, integration_steps):
    "Integrate and find the transfer function for all positions and momenta variables"
    A_y, A_s = gauge_transform(*vector_potential)
    delta_sigma = length/integration_steps
    return M_ds.subs({a_y: A_y, a_s: A_s, ds: delta_sigma})**integration_steps * [*positions, *momenta]