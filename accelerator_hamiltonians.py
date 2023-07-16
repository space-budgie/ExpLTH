from sympy import Symbol, symbols, Function, diff, Matrix, integrate, lambdify, sqrt
from explth import ExpLTH
import numpy as np

# Dependent variables of Hamiltonian
positions = Matrix(symbols('x y z s'))
momenta = Matrix(symbols('p_x p_y delta p_s'))
x, y, z, s = positions
p_x, p_y, delta, p_s = momenta

# Independent variable of Hamiltonians
ds = symbols('Delta_sigma')

# Constants placeholders
beta_0, gamma_0 = symbols('beta_0, gamma_0')

# Vector potentials placeholders
a_y = Function('a_y')(x, y, s)
a_s = Function('a_s')(x, y, s)

# S-dependent Hamiltonians
H1 = -(1/beta_0 + delta) + 1/(2*beta_0**2*gamma_0**2)*(1/beta_0+delta)**(-1)+delta/beta_0 + p_x**2/(2*(1/beta_0 + delta)) + p_s
H2_bar = p_y**2/(2*(1/beta_0+delta))
H2_Iy = -integrate(a_y, (y, 0, y))
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


def integrate_particle(vector_potential: tuple, initial_particle: tuple, beta_0_value, length, integration_steps):
    delta_sigma = length/integration_steps
    A_y, A_s = gauge_transform(*vector_potential)
    integrator = M_ds.subs({a_y: A_y, a_s: A_s, ds: delta_sigma, beta_0: beta_0_value, gamma_0: 1/sqrt(1-beta_0_value**2)}).doit()*[*positions, *momenta]

    #integrator = lambdify([*positions, *momenta], integrator)
    
    #result = np.zeros((integration_steps+1, len(initial_particle)), dtype=np.float128)
    #result[0] = initial_particle
    #for i in range(1, integration_steps+1):
    #    t = integrator(*result[i-1])
    #    result[i] = t.reshape(8,)
        
        
    return integrator


if __name__=="__main__":
    from sympy import sin, cos, sinh, cosh, pi
    B_0, k_x, k_s = symbols('B_0 k_x k_s')
    k_y = sqrt(k_s**2 + k_x**2)
    A_y = -B_0*k_s/(k_x*k_y)*sin(k_x*x)*sinh(k_y*y)*sin(k_s*s)
    A_s = -B_0*1/k_x*sin(k_x*x)*cosh(k_y*y)*cos(k_s*s)
    
    values = {B_0:0.1, k_s: 2*pi/100, k_x: 1/400000}
    vector_potential = [0, A_y.subs(values), A_s.subs(values)]
    A_y, A_s = gauge_transform(*vector_potential)
    subs = {a_y: A_y, a_s: A_s, ds: 1, beta_0: 0.999, gamma_0: 1/sqrt(1-0.999**2)}
    #result = integrate_particle(vector_potential, [0,1,0,0,0,0,0.1,0], 0.999, 100, 100)
