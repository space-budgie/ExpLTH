from explth import ProductExpLTH


def get_yoshida_integrator(prev_integrator: ProductExpLTH, prev_integrator_order, independent_variable, order_increase, ret_all=False):
    "Increase a known integrator of a specific order by a specified amount of orders"

    if order_increase == 0 or order_increase % 2 != 0 or prev_integrator_order % 2 != 0:
        raise ValueError("Specified order has to be an even number")

    if ret_all:
        ret = [prev_integrator]

    integrator = prev_integrator

    for n in range(prev_integrator_order, order_increase+prev_integrator_order, 2):
        x_0 = -2**(1/(n+1)) / (2-2**(1/(n+1)))
        x_1 = 1 / (2-2**(1/(n+1)))
        integrator = integrator.copy_replace_variable(x_1 * independent_variable) * integrator.copy_replace_variable(
            x_0 * independent_variable) * integrator.copy_replace_variable(x_1 * independent_variable)

        if ret_all:
            ret.append(integrator)

    if ret_all:
        return ret
    else:
        return integrator


def simplify_hamiltonians(hamiltonians, substitutions):
    return [h.subs(substitutions).doit() for h in hamiltonians]
