from sympy import Function, Matrix, diff, Expr


class ExpLTH(Function):
    """
    The exponent of a Lie transform of a Hamiltonian which is only a function of position and momenta.
    This way, the operator is exactly given by a first order series expansion:
    e^(t:H:)*F=F+t:H:F
    """

    def __init__(self, variable: Expr, hamiltonian: Expr, position_vars, momentum_vars):

        h_vars = hamiltonian.free_symbols.intersection(
            [*position_vars, *momentum_vars])
        self.h_q_dependence = h_vars.issubset(position_vars)
        self.h_p_dependence = h_vars.issubset(momentum_vars)

        # Check notes for detailed math. In short, if H depends on momentum we want to check F dependence on position
        # To check if the operator is equal to identity, and visa versa with H dependence on position.
        if self.h_p_dependence:
            self.F_dependence = set(position for idx, position in enumerate(
                position_vars) if momentum_vars[idx] in h_vars)
            self.d_hamiltonian = - \
                Matrix([diff(hamiltonian, p) for p in momentum_vars])
        elif self.h_q_dependence:
            self.F_dependence = set(momentum for idx, momentum in enumerate(
                momentum_vars) if position_vars[idx] in h_vars)
            self.d_hamiltonian = Matrix(
                [diff(hamiltonian, q) for q in position_vars])
        else:
            raise ValueError(
                "Hamiltonian is not a function of only position or momenta.")

        self.hamiltonian = hamiltonian
        self.variable = variable
        self.position_vars = position_vars
        self.momentum_vars = momentum_vars

    def copy_replace_variable(self, new_variable):
        return ExpLTH(new_variable, *self.args[1:])

    def __mul__(self, F: Expr):
        # check if iterable
        try:
            result = []
            for i in F:
                result.append(self * i)
            return Matrix(result)

        except TypeError:  # It is not iterable. Continue as normal
            pass

        if isinstance(F, ExpLTH):
            return ProductExpLTH(self, F)

        non_zero_vars = F.free_symbols.intersection(self.F_dependence)

        if not non_zero_vars:
            # Identity operator
            return F

        vars_to_diff = self.momentum_vars if self.h_q_dependence else self.position_vars

        return F + self.variable * self.d_hamiltonian.dot(
            Matrix(
                [diff(F, var) if var in non_zero_vars else 0 for var in vars_to_diff])
        )

    def __pow__(self, other):
        if not isinstance(other, int):
            raise TypeError("ExpLTH raised to the power of a non-integer")
        elif other < 0:
            raise ValueError(
                "ExpLTH raised to the power of a non-positive integer")

        if other == 0:
            return 1  # Identity
        else:
            return ProductExpLTH(self * other)

    def __str__(self):
        return f"exp({self.variable}:{self.hamiltonian}:)"

    def _latex(self, printer):
        var = printer._print(self.variable)
        H = printer._print(self.hamiltonian)

        return r"\operatorname{exp}{\left( %s:%s: \right)}" % (var, H)


class ProductExpLTH(Function):
    """
    Class to handle multiple ExpLTHs multiplied in a row.
    In this case, we keep storing them until we execute it on something.
    """

    def __init__(self, *args: ExpLTH):
        self.terms = list(args)

    def __str__(self):
        return ''.join(i.__str__() + ' * ' for i in self.terms).rstrip(' * ')

    def _latex(self, printer):
        return ''.join(i._latex(printer) for i in self.terms)

    def copy_replace_variable(self, new_variable):
        return ProductExpLTH(*[term.copy_replace_variable(new_variable) for term in self.terms])

    def __mul__(self, other):
        if isinstance(other, ProductExpLTH):
            return ProductExpLTH(*self.terms, *other.terms)
        if isinstance(other, ExpLTH):
            return ProductExpLTH(*self.terms, other)

        result = other
        for i in self.terms[::-1]:
            result = i * result

        return result

    def __pow__(self, other):
        if not isinstance(other, int):
            raise TypeError("ExpLTH raised to the power of a non-integer")
        elif other < 0:
            raise ValueError(
                "ExpLTH raised to the power of a non-positive integer")

        if other == 0:
            return 1  # Identity
        else:
            return ProductExpLTH(*(self.terms * other))
