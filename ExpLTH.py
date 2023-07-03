from sympy import Symbol, Function, Matrix, diff, symbols, Expr


class ExpLTH(Function):
    """
    The exponent of a Lie transform of a Hamiltonian which is only a function of position and momenta.
    This way, the operator is exactly given by a first order series expansion:
    e^(-t:H:)=1-t:H:
    """
    def __init__(self, variable: Expr, hamiltonian: Function, position_vars, momentum_vars):
        if not (hamiltonian.free_symbols.issubset(momentum_vars) or hamiltonian.free_symbols.issubset(position_vars)):
            raise ValueError("Hamiltonian is function of both position and conjugate momenta.")
            
        self.hamiltonian = hamiltonian
        self.variable = variable
        self.dp_hamiltonian = Matrix([diff(hamiltonian, p) for p in momentum_vars])
        self.dq_hamiltonian = Matrix([diff(hamiltonian, q) for q in position_vars])
        self.position_vars = position_vars
        self.momentum_vars = momentum_vars
        
        
    def __str__(self):
        return f"exp(-{self.variable}:{self.hamiltonian}:)"


    def _latex(self, printer):
        var, H = [printer._print(i) for i in self.args][0:2]
        
        var = '-' + var
        if var.startswith('--'):
            var = var.lstrip('--') # minus signs cancel out
        if var == '1':
            var = ''
        elif var == '-1':
            var = '-'

        return r"\operatorname{exp}{\left( %s:%s: \right)}" % (var, H)
    
    
    #def __mul__(self, other):
    #    return (tensorproduct(self.dq_hamiltonian,derive_by_array(other, momenta))     
    #                                    - tensorproduct(self.dp_hamiltonian, derive_by_array(other, positions)))
    
    def __mul__(self, other):
        # I think there might be a cooler way to do this with tensors, maybe??
        
        # check if iterable
        try:
            result = []
            for i in other:
                result.append(self * i)
            return Matrix(result)
        except TypeError: # It is not iterable
            if isinstance(other, ExpLTH):
                return ProductExpLTH(self, other)
            
            return other - self.variable * (
                self.dq_hamiltonian.dot(
                    Matrix([diff(other, p) for p in self.momentum_vars]))
                - self.dp_hamiltonian.dot(
                    Matrix([diff(other, q) for q in self.position_vars]))
                )
        

class ProductExpLTH(Function):
    """
    Class to handle multiple ExpLTHs multiplied in a row.
    In this case, we keep storing them until we execute it on something else.
    """
    
    def __init__(self, *args: ExpLTH):
        self.terms = list(args)
        
    
    def __str__(self):
        return ''.join(i.__str__() + ' * ' for i in self.terms).rstrip(' * ')
    

    def _latex(self, printer):
        return ''.join(i._latex(printer) for i in self.terms)


    def __mul__(self, other):
        if isinstance(other, ProductExpLTH):
            return ProductExpLTH(*self.terms, *other.terms)
        if isinstance(other, ExpLTH):
            return ProductExpLTH(*self.terms, other)
        
        result = other
        for i in self.terms[::-1]:
            result = i * result
            
        return result
            