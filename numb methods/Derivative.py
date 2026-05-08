import sympy as sp

class Derivatives:
    def __init__(self, expr, var='x'):
        """
        expr : sympy вираз, напр. sp.sin(x) або x**2 + 3*x
        var  : змінна, по якій беремо похідну (рядок)
        """
        self.var = sp.Symbol(var)
        self.expr = expr

        self.f = sp.lambdify(self.var, self.expr, 'math')

    def derivative_expr(self, order=1):
        """Повертає символьний вираз похідної порядку `order`"""
        return sp.diff(self.expr, (self.var, order))

    def derivative_func(self, order=1):
        """Повертає числову функцію (math), яка обчислює похідну"""
        expr = self.derivative_expr(order)
        return sp.lambdify(self.var, expr, 'math')

    def eval(self, x, order=0):
        """order=0 -> f(x), order=1 -> f'(x)"""
        if order == 0:
            return self.f(x)
        func = self.derivative_func(order)
        return func(x)

    def __repr__(self):
        return f"f(x) = {self.expr}"
