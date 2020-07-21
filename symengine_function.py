#import symengine as sp
#import sympy as sp

from symengine import sympify, diff, sin, cos, Matrix, symbols, UndefFunction as Function

def gradient(scalar_function, var):
    '''
        Computes the gradient of a scalar function.
        expr : Expression to be differentiated
        var  : Variables which the derivatives are taken with respect to
    '''
    matrix_scalar_function = Matrix([scalar_function])
    return matrix_scalar_function.jacobian(var)

def lie_bracket(expr1, expr2, var):
    return expr2.jacobian(var)*expr1 - expr1.jacobian(var)*expr2

def xp_bracket(expr1, expr2, states, costates):
    if len(expr1) > 1:
        return expr1.jacobian(states)*(gradient(expr2, costates)).T - expr1.jacobian(costates)*(gradient(expr2, states)).T
    else:
        return gradient(expr1, states).dot(gradient(expr2, costates)) - gradient(expr1, costates).dot(gradient(expr2, states))