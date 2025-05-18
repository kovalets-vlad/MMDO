import numpy as np
import sympy as sp

def read_function_from_file(filename):
    with open(filename, 'r') as file:
        func_str = file.read().strip()
    expr = sp.sympify(func_str)
    return expr

def get_variables(expr):
    symbols = sorted(expr.free_symbols, key=lambda s: s.name)
    if len(symbols) != 2:
        raise ValueError(f"Функція повинна мати рівно 2 змінні, знайдено {len(symbols)}")
    x, y = sp.symbols('x y')
    return expr, [x, y]

def compute_gradient(expr, variables):
    return [sp.diff(expr, var) for var in variables]

def compute_hessian(expr, variables):
    x, y = variables
    f_xx = sp.diff(expr, x, x)
    f_xy = sp.diff(expr, x, y)
    f_yx = sp.diff(expr, y, x)
    f_yy = sp.diff(expr, y, y)
    return sp.Matrix([[f_xx, f_xy], [f_yx, f_yy]])

def check_convexity(expr, variables):
    hessian = compute_hessian(expr, variables)
    x, y = variables
    
    hessian_func = sp.lambdify((x, y), hessian, 'numpy')
    
    test_points = [(1, 1), (-1, 1), (1, -1), (-1, -1), (0.5, 0.5)]
    is_convex = True
    valid_points = 0
    
    for point in test_points:
        try:
            H = hessian_func(*point)
            if np.any(np.isnan(H)) or np.any(np.isinf(H)):
                continue
                
            f_xx = H[0, 0]
            det_H = H[0, 0] * H[1, 1] - H[0, 1] * H[1, 0]
            if f_xx <= 0 or det_H <= 0:
                is_convex = False
            valid_points += 1
        except (ZeroDivisionError, ValueError, OverflowError):
            continue
    
    if valid_points == 0:
        print("Неможливо перевірити опуклість через невизначеність у всіх тестових точках.")
        return False
    elif not is_convex:
        print("Функція не є опуклою")
    else:
        print("Функція є опуклою.")
    
    return is_convex

def to_numeric_function(expr, variables):
    return sp.lambdify(variables, expr, 'numpy')

def to_numeric_gradient(gradient, variables):
    grad_funcs = [sp.lambdify(variables, g, 'numpy') for g in gradient]
    return lambda point: np.array([g(*point) for g in grad_funcs])

def fibonacci(n):
    fib = [1, 1]
    for i in range(2, n):
        fib.append(fib[i-1] + fib[i-2])
    return fib

def fibonacci_search(objective_func, x_prev, grad, a, b, tol=1e-6, max_fib=20):
    fib = fibonacci(max_fib)
    n = len(fib) - 1
    while n > 0 and (b - a) > tol:
        rho = 1 - fib[n-1] / fib[n]
        x1 = a + rho * (b - a)
        x2 = b - rho * (b - a)

        f1 = objective_func(*(x_prev - x1 * grad))
        f2 = objective_func(*(x_prev - x2 * grad))
        
        if f1 < f2:
            b = x2
        else:
            a = x1
        n -= 1

    return (a + b) / 2

def gradient_descent(objective_func, gradient_func, starting_point, initial_lr, num_iterations, tolerance=1e-5):
    x_prev = np.array(starting_point, dtype=float)
    history = [x_prev.copy()]

    for i in range(num_iterations):
        grad = gradient_func(x_prev)
        
        lr = fibonacci_search(objective_func, x_prev, grad, 0, initial_lr)
        
        x_new = x_prev - lr * grad

        print(f"Ітерація {i+1}: Точка = {x_prev}, Значення функції = {objective_func(*x_prev)}")

        if np.linalg.norm(x_new - x_prev) < tolerance:
            break

        x_prev = x_new
        history.append(x_prev.copy())

    return x_prev, history

def main(filename='mmdo/function.txt'):
    expr = read_function_from_file(filename)
    expr, variables = get_variables(expr)
    
    check_convexity(expr, variables)
    
    gradient = compute_gradient(expr, variables)
    objective_func = to_numeric_function(expr, variables)
    gradient_func = to_numeric_gradient(gradient, variables)
    
    starting_point = [2.0, 1.0]
    initial_lr = 1e-2  
    num_iterations = 2000
    
    optimal_point, history = gradient_descent(objective_func, gradient_func, starting_point, initial_lr, num_iterations)
    
    print(f"\nОптимальна точка: {optimal_point}")
    print(f"Значення функції в оптимальній точці: {objective_func(*optimal_point)}")

if __name__ == "__main__":
    main(filename='mmdo/function.txt')