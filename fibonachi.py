import sympy as sp

def read_function_from_file(filename):
    x = sp.Symbol('x')  
    with open(filename, 'r') as file:
        expression = file.read().strip()
    
    sympy_expr = sp.sympify(expression)  
    f = sp.lambdify(x, sympy_expr, modules=["math"])  
    print(f"Function read from file: f(x) = {sympy_expr}")
    return f, sympy_expr

def calculate_required_iterations(a, b, epsilon):
    fib = [1, 1]
    n = 2
    while fib[-1] < (b - a) / epsilon:
        fib.append(fib[-1] + fib[-2])
        n += 1
    return fib, n - 2  

def fibonacci_minimize(f, x1, x2, epsilon=0.01):
    fib_seq, iter = calculate_required_iterations(x1, x2, epsilon)
    print(f"\nRequired iterations for ε = {epsilon}: {iter}")

    rho = [1 - (fib_seq[-i - 2] / fib_seq[-i - 1]) for i in range(iter)]

    x3 = x1 + rho[0] * (x2 - x1)
    x4 = x1 + (1 - rho[0]) * (x2 - x1)

    print("\n--- Fibonacci Minimization Steps ---") 
    for i in range(1, iter):
        fx3 = f(x3)
        fx4 = f(x4)

        print(f"Iteration {i}:")
        print(f"  x3 = {x3:.6f}, f(x3) = {fx3:.6f}")
        print(f"  x4 = {x4:.6f}, f(x4) = {fx4:.6f}")

        if rho[i] == 0.5:
            rho[i] -= 1e-6  
        if fx3 > fx4:
            x1 = x3 
            x3 = x4
            x4 = x1 + (1 - rho[i]) * (x2 - x1)
        else:
            x2 = x4
            x4 = x3
            x3 = x1 + rho[i] * (x2 - x1)

    xmin = (x3 + x4) / 2
    fmin = f(xmin)

    print("\n--- Final Result ---")
    print(f"Minimum estimated at x = {xmin:.6f}")
    print(f"Function value at minimum f(x) = {fmin:.6f}")

if __name__ == "__main__":
    f, _ = read_function_from_file("mmdo/fun.txt")
    fibonacci_minimize(f, x1=-5, x2=5, epsilon=0.0001)
