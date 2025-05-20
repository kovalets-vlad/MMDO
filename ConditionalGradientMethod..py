import numpy as np
from scipy.optimize import linprog
from sympy import symbols, sympify, lambdify, diff

with open("mmdo/dfd.txt", "r", encoding="utf-8") as file:
    lines = [line.strip() for line in file if line.strip()]

func_line = lines[0]
func_input = func_line.split('=')[1].strip()

x1, x2 = symbols("x1 x2")
f_expr = sympify(func_input)
f_func = lambdify((x1, x2), f_expr, "numpy")
grad_expr = [diff(f_expr, var) for var in (x1, x2)]
grad_func = lambdify((x1, x2), grad_expr, "numpy")

n_constraints = int(lines[1].split('=')[1])
constraint_lines = lines[2:2+n_constraints]

A_ub = []
b_ub = []

for constraint in constraint_lines:
    constraint = constraint.replace(" ", "")
    if "<=" in constraint:
        lhs, rhs = constraint.split("<=")
        sense = "<="
    elif ">=" in constraint:
        lhs, rhs = constraint.split(">=")
        sense = ">="
    elif "<" in constraint:
        lhs, rhs = constraint.split("<")
        sense = "<"
    elif ">" in constraint:
        lhs, rhs = constraint.split(">")
        sense = ">"
    elif "=" in constraint:
        lhs, rhs = constraint.split("=")
        sense = "="
    else:
        raise ValueError("Обмеження має містити один з символів: <=, >=, <, >, =")

    lhs_expr = sympify(lhs)
    coeffs = [lhs_expr.coeff(var) for var in (x1, x2)]
    coeffs = [float(c) for c in coeffs]
    rhs_val = float(rhs)

    if sense == "<=":
        A_ub.append(coeffs)
        b_ub.append(rhs_val)
    elif sense == "<":
        A_ub.append(coeffs)
        b_ub.append(rhs_val - 1e-8)
    elif sense == ">=":
        A_ub.append([-c for c in coeffs])
        b_ub.append(-rhs_val)
    elif sense == ">":
        A_ub.append([-c for c in coeffs])
        b_ub.append(-rhs_val - 1e-8)
    elif sense == "=":
        A_ub.append(coeffs)
        b_ub.append(rhs_val)
        A_ub.append([-c for c in coeffs])
        b_ub.append(-rhs_val)

x0_line = [line for line in lines if line.startswith("x0")][0]
x0 = list(map(float, x0_line.split('=')[1].split()))

eps_line = [line for line in lines if line.startswith("eps")][0]
eps = float(eps_line.split('=')[1])

def conditional_gradient(x0, epsilon, max_iter=100):
    xk = np.array(x0, dtype=float)
    A_np = np.array(A_ub)
    b_np = np.array(b_ub)
    path = [xk.copy()]

    print("\nПошук мінімуму функції методом умовного градієнта")
    print(f"\n{' Ітерація':^10} | {'x1':^16} | {'x2':^16} | {'f(х)':^16} | {'β':^16}")
    print("-" * 85)

    for k in range(max_iter):
        grad = np.array(grad_func(*xk))

        res = linprog(c=grad, A_ub=A_np, b_ub=b_np, bounds=[(None, None)]*2, method='highs')
        if not res.success:
            print("Помилка у задачі ЛП.")
            break
        yk = res.x
        hk = yk - xk

        def phi(beta):
            return f_func(*(xk + beta * hk))

        beta_vals = np.linspace(0, 1, 100)
        beta_opt = min(beta_vals, key=phi)

        xk1 = xk + beta_opt * hk
        print(f"{k+1:^10} | {xk1[0]:^16.6f} | {xk1[1]:^16.6f} | {f_func(*xk1):^16.6f} | {beta_opt:^16.6f}")

        if np.linalg.norm(xk1 - xk) < epsilon:
            break

        xk = xk1
        path.append(xk.copy())

    return xk, path

solution, path = conditional_gradient(x0, epsilon=eps)

print("\nРезультат:")
print(f"x* = {solution}")
print(f"f(x*) = {f_func(*solution):.6f}")
