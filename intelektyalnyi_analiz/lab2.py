import numpy as np
import pandas as pd
from scipy.optimize import linprog
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

cost_matrix0 = np.array([
    [451, 188, 842, 309, 226],
    [440, 795, 140, 837, 654],
    [463, 269, 950, 382, 232]
], dtype=float)
supply0 = np.array([10, 15, 5], dtype=float)
demand0 = np.array([13, 2, 5, 12, 5], dtype=float)

row_names0 = ["Ковель", "Одеса", "Вараш"]
col_names0 = ["Вінниця", "Львів", "Арциз", "Долина", "Славута"]


def balance_problem(cost_matrix, supply, demand, row_names, col_names):
    m, n = cost_matrix.shape
    supply = supply.copy()
    demand = demand.copy()
    cost = cost_matrix.copy()
    row_names = list(row_names)
    col_names = list(col_names)

    if supply.sum() < demand.sum():
        deficit = demand.sum() - supply.sum()
        supply = np.append(supply, deficit)
        dummy_row = np.full((1, n), 1000.0)
        cost = np.vstack([cost, dummy_row])
        row_names.append("Фейковий")
    elif supply.sum() > demand.sum():
        deficit = supply.sum() - demand.sum()
        demand = np.append(demand, deficit)
        dummy_col = np.full((m, 1), 1000.0)
        cost = np.hstack([cost, dummy_col])
        col_names.append("Фейковий")

    real_rows = [i for i, r in enumerate(row_names) if r != "Фейковий"]
    real_cols = [j for j, c in enumerate(col_names) if c != "Фейковий"]

    return cost, supply, demand, row_names, col_names, real_rows, real_cols


def solve_transport(cost, supply, demand, real_rows=None, real_cols=None):
    m, n = cost.shape
    c = cost.flatten()

    A_eq, b_eq = [], []
    for i in range(m):
        row = np.zeros(m * n)
        row[i * n:(i+1) * n] = 1
        A_eq.append(row); b_eq.append(supply[i])
    for j in range(n):
        row = np.zeros(m * n)
        row[j::n] = 1
        A_eq.append(row); b_eq.append(demand[j])

    res = linprog(c, A_eq=np.array(A_eq), b_eq=np.array(b_eq),
                  bounds=[(0, None)] * (m * n), method="highs")

    if not res.success:
        raise RuntimeError(f"LP не вирішено: {res.message}")

    plan = res.x.reshape(m, n)
    supply_shadow = res.eqlin.marginals[:m]
    demand_shadow = res.eqlin.marginals[m:m+n]

    reduced = np.zeros_like(cost)
    for i in range(m):
        for j in range(n):
            reduced[i, j] = cost[i, j] - (supply_shadow[i] + demand_shadow[j])

    if real_rows is not None:
        plan = plan[real_rows, :]
        supply_shadow = supply_shadow[real_rows]
        reduced = reduced[real_rows, :]
    if real_cols is not None:
        plan = plan[:, real_cols]
        demand_shadow = demand_shadow[real_cols]
        reduced = reduced[:, real_cols]

    real_rows = np.array(real_rows, dtype=int)
    real_cols = np.array(real_cols, dtype=int)
    true_cost = (plan * cost[np.ix_(real_rows, real_cols)]).sum()
    print(f"Загальна вартість: {true_cost}")

    return res, plan, supply_shadow, demand_shadow, reduced


def get_active_routes(plan, row_names, col_names, tol=1e-8):
    routes = []
    for i in range(len(row_names)):
        for j in range(len(col_names)):
            if plan[i, j] > tol:
                routes.append((row_names[i], col_names[j]))
    return set(routes)

cost_bal, supply_bal, demand_bal, row_names, col_names, real_rows, real_cols = balance_problem(
    cost_matrix0, supply0, demand0, row_names0, col_names0
)

res_base, plan_base, s_shadow_base, d_shadow_base, reduced_base = solve_transport(
    cost_bal, supply_bal, demand_bal, real_rows, real_cols
)

plan_clean = plan_base[np.ix_(real_rows, real_cols)]
cost_clean = cost_bal[np.ix_(real_rows, real_cols)]
row_names_clean = [row_names[i] for i in real_rows]
col_names_clean = [col_names[j] for j in real_cols]
s_shadow_clean = s_shadow_base[real_rows]
d_shadow_clean = d_shadow_base[real_cols]
reduced_clean = reduced_base[np.ix_(real_rows, real_cols)]

print("=== Тіньові ціни постачальників ===")
for r, v in zip(row_names_clean, s_shadow_clean):
    print(f"{r}: {v:.6f}")

print("\n=== Тіньові ціни споживачів ===")
for c, v in zip(col_names_clean, d_shadow_clean):
    print(f"{c}: {v:.6f}")


df_reduced = pd.DataFrame([
    {"Від": row_names_clean[i],
     "До": col_names_clean[j],
     "Вартість": cost_clean[i, j],
     "План": plan_clean[i, j],
     "Reduced cost": reduced_clean[i, j],
     "Активний": "✅" if plan_clean[i, j] > 1e-8 else "❌"}
    for i in range(len(row_names_clean))
    for j in range(len(col_names_clean))
])

df_reduced = df_reduced.sort_values(by="Активний", ascending=False)

print("\n=== Таблиця маршрутів (усі, з позначенням активних) ===")
print(df_reduced.to_string(index=False,
                           formatters={
                               "Вартість": "{:.2f}".format,
                               "План": "{:.2f}".format,
                               "Reduced cost": "{:.6f}".format
                           }))



plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.heatmap(cost_clean, annot=True, fmt=".0f", cmap="YlOrRd",
            xticklabels=col_names_clean, yticklabels=row_names_clean)
plt.title("Витрати")

plt.subplot(1, 2, 2)
sns.heatmap(plan_clean, annot=True, fmt=".1f", cmap="YlGnBu",
            xticklabels=col_names_clean, yticklabels=row_names_clean)
plt.title("Оптимальний план")
plt.tight_layout(); plt.show()

G = nx.DiGraph()
for r in row_names_clean: G.add_node(r, bipartite=0)
for c in col_names_clean: G.add_node(c, bipartite=1)

for i, r in enumerate(row_names_clean):
    for j, c in enumerate(col_names_clean):
        if plan_clean[i, j] > 1e-8:
            G.add_edge(r, c, weight=plan_clean[i, j])

pos = {r:(0,i) for i,r in enumerate(row_names_clean)}
pos.update({c:(1,j) for j,c in enumerate(col_names_clean)})

plt.figure(figsize=(10, 6))
nx.draw(G, pos, with_labels=True, node_size=3000, node_color="lightblue")
nx.draw_networkx_edge_labels(G, pos, edge_labels={(u,v):f"{d['weight']:.1f}" for u,v,d in G.edges(data=True)})
plt.title("Граф постачальники → споживачі")
plt.show()


routes_to_vary = [(0, 0), (1, 2)]
percent_range = np.linspace(-0.2, 0.2, 9)  

results = []

for p1 in percent_range:
    for p2 in percent_range:
        cost_mod = cost_matrix0.copy()
        i1, j1 = routes_to_vary[0]
        i2, j2 = routes_to_vary[1]
        cost_mod[i1, j1] = cost_matrix0[i1, j1] * (1 + p1)
        cost_mod[i2, j2] = cost_matrix0[i2, j2] * (1 + p2)

        cost_bal, supply_bal, demand_bal, row_names_bal, col_names_bal, real_rows, real_cols = \
        balance_problem(cost_mod, supply0, demand0, row_names0, col_names0)

        res, plan, s_shadow, d_shadow, reduced = solve_transport(cost_bal, supply_bal, demand_bal, real_rows, real_cols)


        plan_real = plan[np.ix_(real_rows, real_cols)]
        active = get_active_routes(plan_real,
                               [row_names_bal[r] for r in real_rows],
                               [col_names_bal[c] for c in real_cols])


        results.append({
            "Δмаршрут1": f"{p1*100:.0f}%",
            "Δмаршрут2": f"{p2*100:.0f}%",
            "Активні маршрути": active
        })

df = pd.DataFrame(results)
print(df.to_string(index=False))


vary_index = 0 
percent_range = np.linspace(-0.2, 0.2, 9)  
results_demand = []

for p in percent_range:
    demand_mod = demand0.copy()
    demand_mod[vary_index] = demand0[vary_index] * (1 + p)

    cost_bal, supply_bal, demand_bal, row_names_bal, col_names_bal, real_rows, real_cols = \
        balance_problem(cost_matrix0, supply0, demand_mod, row_names0, col_names0)

    res, plan, s_shadow, d_shadow, reduced = solve_transport(
        cost_bal, supply_bal, demand_bal, real_rows, real_cols
    )

    plan_real = plan[np.ix_(real_rows, real_cols)]
    total_cost = (plan_real * cost_matrix0[np.ix_(real_rows, real_cols)]).sum()

    results_demand.append({
        "Зміна попиту": f"{p*100:+.0f}%",
        "Новий попит Вінниці": demand_mod[vary_index],
        "Загальна вартість": total_cost,
    })

df_var = pd.DataFrame(results_demand)
print(df_var.to_string(index=False))