from typing import List, Tuple, Dict, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

def get_balanced_tp(
    supply: List[float], 
    demand: List[float], 
    costs: List[List[float]], 
    penalties: Optional[List[float]] = None
) -> Tuple[List[float], List[float], List[List[float]]]:
    if not supply or not demand or not costs:
        raise ValueError("Supply, demand, and costs must not be empty")
    if any(s < 0 for s in supply) or any(d < 0 for d in demand):
        raise ValueError("Supply and demand must be non-negative")
    
    total_supply = sum(supply)
    total_demand = sum(demand)
    
    if total_supply < total_demand:
        if penalties is None or len(penalties) != len(demand):
            raise ValueError("Penalties required for dummy supply with length equal to demand")
        return supply + [total_demand - total_supply], demand, costs + [penalties]
    elif total_supply > total_demand:
        return supply, demand + [total_supply - total_demand], costs + [[0] * len(demand)]
    return supply, demand, costs

def north_west_corner(supply: List[float], demand: List[float]) -> List[Tuple[Tuple[int, int], float]]:
    supply_copy = supply.copy()
    demand_copy = demand.copy()
    bfs = []
    i, j = 0, 0
    
    while len(bfs) < len(supply) + len(demand) - 1:
        if i >= len(supply) or j >= len(demand):
            break
        v = min(supply_copy[i], demand_copy[j])
        supply_copy[i] -= v
        demand_copy[j] -= v
        bfs.append(((i, j), v))
        if supply_copy[i] == 0 and i < len(supply) - 1:
            i += 1
        elif demand_copy[j] == 0 and j < len(demand) - 1:
            j += 1
    return bfs

def get_us_and_vs(bfs: List[Tuple[Tuple[int, int], float]], costs: List[List[float]]) -> Tuple[List[Optional[float]], List[Optional[float]]]:
    us = [None] * len(costs)
    vs = [None] * len(costs[0])
    us[0] = 0
    bfs_copy = bfs.copy()
    
    while bfs_copy:
        progress = False
        for index, (pos, _) in enumerate(bfs_copy):
            i, j = pos
            if us[i] is None and vs[j] is None:
                continue
            cost = costs[i][j]
            if us[i] is None:
                us[i] = cost - vs[j]
                progress = True
            else:
                vs[j] = cost - us[i]
                progress = True
            bfs_copy.pop(index)
            break
        if not progress:
            break
    return us, vs



def get_ws(bfs: List[Tuple[Tuple[int, int], float]], costs: List[List[float]], us: List[Optional[float]], vs: List[Optional[float]]) -> List[Tuple[Tuple[int, int], float]]:
    ws = []
    for i, row in enumerate(costs):
        for j, cost in enumerate(row):
            if all(p != (i, j) for p, _ in bfs):
                u = us[i] if us[i] is not None else np.nan
                v = vs[j] if vs[j] is not None else np.nan
                ws.append(((i, j), u + v - cost))
    return ws

def can_be_improved(ws: List[Tuple[Tuple[int, int], float]]) -> bool:
    return any(v > 0 for _, v in ws)

def get_entering_variable_position(ws: List[Tuple[Tuple[int, int], float]]) -> Tuple[int, int]:
    return max(ws, key=lambda w: w[1])[0]

def get_possible_next_nodes(loop: List[Tuple[int, int]], not_visited: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    last_node = loop[-1]
    nodes_in_row = [n for n in not_visited if n[0] == last_node[0]]
    nodes_in_column = [n for n in not_visited if n[1] == last_node[1]]
    if len(loop) < 2:
        return nodes_in_row + nodes_in_column
    prev_node = loop[-2]
    return nodes_in_column if prev_node[0] == last_node[0] else nodes_in_row

def get_loop(bv_positions: List[Tuple[int, int]], ev_position: Tuple[int, int]) -> List[Tuple[int, int]]:
    def inner(loop: List[Tuple[int, int]]) -> Optional[List[Tuple[int, int]]]:
        if len(loop) > 3 and len(get_possible_next_nodes(loop, [ev_position])) == 1:
            return loop
        not_visited = list(set(bv_positions) - set(loop))
        for next_node in get_possible_next_nodes(loop, not_visited):
            new_loop = inner(loop + [next_node])
            if new_loop:
                return new_loop
        return None
    return inner([ev_position]) or []

def loop_pivoting(bfs: List[Tuple[Tuple[int, int], float]], loop: List[Tuple[int, int]]) -> List[Tuple[Tuple[int, int], float]]:
    even_cells = loop[0::2]
    odd_cells = loop[1::2]
    get_bv = lambda pos: next(v for p, v in bfs if p == pos)
    leaving_position = min(odd_cells, key=get_bv)
    leaving_value = get_bv(leaving_position)
    
    new_bfs = []
    for p, v in [bv for bv in bfs if bv[0] != leaving_position] + [(loop[0], 0)]:
        if p in even_cells:
            v += leaving_value
        elif p in odd_cells:
            v -= leaving_value
        new_bfs.append((p, v))
    return new_bfs

def transportation_method_detailed(
    supply: List[float], 
    demand: List[float], 
    costs: List[List[float]], 
    penalties: Optional[List[float]] = None
) -> Dict:
    balanced_supply, balanced_demand, balanced_costs = get_balanced_tp(supply, demand, costs, penalties)
    
    def inner(bfs: List[Tuple[Tuple[int, int], float]]) -> Tuple[List[Tuple[Tuple[int, int], float]], List[Optional[float]], List[Optional[float]], List[Tuple[Tuple[int, int], float]]]:
        us, vs = get_us_and_vs(bfs, balanced_costs)
        ws = get_ws(bfs, balanced_costs, us, vs)
        if can_be_improved(ws):
            ev_position = get_entering_variable_position(ws)
            loop = get_loop([p for p, _ in bfs], ev_position)
            return inner(loop_pivoting(bfs, loop))
        return bfs, us, vs, ws
    
    initial_bfs = north_west_corner(balanced_supply, balanced_demand)
    final_bfs, us, vs, ws = inner(initial_bfs)
    
    solution = np.zeros((len(costs), len(costs[0])))
    for (i, j), v in final_bfs:
        if i < len(costs) and j < len(costs[0]):
            solution[i][j] = v
    
    total_cost = sum(costs[i][j] * solution[i][j] for i in range(len(costs)) for j in range(len(costs[0])))
    
    return {
        "bfs": final_bfs,
        "us": us,
        "vs": vs,
        "ws": ws,
        "solution": solution,
        "total_cost": total_cost
    }

supply = [10, 15, 5]  
demand = [13, 2, 5, 12, 5]  
costs = [
    [451, 188, 842, 309, 226],  
    [440, 795, 140, 837, 654],  
    [463, 269, 950, 382, 232]  
]
penalties = [1000, 1000, 1000, 1000, 1000]  
row_names = ["Ковель", "Одеса", "Вараш"]
col_names = ["Вінниця", "Львів", "Арциз", "Долина", "Славута"]

result = transportation_method_detailed(supply, demand, costs, penalties)

balanced_supply, balanced_demand, balanced_costs = get_balanced_tp(supply, demand, costs, penalties)
row_names_balanced = row_names + ["Dummy"] if len(balanced_supply) > len(row_names) else row_names
col_names_balanced = col_names + ["Dummy"] if len(balanced_demand) > len(col_names) else col_names

solution_df = pd.DataFrame(result["solution"], index=row_names, columns=col_names)

bfs_rows = [
    {
        "Постачальник": row_names[i],
        "Споживач": col_names[j],
        "Кількість (BFS)": v,
        "Вартість(за од.)": costs[i][j],
        "Внесок у вартість": costs[i][j] * v
    }
    for (i, j), v in result["bfs"] if i < len(costs) and j < len(costs[0]) and v > 0
]
bfs_df = pd.DataFrame(bfs_rows)


us_df = pd.DataFrame({"Постачальник": row_names, "Тіньова_ціна (u_i)": result["us"][:len(row_names)]})
vs_df = pd.DataFrame({"Споживач": col_names, "Тіньова_ціна (v_j)": result["vs"][:len(col_names)]})

rc_rows = [
    {
        "Постачальник": row_names[i],
        "Споживач": col_names[j],
        "Вартість": costs[i][j],
        "Знижена_вартість": (result["us"][i] or 0) + (result["vs"][j] or 0) - costs[i][j],
        "У_базисі": bool(result["solution"][i][j] > 0)
    }
    for i in range(len(costs)) for j in range(len(costs[0]))
]
rc_df = pd.DataFrame(rc_rows)

print("\nОптимальний план (матриця потоків):")
print(solution_df)
print("\nБазисні змінні та їх вклад у вартість:")
print(bfs_df)
print("\nТіньові ціни для обмежень постачання (u_i):")
print(us_df)
print("\nТіньові ціни для обмежень попиту (v_j):")
print(vs_df)
print("\nЗнижені вартості (маржинали) для всіх маршрутів:")
print(rc_df)



plt.figure(figsize=(10, 6))
sns.heatmap(costs, annot=True, fmt=".0f", cmap="YlOrRd", xticklabels=col_names, yticklabels=row_names)
plt.title("Теплокарта матриці витрат")
plt.xlabel("Споживачі")
plt.ylabel("Постачальники")
plt.savefig("cost_heatmap.png")
plt.close()

plt.figure(figsize=(10, 6))
sns.heatmap(result["solution"], annot=True, fmt=".0f", cmap="YlGnBu", xticklabels=col_names, yticklabels=row_names)
plt.title("Теплокарта оптимального плану")
plt.xlabel("Споживачі")
plt.ylabel("Постачальники")
plt.savefig("optimal_plan_heatmap.png")
plt.close()

G = nx.DiGraph()
for i, supplier in enumerate(row_names):
    G.add_node(supplier, subset=0, shadow_price=result["us"][i] or 0)
for j, consumer in enumerate(col_names):
    G.add_node(consumer, subset=1, shadow_price=result["vs"][j] or 0)


for (i, j), v in result["bfs"]:
    if i < len(costs) and j < len(costs[0]) and v > 0:
        G.add_edge(row_names[i], col_names[j], weight=v)

plt.figure(figsize=(12, 8))
pos = nx.bipartite_layout(G, row_names)
nx.draw(G, pos, with_labels=False, node_color=['lightblue' if n in row_names else 'lightgreen' for n in G.nodes], node_size=2000)
node_labels = {node: f"{node}\n(u_i={G.nodes[node]['shadow_price']:.0f})" if node in row_names else f"{node}\n(v_j={G.nodes[node]['shadow_price']:.0f})" for node in G.nodes}
nx.draw_networkx_labels(G, pos, node_labels, font_size=10)
edge_labels = {(u, v): f"{d['weight']:.0f}" for u, v, d in G.edges(data=True)}
nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=10)
plt.title("Мережа: Постачальники → Споживачі")
plt.savefig("network_graph.png")
plt.close()
