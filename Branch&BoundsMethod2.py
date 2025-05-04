INF = float('inf')

def branch_and_bound(cost_matrix):
    n = len(cost_matrix)
    best_path = []
    best_cost = float('inf')
    
    def get_lower_bound(matrix, path, visited):
        total = 0
        for i in range(len(path) - 1):
            total += matrix[path[i]][path[i + 1]]
            
        unvisited = [i for i in range(n) if i not in visited]
        if not unvisited:
            return total + matrix[path[-1]][path[0]]
            
        for i in unvisited:
            min_cost = min(matrix[i][j] for j in range(n) if j != i and (j in unvisited or j == path[0]))
            total += min_cost
            
        return total

    def search(path, visited, current_cost):
        nonlocal best_path, best_cost
        
        if len(path) == n:
            total_cost = current_cost + cost_matrix[path[-1]][path[0]]
            if total_cost < best_cost:
                best_cost = total_cost
                best_path = path + [path[0]]
            return

        current_city = path[-1]
        for next_city in range(n):
            if next_city not in visited:
                new_cost = current_cost + cost_matrix[current_city][next_city]
                lower_bound = get_lower_bound(cost_matrix, path + [next_city], set(visited) | {next_city})
                if lower_bound < best_cost:
                    search(path + [next_city], visited + [next_city], new_cost)

    for city in range(n):
        search([city], [city], 0)
    
    return [city + 1 for city in best_path], best_cost

cost_matrix = [[INF, 11, 21, 6, 8],
                [13, INF, 17, 8, 11],
                [19, 18, INF, 7, 21],
                [22, 15, 11, INF, 17],
                [32, 4, 12, 6, INF]]

# cost_matrix = [[INF, 6, 16, 10, 10],
#                 [7, INF, 4, 3, 5],
#                 [9, 12, INF, 12, 5],
#                 [18, 7, 2, INF, 12],
#                 [17, 11, 10, 16, INF]]


best_path, best_cost = branch_and_bound(cost_matrix)
print("Optimal route:", best_path)
print("Optimal cost:", best_cost)
