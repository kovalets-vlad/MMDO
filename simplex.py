def simplex(c, A, b):
    limits_count = len(A)  
    var_count = len(c)    

    table = []
    for i in range(limits_count):
        table.append(A[i] + [b[i]])

    table.append([-x for x in c] + [0])

    while True:
        pivot_col = 0
        min_value = table[-1][0]
        for i in range(1, len(table[0]) - 1):
            if (table[-1][i] < min_value):
                min_value = table[-1][i]
                pivot_col = i

        if (min_value) >= 0:
            break

        pivot_row = -1
        min_ratio = float('inf')
        for i in range(limits_count):
            if (table[i][pivot_col] > 0):
                ratio = table[i][-1] / table[i][pivot_col]
                if (ratio < min_ratio):
                    min_ratio = ratio
                    pivot_row = i

        if (pivot_row == -1):
            print("Error: Problem cannot be solved by simplex method.")
            return None, None

        pivot_value = table[pivot_row][pivot_col]
        for i in range(len(table[pivot_row])):
            table[pivot_row][i] /= pivot_value

        for i in range(len(table)):
            if (i != pivot_row):
                factor = table[i][pivot_col]
                for j in range(len(table[i])):
                    table[i][j] -= factor * table[pivot_row][j]

    optimal_value = table[-1][-1]
    solution = [0] * var_count
    for j in range(var_count):
        one_count = 0
        row_index = -1
        for i in range(limits_count):
            if table[i][j] == 1:
                one_count += 1
                row_index = i
            elif table[i][j] != 0:
                one_count = 0
                break
        if one_count == 1:
            solution[j] = table[row_index][-1]

    return optimal_value, solution

c = [8, 2, 0, 0, 0]  

A = [
    [1, -4, 1, 0, 0],   
    [-4, 1, 0, 1, 0],
    [1, 1, 0, 0, 1]    
]

b = [4, 4, 6] 

opt_val, opt_sol = simplex(c, A, b)
print(f"Optimal value of func: {opt_val}")
print(f"Optimal variables: {opt_sol}")
