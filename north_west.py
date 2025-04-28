def north_west_corner_method(supply, demand):
    m = len(supply)   
    n = len(demand)    

    allocation = [[0 for _ in range(n)] for _ in range(m)]

    i = 0  
    j = 0  

    while i < m and j < n:
        allocated = min(supply[i], demand[j])
        allocation[i][j] = allocated

        supply[i] -= allocated
        demand[j] -= allocated

        if supply[i] == 0 and i < m - 1:
            i += 1
        elif demand[j] == 0 and j < n - 1:
            j += 1
        else:
            i += 1
            j += 1

    return allocation


supply = [40, 30, 20]
demand = [30, 25, 18, 20]

result = north_west_corner_method(supply[:], demand[:])  

print("Опорний план (метод північно-західного кута):")
for row in result:
    print(row)

