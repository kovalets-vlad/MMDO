def find_optimal_investing_strategy(costs, profits, budget):
    if len(costs) != len(profits) or any(len(costs[i]) != len(profits[i]) for i in range(len(costs))):
        raise Exception("Error: Costs/Profits Size Mismatch.")
    
    companies_count = len(costs)
    projects_count = len(costs[0])

    dp = [[0.0 for _ in range(budget + 1)] for _ in range(companies_count + 1)]
    strategy = [[-1 for _ in range(budget + 1)] for _ in range(companies_count + 1)]

    for i in range(1, companies_count + 1):
        for w in range(budget + 1):
            for j in range(projects_count):
                cost = costs[i - 1][j]
                profit = profits[i - 1][j]
                if w >= cost:
                    new_profit = profit + dp[i - 1][w - cost]
                    if new_profit >= dp[i][w]:
                        dp[i][w] = new_profit
                        strategy[i][w] = j

    selected_projects = []
    budget_copy = budget
    for i in range(companies_count, 0, -1):
        project_idx = strategy[i][budget_copy]
        if project_idx != -1:
            selected_projects.append((i, project_idx + 1))
            budget_copy -= costs[i - 1][project_idx]

    print(f"\nБюджет: {budget}")
    print(f"Максимальний прибуток: {dp[companies_count][budget]:.1f}")
    print("Вибрані проекти:")
    total_cost = 0.0
    for company, project in reversed(selected_projects):
        cost = costs[company - 1][project - 1]
        profit = profits[company - 1][project - 1]
        print(f"Компанія #{company}: Проект #{project} (Вартість: {cost}, Прибуток: {profit:.1f})")
        total_cost += cost
    print(f"Загальна вартість: {total_cost:.1f}")


company_1 = [(3, 0.8), (2, 0.4), (3, 1.1), (3, 0.9), (3, 0.8)]
company_2 = [(2, 0.5), (2, 0.4), (1, 0.4), (2, 0.6), (5, 1.3)]
company_3 = [(1, 0.3), (4, 0.8), (1, 0.4), (4, 1.2), (3, 0.8)]


costs_case1 = [
    [cost for cost, _ in company_1],
    [cost for cost, _ in company_2],
    [cost for cost, _ in company_3]
]
profits_case1 = [
    [profit for _, profit in company_1],
    [profit for _, profit in company_2],
    [profit for _, profit in company_3]
]
budget_case1 = 9

comp1 = [(4, 15), (8, 27), (12, 39), (16, 60), (20, 71)]
comp2 = [(4, 21), (8, 40), (12, 42), (16, 54), (20, 69)]
comp3 = [(4, 25), (8, 45), (12, 47), (16, 60), (20, 80)]
comp4 = [(4, 19), (8, 39), (12, 52), (16, 82), (20, 90)]

costs_case2 = [
    [cost for cost, _ in comp1],
    [cost for cost, _ in comp2],
    [cost for cost, _ in comp3],
    [cost for cost, _ in comp4]
]
profits_case2 = [
    [profit for _, profit in comp1],
    [profit for _, profit in comp2],
    [profit for _, profit in comp3],
    [profit for _, profit in comp4]
]
budgets_case2 = 20

find_optimal_investing_strategy(costs_case1, profits_case1, budget_case1)


find_optimal_investing_strategy(costs_case2, profits_case2, budgets_case2)