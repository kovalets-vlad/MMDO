import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def perform_statistical_analysis(df):
    print("--- ЗАВДАННЯ 1: Розширений статистичний аналіз ---")
    
    selected_products = df['cm_name'].dropna().unique()
    if len(selected_products) > 10:
        selected_products = np.random.choice(selected_products, 10, replace=False)
    else:
        selected_products = selected_products

    df_selected = df[df['cm_name'].isin(selected_products)]

    price_stats = df_selected.groupby('cm_name')['mp_price'].describe(percentiles=[.25, .5, .75])
    price_stats['variance'] = df_selected.groupby('cm_name')['mp_price'].var()

    period1_df = df_selected[df_selected['mp_year'] < 2015]
    period2_df = df_selected[df_selected['mp_year'] >= 2015]

    price_stats_period1 = period1_df.groupby('cm_name')['mp_price'].describe(percentiles=[.25, .5, .75])
    price_stats_period1['variance'] = period1_df.groupby('cm_name')['mp_price'].var()

    price_stats_period2 = period2_df.groupby('cm_name')['mp_price'].describe(percentiles=[.25, .5, .75])
    price_stats_period2['variance'] = period2_df.groupby('cm_name')['mp_price'].var()

    crosstab_cm_pt = pd.crosstab(df_selected['cm_name'], df_selected['pt_name'])

    plt.figure(figsize=(14, 6))
    sns.heatmap(price_stats[['mean', 'std', 'min', '25%', '50%', '75%', 'max', 'variance']], annot=True, fmt=".2f", cmap="YlGnBu")
    plt.title("Базова статистика для mp_price по cm_name (2007-2024)")
    plt.ylabel("Товар")
    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=True)
    sns.heatmap(price_stats_period1[['mean', 'std', 'min', '25%', '50%', '75%', 'max', 'variance']], annot=True, fmt=".2f", cmap="Blues", ax=axes[0])
    axes[0].set_title("Статистика для 2007-2014")
    axes[0].set_ylabel("Товар")
    sns.heatmap(price_stats_period2[['mean', 'std', 'min', '25%', '50%', '75%', 'max', 'variance']], annot=True, fmt=".2f", cmap="Greens", ax=axes[1])
    axes[1].set_title("Статистика для 2015-2024")
    axes[1].set_ylabel("")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(14, 8))
    sns.heatmap(crosstab_cm_pt, cmap="magma", annot=False, cbar_kws={'label': 'Кількість'})
    plt.title("Зведена таблиця (crosstab) для cm_name та pt_name")
    plt.xlabel("Тип ринку")
    plt.ylabel("Товар")
    plt.tight_layout()
    plt.show()

    print("-" * 50 + "\n")


def visualize_price_trends(df, product_to_analyze):
    print("--- ЗАВДАННЯ 2: Візуалізація трендів цін ---")

    dataset_product = df[df['cm_name'] == product_to_analyze].copy()
    dataset_product['date'] = pd.to_datetime(dataset_product['mp_year'].astype(str) + '-' + dataset_product['mp_month'].astype(str))
    dataset_product = dataset_product.sort_values('date')

    plt.figure(figsize=(12, 5))
    sns.lineplot(x='mp_year', y='mp_price', data=dataset_product)
    plt.title(f'Динаміка цін на товар: {product_to_analyze}')
    plt.xlabel('Рік')
    plt.ylabel('Ціна')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.scatter(dataset_product["mp_year"], np.log(dataset_product["mp_price"]), alpha=0.3)
    plt.title(f"Розсіювання цін по роках для: {product_to_analyze}")
    plt.xlabel('Рік')
    plt.ylabel('Логарифм ціни')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    price_log_mexico = df[df["adm0_name"] == "Mexico"].copy()
    price_log = np.log(price_log_mexico["mp_price"])

    plt.figure(figsize=(18, 5))
    plt.suptitle("Гістограми розподілу цін в Мексиці")
    plt.subplot(1, 3, 1)
    plt.hist(price_log, bins=10, edgecolor='black')
    plt.title('Надто мало bins (bins=10)')
    plt.xlabel('Логарифм ціни')
    plt.ylabel('Частота')

    plt.subplot(1, 3, 2)
    plt.hist(price_log, bins=50, edgecolor='black')
    plt.title('Оптимальна кількість (bins=50)')
    plt.xlabel('Логарифм ціни')

    plt.subplot(1, 3, 3)
    plt.hist(price_log, bins=200, edgecolor='black')
    plt.title('Надто багато bins (bins=200)')
    plt.xlabel('Логарифм ціни')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    print("-" * 50 + "\n")


def preprocess_and_visualize_encoding(df):
    print("--- ЗАВДАННЯ 3: Обробка пропусків та кодування ---")
    
    print("Кількість пропусків до обробки:")
    print(df[['adm1_name', 'cm_name']].isnull().sum())

    data_cleaned = df.dropna(subset=['adm1_name', 'cm_name']).copy()

    print("\nКількість пропусків після обробки:")
    print(data_cleaned[['adm1_name', 'cm_name']].isnull().sum())

    labels, unique_categories = pd.factorize(data_cleaned['pt_name'])
    data_cleaned['pt_name_encoded'] = labels + 1

    encoding_map = {i + 1: category for i, category in enumerate(unique_categories)}

    plt.figure(figsize=(12, 7))
    plt.scatter(data_cleaned['mp_year'], data_cleaned['pt_name_encoded'], alpha=0.1)
    plt.title('Розподіл типів ринків по роках')
    plt.xlabel('Рік')
    plt.ylabel('Тип ринку (закодований)')
    plt.yticks(list(encoding_map.keys()), list(encoding_map.values()))
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    print("-" * 50 + "\n")


def perform_correlation_analysis(df):
    print("--- ЗАВДАННЯ 4: Кореляційний аналіз ---")
    
    numeric_cols = ['mp_price', 'mp_year', 'mp_month']
    correlation_matrix = df[numeric_cols].corr(method='pearson')

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        correlation_matrix,
        annot=True,
        cmap='coolwarm',
        fmt=".2f",
        linewidths=.5
    )
    plt.title('Теплова карта кореляції')
    plt.show()

    plt.figure(figsize=(12, 5))
    plt.scatter(correlation_matrix["mp_year"], correlation_matrix["mp_price"], alpha=0.7)
    plt.title('Кореляція між роком та ціною')
    plt.xlabel('Рік')
    plt.ylabel('Ціна')
    plt.show()

    plt.scatter(correlation_matrix["mp_month"], correlation_matrix["mp_price"], alpha=0.7)
    plt.title('Кореляція між роком та ціною')
    plt.xlabel('Місяць')
    plt.ylabel('Ціна')
    plt.show()

def forecasting_using_linear_regression(df, product_to_analyze):
    print("--- ЗАВДАННЯ 5: Лінійна регресія ---")
    product_df = df[df['cm_name'] == product_to_analyze].copy()
    yearly_avg_price = product_df.groupby('mp_year')['mp_price'].mean().reset_index()

    X = yearly_avg_price[['mp_year']] 
    y = yearly_avg_price['mp_price']

    model = LinearRegression()
    model.fit(X, y)

    y_pred = model.predict(X)

    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)

    print(f"--- Аналіз моделі для '{product_to_analyze}' ---")
    print(f"Середня абсолютна помилка (MAE): {mae:.2f}")
    print(f"Коренева середньоквадратична помилка (RMSE): {rmse:.2f}")
    print(f"Коефіцієнт детермінації (R²): {r2:.2f}")

    last_year = X['mp_year'].max()
    future_years = np.array([
        [last_year + 1],
        [last_year + 2],
        [last_year + 3]
    ])

    future_predictions = model.predict(future_years)

    print("\nПрогноз середньої ціни на майбутні роки:")
    for year, prediction in zip(future_years.flatten(), future_predictions):
        print(f"{year}: {prediction:.2f}")


    plt.figure(figsize=(12, 7))
    plt.scatter(X, y, color='blue', label='Реальні середні ціни')
    plt.plot(X, y_pred, color='green', linewidth=2, label='Лінія регресії (тренд)')
    plt.plot(future_years, future_predictions, color='red', linestyle='--', linewidth=2, label='Прогноз')

    plt.title(f'Прогноз цін на "{product_to_analyze}"')
    plt.xlabel('Рік')
    plt.ylabel('Середня ціна')
    plt.legend()
    plt.grid(True)
    plt.show()

    data = {
        "mae_lr": mae,
        "rmse_lr": rmse,
        "r2_lr": r2
    }

    return data

def forecasting_using_mlp(df, product_to_analyze):
    print("--- ЗАВДАННЯ 6: Нейро мережі ---")
    product_df = df[df['cm_name'] == product_to_analyze].copy()
    monthly_avg_price = product_df.groupby(['mp_year', 'mp_month'])['mp_price'].mean().reset_index()

    X = monthly_avg_price[['mp_year', 'mp_month']]
    y = monthly_avg_price['mp_price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    mlp_model = MLPRegressor(
        hidden_layer_sizes=(20, 10), 
        max_iter=1000,                
        activation='relu',            
        random_state=42
    )
    mlp_model.fit(X_train_scaled, y_train)
    y_pred_mlp = mlp_model.predict(X_test_scaled)

    mae_mlp = mean_absolute_error(y_test, y_pred_mlp)
    rmse_mlp = np.sqrt(mean_squared_error(y_test, y_pred_mlp))
    r2_mlp = r2_score(y_test, y_pred_mlp)

    print("\n--- Аналіз MLP моделі ---")
    print(f"{'Metric':<10} | {'MLP Regressor':<15}")
    print("-" * 55)
    print(f"{'MAE':<10} | {mae_mlp:<15.2f}")
    print(f"{'RMSE':<10} | {rmse_mlp:<15.2f}")
    print(f"{'R² Score':<10} | {r2_mlp:<15.2f}")

    results_df = X_test.copy()
    results_df['actual_price'] = y_test
    results_df['mlp_prediction'] = y_pred_mlp
    results_df_sorted = results_df.sort_values(by=['mp_year', 'mp_month'])

    plt.figure(figsize=(15, 8))
    plt.plot(results_df_sorted['actual_price'].values, label='Реальні дані', color='blue', marker='o', linestyle='None')
    plt.plot(results_df_sorted['mlp_prediction'].values, label='Прогноз MLP Regressor', color='red', linewidth=2)

    plt.title(f'Прогноз для "{product_to_analyze}"')
    plt.xlabel('Часові точки (відсортовані)')
    plt.ylabel('Ціна')
    plt.legend()
    plt.grid(True)
    plt.show()

    data = { 
        "mae_mlp": mae_mlp,
        "rmse_mlp": rmse_mlp,
        "r2_mlp": r2_mlp
    }
    return data

def analysis_of_market_relationships(df, product_to_analyze, market1, market2):
    print("--- ЗАВДАННЯ 7: Аналіз взаємозв'язків між ринками ---")
    market_df = df[
    (df['cm_name'] == product_to_analyze) &
    (df['mkt_name'].isin([market1, market2]))
]

    monthly_avg = market_df.groupby(['mp_year', 'mp_month', 'mkt_name'])['mp_price'].mean().reset_index()

    aligned_prices = monthly_avg.pivot_table(
        index=['mp_year', 'mp_month'],
        columns='mkt_name',
        values='mp_price'
    ).dropna()

    print(f"Вирівняні дані для ринків '{market1}' та '{market2}':")
    print(aligned_prices.head())

    X = aligned_prices[[market1]]
    y = aligned_prices[market2]

    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, alpha=0.5, label='Реальні дані')
    plt.plot(X, y_pred, color='red', linewidth=2, label='Лінія регресії')
    plt.title(f'Прогноз цін в {market2} залежно від цін в {market1}')
    plt.xlabel(f'Ціна в {market1}')
    plt.ylabel(f'Ціна в {market2}')
    plt.legend()
    plt.grid(True)
    plt.show()

    coef = model.coef_[0]
    intercept = model.intercept_
    r2 = r2_score(y, y_pred)
    print("\n--- Аналіз чутливості: Базова модель ---")
    print(f"Коефіцієнт (slope): {coef:.2f}")
    print(f"Перетин (intercept): {intercept:.2f}")
    print(f"R² score: {r2:.2f}")
    print(f"Інтерпретація: При зміні ціни в '{market1}' на 1 одиницю, ціна в '{market2}' змінюється на {coef:.2f} одиниць.")

    outlier_index = aligned_prices[market1].idxmax()
    aligned_prices_no_outlier = aligned_prices.drop(outlier_index)

    X_no_outlier = aligned_prices_no_outlier[[market1]]
    y_no_outlier = aligned_prices_no_outlier[market2]

    model_no_outlier = LinearRegression()
    model_no_outlier.fit(X_no_outlier, y_no_outlier)
    y_pred_no_outlier = model_no_outlier.predict(X_no_outlier)

    coef_no_outlier = model_no_outlier.coef_[0]
    print(f"\n--- Аналіз чутливості: Модель без викиду ---")
    print(f"Новий коефіцієнт (slope): {coef_no_outlier:.2f}")
    print(f"Зміна коефіцієнту: {((coef_no_outlier - coef) / coef * 100):.2f}%")

    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, alpha=0.3, label='Реальні дані (з викидом)')
    plt.plot(X, y_pred, color='red', linewidth=2, label=f'Регресія з викидом (coef={coef:.2f})')
    plt.plot(X_no_outlier, y_pred_no_outlier, color='green', linestyle='--', linewidth=2, label=f'Регресія без викиду (coef={coef_no_outlier:.2f})')
    plt.scatter(aligned_prices.loc[outlier_index, market1], aligned_prices.loc[outlier_index, market2],
                color='purple', s=200, edgecolor='black', zorder=5, label='Видалений викид')
    plt.title('Аналіз чутливості до викидів')
    plt.xlabel(f'Ціна в {market1}')
    plt.ylabel(f'Ціна в {market2}')
    plt.legend()
    plt.grid(True)
    plt.show()

def analysis_of_the_impact_of_currency(prices_df, rates_df, COMMODITY):
    print("--- ЗАВДАННЯ 8: Аналіз впливу валютних курсів ---")

    export_countries = ['Thailand', 'Viet Nam', 'China', 'Pakistan', 'Ukraine']

    prices_subset = prices_df[
        (prices_df['adm0_name'].isin(export_countries)) &
        (prices_df['cm_name'] == COMMODITY)
    ].copy()

    
    rates_df['date'] = pd.to_datetime(rates_df['Month/Year'], dayfirst=True)
    rates_df['year'] = rates_df['date'].dt.year
    rates_df['month'] = rates_df['date'].dt.month
    
    currency_columns = [col for col in rates_df.columns if col not in ['Month/Year', 'date', 'year', 'month']]
    
    rates_long = pd.melt(
        rates_df,
        id_vars=['year', 'month'],
        value_vars=currency_columns,
        var_name='currency',
        value_name='rate_to_usd'
    )
    
    prices_to_merge = prices_subset.rename(columns={
        'mp_year': 'year',
        'mp_month': 'month',
        'cur_name': 'currency'
    })
    
    merged_df = pd.merge(
        prices_to_merge,
        rates_long,
        on=['year', 'month', 'currency'],
        how='inner'
    )
    
    if merged_df.empty:
        print("Після об'єднання не залишилося даних. Можливо, назви валют не збігаються або немає пересічних дат.")
        return

    
    merged_df['price_usd'] = merged_df['mp_price'] / merged_df['rate_to_usd']
    

    yearly_avg_usd = merged_df.groupby('year')['price_usd'].mean().reset_index()

    X = yearly_avg_usd[['year']]
    y = yearly_avg_usd['price_usd']

    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    print("\nПрогноз середньої ціни на рис в USD для країн-експортерів:")
    print(pd.DataFrame({'year': X['year'].values, 'predicted_price_usd': y_pred}))

    plt.figure(figsize=(12, 7))
    plt.scatter(X, y, color='blue', label='Середня ціна в USD (реальна)')
    plt.plot(X, y_pred, color='red', linewidth=2, label='Лінія регресії (USD)')
    plt.title(f'Прогноз цін на "{COMMODITY}" в USD для країн-експортерів')
    plt.xlabel('Рік')
    plt.ylabel('Середня ціна (USD)')
    plt.legend()
    plt.grid(True)
    plt.show()

def perform_sensitivity_analysis(df, products_to_analyze):
    print("--- ЗАВДАННЯ 9: Аналіз чутливості ціни до року ---")
    
    sensitivities = {}

    for product in products_to_analyze:
        product_df = df[df['cm_name'] == product].copy()
        
        if product_df.empty or len(product_df['mp_year'].unique()) < 2:
            print(f"Недостатньо даних для аналізу чутливості для продукту: {product}")
            continue

        yearly_avg_price = product_df.groupby('mp_year')['mp_price'].mean().reset_index()

        X = yearly_avg_price[['mp_year']]
        y = yearly_avg_price['mp_price']

        model = LinearRegression()
        model.fit(X, y)

        sensitivity = model.coef_[0]
        sensitivities[product] = sensitivity
        
        print(f"Чутливість для '{product}': {sensitivity:.2f}.")

    if not sensitivities:
        print("Не вдалося розрахувати чутливість для жодного продукту.")
        return

    sensitivity_df = pd.DataFrame(list(sensitivities.items()), columns=['Product', 'Sensitivity (Price Change per Year)'])
    sensitivity_df = sensitivity_df.sort_values('Sensitivity (Price Change per Year)', ascending=False)

    plt.figure(figsize=(12, 8))
    sns.barplot(
        x='Sensitivity (Price Change per Year)', 
        y='Product', 
        data=sensitivity_df, 
        palette='viridis'
    )
    plt.title('Аналіз чутливості: Вплив року на ціну товару')
    plt.xlabel('Середня зміна ціни за рік (коефіцієнт регресії)')
    plt.ylabel('Товар')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    
    print("-" * 50 + "\n")

def analyze_product_categories(df, category1, category2):
    print(f"--- ЗАВДАННЯ 10: Аналіз товарних категорій та регресія між '{category1}' та '{category2}' ---")

    def get_category(cm_name):
        cm_name = cm_name.lower()
        if any(keyword in cm_name for keyword in ['rice', 'wheat', 'bread', 'maize', 'sorghum']):
            return 'Grains'
        elif any(keyword in cm_name for keyword in ['milk', 'eggs', 'cheese']):
            return 'Dairy & Eggs'
        elif any(keyword in cm_name for keyword in ['meat', 'fish', 'poultry']):
            return 'Meat & Fish'
        elif any(keyword in cm_name for keyword in ['oil']):
            return 'Oils & Fats'
        elif any(keyword in cm_name for keyword in ['tomatoes', 'onions', 'cabbage', 'beans', 'lentils', 'peas', 'potatoes']):
            return 'Produce & Legumes'
        else:
            return 'Other'

    df['category'] = df['cm_name'].apply(get_category)
    
    category_stats = df.groupby('category')['mp_price'].describe()
    print("\nСтатистика цін по товарних категоріях:")
    print(category_stats)

    df['date'] = pd.to_datetime(df['mp_year'].astype(str) + '-' + df['mp_month'].astype(str))
    monthly_category_prices = df.groupby(['date', 'category'])['mp_price'].mean().reset_index()

    plt.figure(figsize=(15, 8))
    sns.lineplot(data=monthly_category_prices, x='date', y='mp_price', hue='category', lw=2)
    plt.title('Середні цінові тренди по категоріях товарів')
    plt.xlabel('Дата')
    plt.ylabel('Середня ціна')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(title='Категорія')
    plt.show()

    aligned_prices = monthly_category_prices.pivot_table(
        index='date',
        columns='category',
        values='mp_price'
    )[[category1, category2]].dropna()

    if aligned_prices.empty or len(aligned_prices) < 2:
        print(f"\nНедостатньо спільних даних для побудови регресії між '{category1}' та '{category2}'.")
        return
        
    print(f"\nВирівняні дані для регресії ({len(aligned_prices)} записів):")
    print(aligned_prices.head())

    X = aligned_prices[[category1]]
    y = aligned_prices[category2]

    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    
    coef = model.coef_[0]
    r2 = r2_score(y, y_pred)
    
    print(f"\n--- Результати регресії ---")
    print(f"Коефіцієнт (slope): {coef:.2f}")
    print(f"R² score: {r2:.2f}")
    print(f"Інтерпретація: Зміна середньої ціни в категорії '{category1}' на 1 одиницю, \nв середньому призводить до зміни ціни в категорії '{category2}' на {coef:.2f} одиниць.")

    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, alpha=0.5, label='Реальні середньомісячні ціни')
    plt.plot(X, y_pred, color='red', linewidth=2, label=f'Лінія регресії (R²={r2:.2f})')
    plt.title(f'Залежність цін "{category2}" від цін "{category1}"')
    plt.xlabel(f'Середня ціна "{category1}"')
    plt.ylabel(f'Середня ціна "{category2}"')
    plt.legend()
    plt.grid(True)
    plt.show()
    print("-" * 50 + "\n")

def perform_advanced_sensitivity_analysis(df, product_to_analyze):
    print("--- ЗАВДАННЯ 11: Розширений аналіз чутливості ---")

    product_df = df[df['cm_name'] == product_to_analyze].copy()
    monthly_avg_price = product_df.groupby(['mp_year', 'mp_month'])['mp_price'].mean().reset_index()

    X = monthly_avg_price[['mp_year', 'mp_month']]
    y = monthly_avg_price['mp_price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    mlp_model = MLPRegressor(
        hidden_layer_sizes=(20, 10), 
        max_iter=1000,                
        activation='relu',            
        random_state=42
    )
    mlp_model.fit(X_train_scaled, y_train)

    base_input_unscaled = X_test.mean().values.reshape(1, -1)
    base_input_scaled = scaler.transform(base_input_unscaled)
    base_prediction = mlp_model.predict(base_input_scaled)[0]
    
    print(f"Базовий прогноз ціни для середніх значень (рік: {base_input_unscaled[0,0]:.0f}, місяць: {base_input_unscaled[0,1]:.0f}): {base_prediction:.2f}")

    scenarios = {
        "Зміна тільки року": [True, False],
        "Зміна тільки місяця": [False, True],
        "Комбінована зміна": [True, True]
    }
    percentage_changes = [-0.20, -0.10, -0.05, 0.05, 0.10, 0.20]
    results = []

    for scenario_name, mask in scenarios.items():
        for p in percentage_changes:
            new_input_unscaled = base_input_unscaled.copy()
            
            if mask[0]: 
                new_input_unscaled[0, 0] *= (1 + p)
            if mask[1]: 
                new_input_unscaled[0, 1] *= (1 + p)

            new_input_scaled = scaler.transform(new_input_unscaled)
            new_prediction = mlp_model.predict(new_input_scaled)[0]

            sensitivity = ((new_prediction - base_prediction) / base_prediction) * 100
            
            results.append({
                "input_change_pct": p * 100,
                "output_change_pct": sensitivity,
                "scenario": scenario_name
            })
    
    results_df = pd.DataFrame(results)

    plt.figure(figsize=(12, 8))
    sns.lineplot(
        data=results_df,
        x='input_change_pct',
        y='output_change_pct',
        hue='scenario',
        marker='o',
        linewidth=2.5
    )
    plt.title(f'Аналіз чутливості моделі для "{product_to_analyze}"')
    plt.xlabel('Відсоток зміни вхідних змінних (%)')
    plt.ylabel('Відсоток зміни прогнозованої ціни (%)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.legend(title='Сценарій аналізу')
    plt.show()

    print("\n--- Висновки аналізу чутливості ---")
    max_impact_scenario = results_df.loc[results_df['output_change_pct'].abs().idxmax()]
    print(f"Найбільший вплив на ціну має сценарій: '{max_impact_scenario['scenario']}'.")
    print(f"При зміні вхідних даних на {max_impact_scenario['input_change_pct']:.0f}%, прогнозована ціна змінюється на {max_impact_scenario['output_change_pct']:.2f}%.")
    print("Аналіз графіка показує, що зміна року ('mp_year') має значно більший вплив на прогнозовану ціну, ніж зміна місяця ('mp_month').")
    print("Комбінована зміна має найбільший ефект, оскільки вона поєднує вплив обох факторів.")
    print("-" * 50 + "\n")

if __name__ == "__main__":
    dataset = pd.read_csv("global_food_prices.csv", low_memory=False)
    dataset_filtered = dataset[dataset['mp_year'] >= 2007].copy()
    rates_df = pd.read_csv("rates.csv", low_memory=False)
    PRODUCT_TO_ANALYSE = "Rice - Retail"
    MARKET_1 = 'Osaka' 
    MARKET_2 = 'Tokyo'

    perform_statistical_analysis(dataset_filtered)
        
    visualize_price_trends(dataset_filtered, PRODUCT_TO_ANALYSE)
        
    preprocess_and_visualize_encoding(dataset_filtered)
        
    perform_correlation_analysis(dataset_filtered)

    res_lr = forecasting_using_linear_regression(dataset_filtered, PRODUCT_TO_ANALYSE)

    res_mlp = forecasting_using_mlp(dataset_filtered, PRODUCT_TO_ANALYSE)

    print(res_mlp)

    print("\n--- Порівняння точності моделей ---")
    print(f"{'Metric':<10} | {'MLP Regressor':<15} | {'Linear Regression':<20}")
    print("-" * 55)
    print(f"{'MAE':<10} | {res_mlp["mae_mlp"]:<15.2f} | {res_lr["mae_lr"]:<20.2f}")
    print(f"{'RMSE':<10} | {res_mlp["rmse_mlp"]:<15.2f} | {res_lr["rmse_lr"]:<20.2f}")
    print(f"{'R² Score':<10} | {res_mlp["r2_mlp"]:<15.2f} | {res_lr["r2_lr"]:<20.2f}")
    analysis_of_market_relationships(dataset_filtered, PRODUCT_TO_ANALYSE, MARKET_1, MARKET_2)
    analysis_of_the_impact_of_currency(dataset_filtered, rates_df, PRODUCT_TO_ANALYSE)

    products_for_sensitivity = [
        "Rice - Retail",
        "Bread - Retail",
        "Milk - Retail",
        "Eggs - Retail",
        "Wheat flour - Retail"
    ]
    perform_sensitivity_analysis(dataset_filtered, products_for_sensitivity)
    analyze_product_categories(dataset_filtered, category1='Grains', category2='Dairy & Eggs')
    perform_advanced_sensitivity_analysis(dataset_filtered, PRODUCT_TO_ANALYSE)