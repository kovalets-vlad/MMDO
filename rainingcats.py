import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
import time

df = pd.read_csv('cat_data2.csv', sep=';', decimal=',')

df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

features = [
    'Age_in_years', 'Gender', 'Neutered_or_spayed', 'Body_length',
    'Allowed_outdoor', 'Preferred_food', 'Owner_play_time_minutes',
    'Sleep_time_hours', 'Breed', 'Fur_colour_dominant', 'Fur_pattern', 'Eye_colour'
]
target = 'Weight'

df[target] = pd.to_numeric(df[target], errors='coerce')
df.dropna(subset=[target] + features, inplace=True)

X = df[features]
y = df[target]

cat_cols = ['Gender', 'Neutered_or_spayed', 'Allowed_outdoor', 'Preferred_food', 
            'Breed', 'Fur_colour_dominant', 'Fur_pattern', 'Eye_colour']
num_cols = [col for col in features if col not in cat_cols]

encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
X_encoded = encoder.fit_transform(X[cat_cols])
X_numerical = X[num_cols].values
X_final = np.hstack((X_numerical, X_encoded))

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_final)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

alphas = [0.001, 0.01, 0.1, 1.0]
results = []

for alpha in alphas:
    lasso = Lasso(alpha=alpha, max_iter=10000, tol=1e-6) 
    start_time = time.time()
    
    lasso.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    
    y_pred = lasso.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    iterations = lasso.n_iter_
    
    results.append({
        'Alpha': alpha,
        'Training Time (s)': round(training_time, 4),
        'Iterations': iterations,
        'MSE': round(mse, 4),
        'RMSE': round(rmse, 4)
    })

results_df = pd.DataFrame(results)
print("\nТаблиця порівняння для різних значень alpha:")
print(results_df)

best_alpha = results_df.loc[results_df['MSE'].idxmin(), 'Alpha']
lasso_best = Lasso(alpha=best_alpha, max_iter=10000, tol=1e-4)
lasso_best.fit(X_train, y_train)
feature_names = num_cols + list(encoder.get_feature_names_out(cat_cols))
coefficients = pd.Series(lasso_best.coef_, index=feature_names)
print(f"\nНенульові коефіцієнти для alpha={best_alpha}:")
print(coefficients[coefficients != 0].sort_values(ascending=False))