# This file will contain methods to evaluating both algorithms the same way (and some other utilities)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score

def load_data():
    df = pd.read_csv('steel.csv')
    print(f"Loaded {len(df)} rows from CSV")
    return df

def split_data(df):
    X = df.drop('tensile_strength', axis=1)
    y = df['tensile_strength']

    kfold = KFold(n_splits=10, shuffle=True, random_state=42)
    print(f"Data split into 10 folds")

    return X, y, kfold


def evaluate_model(model, X_train, y_train, X_test, y_test):
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)

    return {
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_mae': train_mae,
        'test_mae': test_mae
    }


def cross_validate_model(model_class, X, y, kfold, **model_params):
    results = {
        'train_r2': [], 'test_r2': [],
        'train_mae': [], 'test_mae': []
    }

    for fold_num, (train_idx, test_idx) in enumerate(kfold.split(X), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = model_class(**model_params)
        model.fit(X_train, y_train)

        scores = evaluate_model(model, X_train, y_train, X_test, y_test)

        results['train_r2'].append(scores['train_r2'])
        results['test_r2'].append(scores['test_r2'])
        results['train_mae'].append(scores['train_mae'])
        results['test_mae'].append(scores['test_mae'])

        print(f"Fold {fold_num}:")
        print(f"  Train - R²: {scores['train_r2']:.4f}, MAE: {scores['train_mae']:.2f}")
        print(f"  Test  - R²: {scores['test_r2']:.4f}, MAE: {scores['test_mae']:.2f}")


    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].boxplot([results['test_r2'], results['train_r2']])
    axes[0].set_title('R²', fontweight='bold')
    axes[0].set_ylabel('R²')
    axes[0].set_xticklabels(['Test R²', 'Train R²'])

    axes[1].boxplot([results['test_mae'], results['train_mae']])
    axes[1].set_title('MAE', fontweight='bold')
    axes[1].set_ylabel('MAE')
    axes[1].set_xticklabels(['Test MAE', 'Train MAE'])

    plt.tight_layout()
    plt.show()

    print(f"Mean Results Across Folds:")
    print(f"  Train R²:  {np.mean(results['train_r2']):.4f}")
    print(f"  Test R²:   {np.mean(results['test_r2']):.4f}")
    print(f"  Train MAE: {np.mean(results['train_mae']):.2f}")
    print(f"  Test MAE:  {np.mean(results['test_mae']):.2f}")

    return results


def get_optimal_params(model, param_grid, X, y, kfold):
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=kfold,
        scoring='r2',
        return_train_score=True,
        n_jobs=-1
    )

    grid_search.fit(X, y)

    print(f"Best params: {grid_search.best_params_}")

    return grid_search.best_params_