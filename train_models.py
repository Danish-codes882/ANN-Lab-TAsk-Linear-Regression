"""Train both Linear Regression models and print their test metrics."""
from regression_core import train_all

if __name__ == "__main__":
    for result in train_all():
        print(f"\n{result['title']}")
        print(f"Rows: {result['rows']} | Train: {result['train_rows']} | Test: {result['test_rows']}")
        print(f"MAE: {result['mae']}")
        print(f"RMSE: {result['rmse']}")
        print(f"R²: {result['r2']}")
    print("\nSaved models, metrics, and plots in the artifacts/ folder.")
