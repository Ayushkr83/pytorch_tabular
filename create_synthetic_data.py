import pandas as pd
import numpy as np
from sklearn.datasets import make_classification

def main():
    print("Generating synthetic data...")
    X, y = make_classification(
        n_samples=10000, 
        n_features=10, 
        n_informative=5, 
        n_classes=2, 
        random_state=42
    )

    # We will use the first 8 features as continuous
    df = pd.DataFrame(X[:, :8], columns=[f'num_col_{i}' for i in range(1, 9)])

    # We will convert the last 2 features into categorical strings
    df['cat_col_1'] = pd.qcut(X[:, 8], q=4, labels=['A', 'B', 'C', 'D']).astype(str)
    df['cat_col_2'] = pd.qcut(X[:, 9], q=3, labels=['X', 'Y', 'Z']).astype(str)

    # Add target column
    df['target_column'] = y

    # Save to CSV
    output_file = 'synthetic_data.csv'
    df.to_csv(output_file, index=False)
    print(f"Successfully generated 10,000 rows of synthetic data in '{output_file}'!")

if __name__ == "__main__":
    main()
