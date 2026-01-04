import pandas as pd
import os


def min_max_fit(col):
    """Calculate min and max values for a column."""
    return col.min(), col.max()


def min_max_transform(x, min_val, max_val):
    """Apply min-max normalization to a single value."""
    if max_val == min_val:
        return 0.0
    return (x - min_val) / (max_val - min_val)


def normalize_dataframe(df, feature_cols):
    """
    Apply min-max normalization to specified feature columns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    feature_cols : list
        List of column names to normalize
        
    Returns:
    --------
    df_norm : pd.DataFrame
        Normalized dataframe
    params : dict
        Dictionary mapping column names to (min, max) tuples
    """
    params = {}
    df_norm = df.copy()

    for col in feature_cols:
        min_val, max_val = min_max_fit(df[col])
        params[col] = (min_val, max_val)
        df_norm[col] = df[col].apply(lambda x: min_max_transform(x, min_val, max_val))

    return df_norm, params


def apply_normalize(df, feature_cols, params):
    """
    Apply min-max normalization using pre-computed parameters.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    feature_cols : list
        List of column names to normalize
    params : dict
        Dictionary mapping column names to (min, max) tuples
        
    Returns:
    --------
    df_norm : pd.DataFrame
        Normalized dataframe
    """
    df_norm = df.copy()

    for col in feature_cols:
        min_val, max_val = params[col]
        if max_val == min_val:
            df_norm[col] = 0.0
        else:
            df_norm[col] = (df[col] - min_val) / (max_val - min_val)

    return df_norm


def clean_and_normalize_iris(input_path, output_path):
    """
    Read IRIS dataset, normalize features, and save to new file.
    
    Parameters:
    -----------
    input_path : str
        Path to input CSV file
    output_path : str
        Path to output CSV file
    """
    # Read the original data
    df = pd.read_csv(input_path)
    
    print("=== THONG TIN DU LIEU GOC ===")
    print(f"So luong mau: {len(df)}")
    print(f"Cac cot: {list(df.columns)}")
    print(f"\nThong ke mo ta:\n{df.describe()}")
    
    # Define feature columns
    feature_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    
    # Normalize the data
    df_normalized, params = normalize_dataframe(df, feature_cols)
    
    print("\n=== THAM SO CHUAN HOA (MIN-MAX) ===")
    for col, (min_val, max_val) in params.items():
        print(f"{col}: min = {min_val}, max = {max_val}")
    
    print(f"\n=== THONG KE SAU CHUAN HOA ===")
    print(df_normalized.describe())
    
    # Save to new file
    df_normalized.to_csv(output_path, index=False)
    print(f"\n=== DA LUU FILE CHUAN HOA ===")
    print(f"Output: {output_path}")
    
    return df_normalized, params


if __name__ == "__main__":
    # Get the project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    input_file = os.path.join(project_root, "data", "IRIS.csv")
    output_file = os.path.join(project_root, "data", "IRIS_cleaned.csv")
    output_file_ohe = os.path.join(project_root, "data", "IRIS_cleaned_ohe.csv")
    
    # Step 1: Normalize
    df_cleaned, params = clean_and_normalize_iris(input_file, output_file)
    
    # Step 2: Split data set into train and test (80%-20%) and save
    from sklearn.model_selection import train_test_split
    df_train, df_test = train_test_split(df_cleaned, test_size=0.2, random_state=62, stratify=df_cleaned["species"])
    df_train.to_csv(os.path.join(project_root, "data", "IRIS_train.csv"), index=False)
    df_test.to_csv(os.path.join(project_root, "data", "IRIS_test.csv"), index=False)
