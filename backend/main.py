
import json
import pandas as pd

from fastapi import FastAPI, File, UploadFile

app = FastAPI()

# Path to the JSON file
DATA_FILE = "data.json"


def load_data() -> pd.DataFrame:
    """
    Load the data from the JSON file into a pandas DataFrame.
    
    Returns:
    pd.DataFrame. The data loaded from the JSON file.
    """
    with open(DATA_FILE, "r") as f:
        data = json.load(f)
    df = pd.DataFrame.from_dict(data)
    return df


def save_data(df: pd.DataFrame) -> None:
    """
    Save the data from a pandas DataFrame to the JSON file.
    
    Args:
    df: pd.DataFrame. The data to be saved to the JSON file.
    """
    data = df.to_dict(orient="records")
    with open(DATA_FILE, "w") as f:
        json.dump(data, f)


@app.post('/structured-data-eda')
async def structured_data_eda(file: UploadFile):
    """
    Endpoint to perform EDA on the uploaded data.
    """
    file_ext = file.filename.split(".")[-1]

    if file_ext not in ["xlsx", "csv"]:
        return {"Error": "Invalid file format. Please upload either an Excel file or a CSV file."}

    if file_ext == 'xlsx':
        df = pd.read_excel(file.file)
    else:
        df = pd.read_csv(file.file)

    # Save the data to the JSON file
    save_data(df)

    row, col = df.shape
    summary = df.describe()
    unique_vals = df.nunique()
    count_missing_values = df.isna().sum()
    data_types = df.dtypes.astype(str).to_dict()
    first_five = df.head(5)
    last_five = df.tail(5)

    eda_results = {
        'num_rows': row,
        'num_cols': col,
        'summary_stats': summary.to_dict(),
        'unique_values': unique_vals.to_dict(),
        'count_missing_values': count_missing_values.to_dict(),
        'data_type': data_types,
        'first_five': first_five.to_dict(),
        'last_five': last_five.to_dict(),
    }

    return eda_results

@app.get('/get_data/{column_name}')
async def get_column_data(column_name: str):
    try:
        df = load_data()
        return df[column_name]
    except Exception as e:
        return f"{e}"

@app.put('/update_column/{column_name}')
async def update_column_endpoint(column_name: str, new_column_name: str):
    """
    Endpoint to update the name of a column in the data.
    """
    # Load the data from the JSON file
    df = load_data()

    # Check if the column exists in the DataFrame
    if column_name not in df.columns:
        return {'message': f"Column '{column_name}' not found in DataFrame."}

    # Rename the specified column
    df = df.rename(columns={column_name: new_column_name})

    # Save the updated data to the JSON file
    save_data(df)

    # Return confirmation message
    return {'message': f"Column '{column_name}' renamed to '{new_column_name}' in DataFrame."}


@app.delete('/delete_column/{column_name}')
async def delete_column_endpoint(column_name: str):

    """
    Endpoint to delete a specified column from the data.
    """
    # Load the data from the JSON file
    df = load_data()

    # Check if the column exists in the DataFrame
    if column_name not in df.columns:
        return {'message': f"Column '{column_name}' not found in DataFrame."}

    # Delete the specified column
    df = df.drop(column_name, axis=1)
    return {"message": f"Column '{column_name}' deleted from DataFrame."}

