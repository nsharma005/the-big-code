import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Path to input CSV files
DATA_INPUT_DIR = BASE_DIR / "data" / "input"


def load_all_comments(): 
    files = [
        "Youtube01-Psy.csv",
        "Youtube02-KatyPerry.csv",
        "Youtube03-LMFAO.csv",
        "Youtube04-Eminem.csv",
        "Youtube05-Shakira.csv"
    ]

    dfs = []

    for f in files:
        file_path = DATA_INPUT_DIR / f
        df = pd.read_csv(file_path)
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True) 

    combined = combined.rename(columns={
        "CONTENT": "text",
    })

    combined["text"] = combined["text"].astype(str)
    combined = combined.drop_duplicates(subset=["text"])

    return combined


def load_real_users():
    users_path = BASE_DIR / "data" / "users.csv" 
    df = pd.read_csv(users_path)

    return df