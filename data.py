import pandas as pd
import os
from pymongo import MongoClient
import json

# Configure MongoDB client for Sauron
db = MongoClient(host="mongodb07.nersc.gov", 
                 username="alab_completed_ro",
                 password="CEDERALAB_RO", 
                 authSource="alab_completed")["alab_completed"]
collection = db['samples']

# MongoDB connection details for Dara
host = "mongodb07.nersc.gov"
username = "olympiadartsi"
password = "ChVMtfb4mU5M"
database_name = "alab-refinement"
collection_name = "results"

# Form the connection string
connection_string = f"mongodb://{username}:{password}@{host}/{database_name}"

# Connect to MongoDB
client2 = MongoClient(connection_string)
db2 = client2[database_name]
collection2 = db2[collection_name]

# Cache file paths
CACHE_FILE = 'datasets/df_cache.csv'
CACHE_FILE2 = 'datasets/df_cache2.csv'

df_cache = None
df_cache2 = None

def serialize_complex(data):
    """
    Serializes complex data structures into JSON strings.
    """
    if isinstance(data, (dict, list)):
        return json.dumps(data)
    return data

def deserialize_complex(data):
    """
    Deserializes JSON strings back into complex data structures.
    """
    try:
        return json.loads(data)
    except (TypeError, json.JSONDecodeError):
        return data

def deserialize_metadata_column(df):
    """
    Deserializes the 'metadata' and 'output' columns in the dataframe.
    """
    df['metadata'] = df['metadata'].apply(deserialize_complex)
    if 'output' in df.columns:
        df['output'] = df['output'].apply(deserialize_complex)
    return df

def cache_data(fetch_function):
    cache = {}
    def wrapper(*args, **kwargs):
        if fetch_function.__name__ in cache:
            print("Using cached data from memory.")
            return cache[fetch_function.__name__]
        result = fetch_function(*args, **kwargs)
        cache[fetch_function.__name__] = result
        return result
    
    return wrapper


@cache_data
def fetch_data():
    global df_cache
    if df_cache is not None:
        print("Using global cached data.")
        return df_cache
    
    if os.path.exists(CACHE_FILE):
        print("Loading data from local cache file.")
        df_cache = pd.read_csv(CACHE_FILE)
        df_cache = deserialize_metadata_column(df_cache)
        return df_cache

    try:
        print("Querying MongoDB for Sauron data.")
        data = list(collection.find())
        if not data:
            print("No data found in MongoDB.")
            df_cache = pd.DataFrame()  
            return df_cache
        
        df_cache = pd.DataFrame(data)
        df_cache['metadata'] = df_cache['metadata'].apply(serialize_complex)  # Serialize metadata
        df_cache.to_csv(CACHE_FILE, index=False)  # Save to local cache file
        print(f"Fetched {len(df_cache)} records from MongoDB.")
        return df_cache
    except Exception as e:
        print(f"Error fetching data from MongoDB: {e}")
        return pd.DataFrame()

@cache_data
def fetch_data2():
    global df_cache2
    if df_cache2 is not None:
        print("Using global cached data.")
        return df_cache2
    
    if os.path.exists(CACHE_FILE2):
        print("Loading data from local Dara cache file.")
        df_cache2 = pd.read_csv(CACHE_FILE2)
        df_cache2 = deserialize_metadata_column(df_cache2)
        return df_cache2

    try:
        print("Querying MongoDB for Dara data.")
        data = list(collection2.find())
        if not data:
            print("No data found in MongoDB.")
            df_cache2 = pd.DataFrame()  
            return df_cache2
        
        df_cache2 = pd.DataFrame(data)
        df_cache2['metadata'] = df_cache2['metadata'].apply(serialize_complex)  # Serialize metadata
        df_cache2['output'] = df_cache2['output'].apply(serialize_complex)  # Serialize output if present
        df_cache2.to_csv(CACHE_FILE2, index=False)  # Save to local cache file
        print(f"Fetched {len(df_cache2)} records from MongoDB.")
        return df_cache2
    except Exception as e:
        print(f"Error fetching data from MongoDB: {e}")
        return pd.DataFrame()
