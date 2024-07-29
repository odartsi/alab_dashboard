import pandas as pd
from pymongo import MongoClient

# Configure MongoDB client
db= MongoClient(host="mongodb07.nersc.gov", 
                    username="alab_completed_ro",
                    password="CEDERALAB_RO", 
                    authSource="alab_completed")["alab_completed"]
collection= db['samples']

# MongoDB connection details -- Dara
host = "mongodb07.nersc.gov"
username = "olympiadartsi"
password = "ChVMtfb4mU5M"
database_name = "alab-refinement"
collection_name = "results"

# Form the connection string
connection_string = f"mongodb://{username}:{password}@{host}/{database_name}"

# Connect to MongoDB
client2 = MongoClient(connection_string)

# Access the database and collection
db2 = client2[database_name]
collection2 = db2[collection_name]


df_cache = None
df_cache2 = None

def cache_data(fetch_function):
    cache = {}

    def wrapper(*args, **kwargs):
        if fetch_function.__name__ in cache:
            print("Using cached data.")
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
    try:
        print("Querying MongoDB for Sauron data.")
        data = list(collection.find())
        if not data:
            print("No data found in MongoDB.")
            df_cache = pd.DataFrame()  # return empty DataFrame if no data
            return df_cache
        df_cache = pd.DataFrame(data)
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
    try:
        print("Querying MongoDB for Dara data.")
        data = list(collection2.find())
        if not data:
            print("No data found in MongoDB.")
            df_cache2 = pd.DataFrame()  # return empty DataFrame if no data
            return df_cache2
        df_cache2 = pd.DataFrame(data)
        print(f"Fetched {len(df_cache2)} records from MongoDB.")
        return df_cache2
    except Exception as e:
        print(f"Error fetching data from MongoDB: {e}")
        return pd.DataFrame()