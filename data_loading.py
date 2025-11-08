import zipfile
import pandas as pd
import pymongo
from io import TextIOWrapper
import os
from config import MONGO_URI, DB_NAME, COLLECTION_NAME, ZIP_PATH, POLLUTANTS


def load_data_to_mongodb():
    """Load all pollutant data from ZIP file to MongoDB"""
    try:
        client = pymongo.MongoClient(MONGO_URI)
        db = client[DB_NAME]
        collection = db[COLLECTION_NAME]
        
        # Clear existing data
        collection.delete_many({})
        
        with zipfile.ZipFile(ZIP_PATH, "r") as z:
            file_list = [f for f in z.namelist() if f.endswith(".csv")]
            print(f"Found {len(file_list)} CSV files in your dataset")
            
            for filename in file_list:
                print(f"Processing: {filename}")
                with z.open(filename) as f:
                    df = pd.read_csv(TextIOWrapper(f, 'utf-8'))
                    
                    # Clean column names
                    df.columns = [
                        col.strip().lower()
                        .replace(" ", "_")
                        .replace("#", "num")
                        .replace(".", "")
                        .replace("(", "")
                        .replace(")", "")
                        for col in df.columns
                    ]
                    
                    # Add station name
                    station_name = os.path.basename(filename).replace(".csv", "")
                    df["station_name"] = station_name
                    
                    # Detect date column
                    possible_dates = ['from_date', 'date', 'from', 'timestamp', 'datetime']
                    date_col = next((col for col in possible_dates if col in df.columns), None)
                    
                    if date_col:
                        df['date'] = pd.to_datetime(df[date_col], errors='coerce')
                    else:
                        df['date'] = pd.date_range(start='2019-01-01', periods=len(df), freq='D')
                    
                    df = df.dropna(subset=['date'])
                    
                    records = df.to_dict(orient="records")
                    if records:
                        collection.insert_many(records)
                        print(f" Loaded {len(records)} records from {filename}")
        
        print("All historical data loaded successfully!")
        client.close()
        return True
        
    except Exception as e:
        print(f"Error loading your data: {e}")
        return False


def get_city_pollutant_data(city_name, pollutant='pm25'):
    """Get historical data for a specific pollutant for a city"""
    try:
        client = pymongo.MongoClient(MONGO_URI)
        db = client[DB_NAME]
        collection = db[COLLECTION_NAME]
        
        query = {"station_name": {"$regex": city_name, "$options": "i"}}
        cursor = collection.find(query)
        df = pd.DataFrame(list(cursor))
        
        if df.empty:
            print(f" No data found for {city_name} in your dataset")
            return pd.DataFrame()
        

        # Only 5 pollutants (CO removed)
        pollutant_variations = {
            'pm25': ['pm25', 'pm2.5', 'pm_25', 'pm25_ug/m3'],
            'pm10': ['pm10', 'pm_10', 'pm10_ug/m3'],
            'o3': ['o3', 'ozone', 'o3_ug/m3'],
            'no2': ['no2', 'nitrogen_dioxide', 'no2_ug/m3'],
            'so2': ['so2', 'sulfur_dioxide', 'so2_ug/m3']
        }
        
        pollutant_col = None
        for col in df.columns:
            col_lower = col.lower()
            for variation in pollutant_variations.get(pollutant, []):
                if variation in col_lower:
                    pollutant_col = col
                    break
            if pollutant_col:
                break
        
        if not pollutant_col:
            print(f"{pollutant.upper()} data not found for {city_name}")
            return pd.DataFrame()
        
        clean_df = df[['date', pollutant_col]].copy()
        clean_df = clean_df.rename(columns={pollutant_col: pollutant})
        clean_df['date'] = pd.to_datetime(clean_df['date'])
        clean_df = clean_df.sort_values('date')
        
        clean_df = clean_df.dropna()
        clean_df = clean_df[(clean_df[pollutant] > 0) & (clean_df[pollutant] < 10000)]
        
        print(f"{city_name} - {pollutant.upper()}: {len(clean_df)} records")
        return clean_df
        
    except Exception as e:
        print(f"Error getting data for {city_name} - {pollutant}: {e}")
        return pd.DataFrame()


def get_all_pollutants_data(city_name):
    """Get all pollutant data for a city"""
    all_data = {}
    for pollutant in POLLUTANTS:  # Now only 5 pollutants
        data = get_city_pollutant_data(city_name, pollutant)
        if not data.empty:
            all_data[pollutant] = data
    return all_data


if __name__ == "__main__":
    print("Testing data loader...")
    load_data_to_mongodb()
    
    print("\nTesting data retrieval for all cities...")

    #  Only these 4 cities are used in project
    cities = ["Delhi", "Mumbai", "Chennai", "Kolkata"]

    for city in cities:
        print(f"\n Checking data for: {city}")
        data = get_all_pollutants_data(city)
        print(f"  Found data for {len(data)} pollutants")
