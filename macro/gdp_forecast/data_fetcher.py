from fredapi import Fred
import pandas as pd 
from dotenv import load_dotenv
import os

load_dotenv()


class DataFetcher:
    def __init__(self):
        fred_api_key = os.getenv("FRED_API_KEY")
        if not fred_api_key:
            raise ValueError("Missing Fred Api Key!")
        self.fred = Fred(api_key=fred_api_key)

    def get_series(self, series_ids: dict):
        data = {}
        for series_name, series_id in series_ids.items():
            try: 
                print(f'Fetching {series_name}...')
                series = self.fred.get_series_latest_release(series_id)
                data[series_name] = series
            except Exception as e:
                print(f'Error fetching {series_name}: {e}')
        data = pd.DataFrame(data)
        data.sort_index(inplace=True)
        return data
    

if __name__ == "__main__":
    f = DataFetcher()
    series_ids = {'GDP':'GDP', 'RealGdp':'GDPC1'}
    data = f.get_series(series_ids)
    print(data.tail())
    
    test_size = 0.1
    n = len(data)
    split_idx = int(n * (1-test_size))
    print((split_idx))
    