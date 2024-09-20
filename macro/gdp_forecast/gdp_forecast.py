from data_fetcher import DataFetcher
from feature_engineer import FeatureEngineer
from gdp_model import ShortTermGDPModel, LongTermGDPModel


class GDPForecast:
    def __init__(self):
        self.f = DataFetcher()
        self.series_ids = {
            'gdp': 'GDP',
            'unemployment_rate': 'UNRATE',
            'consumer_spending': 'PCE',
            'industrial_production': 'INDPRO',
            # Add more series as needed
        }

    def fetch_data(self):
        data = self.f.get_series(self.series_ids)
        print("Data fetched successfully.\n")
        return data

    def feature_engineering(self, data):
        feature_engineer = FeatureEngineer(data)
        feature_engineer.handle_missing_values()
        column_names = ['gdp', 'unemployment_rate', 'consumer_spending', 'industrial_production']
        feature_engineer.create_features(column_names)
        print("Features created successfully.\n")
        return feature_engineer

    def run(self):
        data = self.fetch_data()
        feature_engineer = self.feature_engineering(data)
        X, y = feature_engineer.get_features_and_target(target_variable='gdp')

        print("Training short-term GDP model...")
        short_term_model = ShortTermGDPModel(X, y)
        short_term_model.train()

        print("\nTraining long-term GDP model...")
        long_term_model = LongTermGDPModel(X, y)
        long_term_model.train()


if __name__ == '__main__':
    gdp_forecast = GDPForecast()
    gdp_forecast.run()
