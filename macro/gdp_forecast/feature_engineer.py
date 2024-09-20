import pandas as pd
import numpy as np
from typing import List


class FeatureEngineerError(Exception):
    pass

class MissingColumnError(FeatureEngineerError):
    def __init__(self, column_name: str):
        self.message = f"Column '{column_name}' not found in the dataframe."
        super().__init__(self.message)

class TargetVariableError(FeatureEngineerError):  
    def __init__(self, target_variable: str):
        self.message = f"Target variable '{target_variable}' not found in the dataframe."
        super().__init__(self.message)


class FeatureEngineer:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        
    def handle_missing_values(self):
        self.data.fillna(method='ffill', inplace=True)
        self.data.dropna(inplace=True)
        
    def create_features(self, column_names: List[str]):
        for column_name in column_names:
            if column_name not in self.data.columns:  
                raise MissingColumnError(column_name)
            self.data[f'{column_name}_change'] = self.data[column_name].pct_change()
            self.data[f'{column_name}_lag1'] = self.data[column_name].shift(1)
        self.data.dropna(inplace=True)

    def get_features_and_target(self, target_variable: str):
        if target_variable not in self.data.columns:
            raise TargetVariableError(target_variable)
        
        X = self.data.drop(columns=[target_variable])
        y = self.data[target_variable]
        return X, y
