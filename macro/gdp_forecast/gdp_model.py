from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np 
import pandas as pd


class GDPModel:
    def __init__(self, X, y):
        self.X = X
        self.y = y 
        self.model = None
    
    def train_test_split(self, n=4):
        split_idx = len(self.X) - n
        X_train, X_test = self.X[:split_idx], self.X[split_idx:]
        y_train, y_test = self.y[:split_idx], self.y[split_idx:]  
        return X_train, X_test, y_train, y_test

    def train(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def evaluate(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        score = r2_score(y_test, y_pred)
        print(f"Model R2 score: {score:.4f}")
        return score
    
    
class ShortTermGDPModel(GDPModel):
    def train(self):
        from sklearn.linear_model import LinearRegression
        
        X_train, X_test, y_train, y_test = self.train_test_split()
        self.model = LinearRegression()
        self.model.fit(X_train, y_train)
        self.evaluate(X_test, y_test)
       

class LongTermGDPModel(GDPModel):
    def train(self):
        from sklearn.ensemble import RandomForestRegressor
        
        X_train, X_test, y_train, y_test = self.train_test_split()
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        self.evaluate(X_test, y_test)
        


