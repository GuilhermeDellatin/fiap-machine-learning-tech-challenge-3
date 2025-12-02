import numpy as np
import pandas as pd

class CyclicalEncoder:
    """
    Cyclical encoder using Sin/Cos transformation.
    """

    def __init__(self, feature_name, period):
        self.feature_name = feature_name
        self.period = period
    
    def fit(self, df):
        """Nothing to learn — included for pipeline compatibility."""
        return self
    
    def transform(self, df):
        df = df.copy()
        df[f"{self.feature_name}_sin"] = np.sin(2 * np.pi * df[self.feature_name] / self.period)
        df[f"{self.feature_name}_cos"] = np.cos(2 * np.pi * df[self.feature_name] / self.period)
        return df.drop(columns=[self.feature_name])
    
    def fit_transform(self, df):
        return self.fit(df).transform(df)
