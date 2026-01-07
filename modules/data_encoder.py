import pandas as pd
from modules.smoothed_target_encoder import SmoothedTargetEncoder
from modules.cyclical_encoder import CyclicalEncoder

class DataEncoder:
    def __init__(self):
        self.final_features = [
            'state_origin', 'state_dest', 'distance',
            'origin_airport', 'destination_airport', 'airline',
            'dep_dayofyear_sin', 'dep_dayofyear_cos',
            'dep_hour_sin', 'dep_hour_cos',
            'arr_dayofyear_sin', 'arr_dayofyear_cos',
            'arr_hour_sin', 'arr_hour_cos'
        ]
        
        self.target_encoders = {
            'distance': SmoothedTargetEncoder('distance', 'is_delayed'),
            'origin_airport': SmoothedTargetEncoder('origin_airport', 'is_delayed'),
            'destination_airport': SmoothedTargetEncoder('destination_airport', 'is_delayed'),
            'airline': SmoothedTargetEncoder('airline', 'is_delayed'),
            'state_origin': SmoothedTargetEncoder('state_origin', 'is_delayed'),
            'state_dest': SmoothedTargetEncoder('state_dest', 'is_delayed')
        }
        
        self.cyclical_steps = [
            CyclicalEncoder('dep_dayofyear', 365),
            CyclicalEncoder('dep_hour', 24),
            CyclicalEncoder('arr_dayofyear', 365),
            CyclicalEncoder('arr_hour', 24)
        ]

    def _add_time_features(self, df):
        df = df.copy()
        df['scheduled_departure'] = pd.to_datetime(df['scheduled_departure'])
        df['scheduled_arrival']   = pd.to_datetime(df['scheduled_arrival'])

        df['dep_hour'] = df['scheduled_departure'].dt.hour + df['scheduled_departure'].dt.minute / 60
        df['dep_dayofyear'] = df['scheduled_departure'].dt.dayofyear
        df['arr_hour'] = df['scheduled_arrival'].dt.hour + df['scheduled_arrival'].dt.minute / 60
        df['arr_dayofyear'] = df['scheduled_arrival'].dt.dayofyear
        return df

    def fit_transform(self, X, y):
        """Fits on Training data only."""
        X = self._add_time_features(X)
        
        for col, enc in self.target_encoders.items():
            data_to_fit = pd.concat([X[col], y], axis=1)
            X[col] = enc.fit_transform(data_to_fit)
            
        for enc in self.cyclical_steps:
            X = enc.transform(X)
            
        return X[self.final_features]

    def transform(self, X):
        """Transforms Test or New Prediction data."""
        X = self._add_time_features(X)
        
        for col, enc in self.target_encoders.items():
            X[col] = enc.transform(X[col])
            
        for enc in self.cyclical_steps:
            X = enc.transform(X)
            
        return X[self.final_features]