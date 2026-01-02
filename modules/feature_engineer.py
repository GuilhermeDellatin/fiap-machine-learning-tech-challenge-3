import pandas as pd

class FlightFeatureEngineer:
   """
   Creates additional features to improve flight delay prediction.
   """


   def __init__(self):
       self.peak_hours = [6, 7, 8, 9, 17, 18, 19, 20]  # Morning and evening rush
       self.redeye_hours = [0, 1, 2, 3, 4, 5]  # Late night / early morning flights


   def fit(self, df):
       """Nothing to learn — included for pipeline compatibility."""
       return self


   def transform(self, df):
       """
       Create additional features from scheduled departure/arrival times.


       Expected columns:
           - scheduled_departure (datetime)
           - scheduled_arrival (datetime)
           - day_of_week (int)
       """
       df = df.copy()


       # Ensure datetime columns
       if not pd.api.types.is_datetime64_any_dtype(df['scheduled_departure']):
           df['scheduled_departure'] = pd.to_datetime(df['scheduled_departure'])
       if not pd.api.types.is_datetime64_any_dtype(df['scheduled_arrival']):
           df['scheduled_arrival'] = pd.to_datetime(df['scheduled_arrival'])


       # === Time-based features ===


       # Peak hour indicator (morning and evening rush hours)
       df['is_peak_hour'] = df['scheduled_departure'].dt.hour.isin(self.peak_hours).astype(int)


       # Red-eye flight indicator (late night / early morning)
       df['is_redeye'] = df['scheduled_departure'].dt.hour.isin(self.redeye_hours).astype(int)


       # Weekend indicator
       df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)  # Saturday=5, Sunday=6


       # Day of month (beginning/end of month may have more traffic)
       df['day_of_month'] = df['scheduled_departure'].dt.day


       # Week of year
       df['week_of_year'] = df['scheduled_departure'].dt.isocalendar().week.astype(int)


       # === Duration features ===


       # Scheduled flight duration in minutes
       duration = (df['scheduled_arrival'] - df['scheduled_departure']).dt.total_seconds() / 60
       # Handle negative durations (overnight flights) by adding 24 hours
       duration = duration.where(duration > 0, duration + 1440)
       df['scheduled_duration_min'] = duration


       # Categorize flight by duration
       df['is_short_flight'] = (df['scheduled_duration_min'] < 120).astype(int)  # < 2 hours
       df['is_long_flight'] = (df['scheduled_duration_min'] > 300).astype(int)   # > 5 hours


       # === Categorical interaction features ===


       # These will be encoded later, but create the interaction columns
       if 'airline' in df.columns and 'origin_airport' in df.columns:
           df['airline_origin'] = df['airline'].astype(str) + '_' + df['origin_airport'].astype(str)


       if 'airline' in df.columns and 'destination_airport' in df.columns:
           df['airline_destination'] = df['airline'].astype(str) + '_' + df['destination_airport'].astype(str)


       # Route feature (origin -> destination)
       if 'origin_airport' in df.columns and 'destination_airport' in df.columns:
           df['route'] = df['origin_airport'].astype(str) + '_' + df['destination_airport'].astype(str)


       return df


   def fit_transform(self, df):
       return self.fit(df).transform(df)