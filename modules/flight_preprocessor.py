import pandas as pd
from datetime import datetime, timedelta

class FlightPreprocessor:
    def __init__(self, base_date=None):
        """
        base_date: optional datetime used to anchor times (default: 2024-01-01)
        """
        self.base_date = base_date or datetime(2024, 1, 1)

    def _hhmm_to_time(self, hhmm):
        """Convert 5 → '00:05', 2354 → '23:54'."""
        if pd.isna(hhmm):
            return None
        hhmm = int(hhmm)
        hour = hhmm // 100
        minute = hhmm % 100
        return f"{hour:02d}:{minute:02d}"

    def _to_datetime(self, time_str):
        """Convert 'HH:MM' to datetime using base_date; handle 24:00 as next day 00:00."""
        if pd.isna(time_str) or not isinstance(time_str, str):
            return None

        try:
            if time_str == "24:00":
                # Treat 24:00 as midnight of the next day
                dt = datetime.strptime("00:00", "%H:%M").replace(
                    year=self.base_date.year,
                    month=self.base_date.month,
                    day=self.base_date.day
                ) + timedelta(days=1)
            else:
                dt = datetime.strptime(time_str, "%H:%M").replace(
                    year=self.base_date.year,
                    month=self.base_date.month,
                    day=self.base_date.day
                )
            return dt
        except ValueError:
            print(f"invalid time format skipped: {time_str}")
            return None

    def preprocess(self, df):
        """
        Cleans and augments the flight schedule DataFrame.
        Expected columns:
            scheduled_departure, departure_time, scheduled_arrival, arrival_time
        """
        df = df.copy()
        time_cols = ['scheduled_departure', 'departure_time', 'scheduled_arrival', 'arrival_time']

        # Convert numeric HHMM → "HH:MM"
        for col in time_cols:
            df[col] = df[col].apply(self._hhmm_to_time)

        # Convert "HH:MM" → datetime
        for col in time_cols:
            df[col] = df[col].apply(self._to_datetime)

        # Adjust for next-day arrivals
        mask = df['arrival_time'] < df['departure_time']
        df.loc[mask, ['scheduled_arrival', 'arrival_time']] = \
            df.loc[mask, ['scheduled_arrival', 'arrival_time']] + timedelta(days=1)

        return df
