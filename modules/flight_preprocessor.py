from datetime import datetime, timedelta
import pandas as pd
import numpy as np

class FlightPreprocessor:
    def __init__(self, base_date=None):
        # base_date used only as a fallback when rows don't contain year/month/day
        self.base_date = base_date or datetime(2024, 1, 1)
        self.date_summary = {}

    def _hhmm_to_hhmmstr(self, hhmm):
        """Convert numeric HHMM (or string) to 'HH:MM' string. Return None for nulls."""
        if pd.isna(hhmm):
            return None
        try:
            hhmm = int(hhmm)
        except Exception:
            # if it's already like 'HH:MM' accept it
            if isinstance(hhmm, str) and ":" in hhmm:
                return hhmm
            return None
        hour, minute = divmod(hhmm, 100)
        return f"{hour:02d}:{minute:02d}"

    def _make_date_series(self, df):
        """
        Returns a pd.Series of datetimes (date only) using df['year','month','day'] if present,
        otherwise a constant date series from self.base_date.
        """
        if {'year', 'month', 'day'}.issubset(df.columns):
            # coerce to numeric then to datetime; invalid rows become NaT
            try:
                # ensure ints (but keep NaNs)
                y = pd.to_numeric(df['year'], errors='coerce').astype('Int64')
                m = pd.to_numeric(df['month'], errors='coerce').astype('Int64')
                d = pd.to_numeric(df['day'], errors='coerce').astype('Int64')
                date_df = pd.DataFrame({'year': y, 'month': m, 'day': d})
                date_series = pd.to_datetime(date_df[['year','month','day']], errors='coerce')
                # Where any of y/m/d missing, date_series will be NaT.
                # Fill those with base_date
                fallback = pd.Timestamp(self.base_date.date())
                date_series = date_series.fillna(fallback)
                return date_series
            except Exception:
                # safe fallback
                return pd.Series(pd.Timestamp(self.base_date.date()), index=df.index)
        else:
            return pd.Series(pd.Timestamp(self.base_date.date()), index=df.index)

    def _combine_date_and_time(self, date_series, time_str_series):
        """
        Combine a date Series (pd.Timestamp date-only) and a time string Series ("HH:MM" or None)
        into a datetime Series. Handles "24:00" by rolling to next day.
        """
        # initialize result as NaT
        result = pd.Series(pd.NaT, index=time_str_series.index, dtype='datetime64[ns]')

        # mask of non-null time strings
        mask = time_str_series.notna()
        if not mask.any():
            return result

        times = time_str_series[mask].astype(str)

        # split into hour/minute safely
        parts = times.str.split(':', expand=True)
        hours = pd.to_numeric(parts[0], errors='coerce').fillna(0).astype(int)
        minutes = pd.to_numeric(parts[1], errors='coerce').fillna(0).astype(int)

        # base date for those rows
        base_dates = date_series[mask].dt.normalize()

        # create timedeltas and add; if hours == 24 we'll naturally advance to next day
        td = pd.to_timedelta(hours, unit='h') + pd.to_timedelta(minutes, unit='m')
        combined = base_dates + td

        result.loc[mask] = combined
        return result

    def preprocess(self, df):
        df = df.copy()

        time_cols = [
            'scheduled_departure', 'departure_time',
            'scheduled_arrival', 'arrival_time'
        ]

        # Step 1: convert raw HHMM-like values to "HH:MM" strings (or None)
        for col in time_cols:
            if col in df.columns:
                df[col + '_hhmmstr'] = df[col].apply(self._hhmm_to_hhmmstr)
            else:
                df[col + '_hhmmstr'] = None

        # Step 2: build a date series to use for combination (per-row if possible)
        date_series = self._make_date_series(df)

        # Step 3: combine date + time strings into datetimes
        for col in time_cols:
            hhmm_col = col + '_hhmmstr'
            if hhmm_col in df.columns:
                df[col] = self._combine_date_and_time(date_series, df[hhmm_col])
            else:
                df[col] = pd.NaT

            # drop helper column
            if hhmm_col in df.columns:
                df.drop(columns=[hhmm_col], inplace=True)

        # ==========================================================
        # === DEPARTURE corrections (multi-day rollover detection) ===
        # ==========================================================
        if 'departure_delay' in df.columns and df['departure_time'].notna().any() and df['scheduled_departure'].notna().any():
            calc_delay = (df['departure_time'] - df['scheduled_departure']).dt.total_seconds() / 60
            diff = calc_delay - df['departure_delay']

            too_far_ahead_1d = (diff > 1000) & (diff < 2000) & (df['departure_delay'] < 0)
            too_far_ahead_2d = (diff >= 2000) & (df['departure_delay'] < 0)
            too_far_back_1d = (diff < -1000) & (diff > -2000) & (df['departure_delay'] > 0)
            too_far_back_2d = (diff <= -2000) & (df['departure_delay'] > 0)

            df.loc[too_far_ahead_1d, ['departure_time', 'arrival_time', 'scheduled_arrival']] -= timedelta(days=1)
            df.loc[too_far_ahead_2d, ['departure_time', 'arrival_time', 'scheduled_arrival']] -= timedelta(days=2)
            df.loc[too_far_back_1d, ['departure_time', 'arrival_time', 'scheduled_arrival']] += timedelta(days=1)
            df.loc[too_far_back_2d, ['departure_time', 'arrival_time', 'scheduled_arrival']] += timedelta(days=2)

        # ======================================================
        # === ARRIVAL corrections (same logic mirrored) ===
        # ======================================================
        if 'arrival_delay' in df.columns and df['arrival_time'].notna().any() and df['scheduled_arrival'].notna().any():
            calc_arr_delay = (df['arrival_time'] - df['scheduled_arrival']).dt.total_seconds() / 60
            arr_diff = calc_arr_delay - df['arrival_delay']

            arr_ahead_1d = (arr_diff > 1000) & (arr_diff < 2000) & (df['arrival_delay'] < 0)
            arr_ahead_2d = (arr_diff >= 2000) & (df['arrival_delay'] < 0)
            arr_back_1d = (arr_diff < -1000) & (arr_diff > -2000) & (df['arrival_delay'] > 0)
            arr_back_2d = (arr_diff <= -2000) & (df['arrival_delay'] > 0)

            df.loc[arr_ahead_1d, ['arrival_time']] -= timedelta(days=1)
            df.loc[arr_ahead_2d, ['arrival_time']] -= timedelta(days=2)
            df.loc[arr_back_1d, ['arrival_time']] += timedelta(days=1)
            df.loc[arr_back_2d, ['arrival_time']] += timedelta(days=2)

        # ---- Fix arrivals that are still before departures ----
        # Only apply when both datetime columns exist
        if 'arrival_time' in df.columns and 'departure_time' in df.columns:
            arr_mask = (df['arrival_time'] < df['departure_time']) & df['arrival_time'].notna() & df['departure_time'].notna()
            df.loc[arr_mask, ['scheduled_arrival', 'arrival_time']] += timedelta(days=1)

        # ---- Recompute derived fields ----
        if 'departure_time' in df.columns and 'scheduled_departure' in df.columns:
            df['dep_delay_min'] = (df['departure_time'] - df['scheduled_departure']).dt.total_seconds() / 60
        else:
            df['dep_delay_min'] = np.nan

        if 'arrival_time' in df.columns and 'scheduled_arrival' in df.columns:
            df['arr_delay_min'] = (df['arrival_time'] - df['scheduled_arrival']).dt.total_seconds() / 60
        else:
            df['arr_delay_min'] = np.nan

        if 'is_delayed' not in df.columns:
            # conservatively use dep_delay_min when available
            df['is_delayed'] = df['dep_delay_min'] > 15

        # ======================================================
        # === Ensure calendar components exist and are correct ===
        # ======================================================
        # Prefer existing year/month/day columns if they exist; otherwise derive from scheduled_departure
        if {'year','month','day'}.issubset(df.columns):
            # keep as-is but ensure integer types and no bogus values
            df['year'] = pd.to_numeric(df['year'], errors='coerce').astype('Int64')
            df['month'] = pd.to_numeric(df['month'], errors='coerce').astype('Int64')
            df['day'] = pd.to_numeric(df['day'], errors='coerce').astype('Int64')
            # create a canonical scheduled_departure if it was missing earlier
            if 'scheduled_departure' not in df.columns or df['scheduled_departure'].isna().all():
                df['scheduled_departure'] = pd.to_datetime(df[['year','month','day']], errors='coerce')
        else:
            # derive from scheduled_departure
            if 'scheduled_departure' in df.columns:
                df['year'] = df['scheduled_departure'].dt.year
                df['month'] = df['scheduled_departure'].dt.month
                df['day'] = df['scheduled_departure'].dt.day

        # day_of_week always derived from scheduled_departure if possible
        if 'scheduled_departure' in df.columns:
            df['day_of_week'] = df['scheduled_departure'].dt.dayofweek  # Monday=0

        # ======================================================
        # === Store distinct values summary ===
        # ======================================================
        cols = ['year', 'month', 'day', 'day_of_week']
        self.date_summary = {}
        for col in cols:
            if col in df.columns:
                vals = df[col].dropna().unique().tolist()
                try:
                    vals = sorted(vals)
                except Exception:
                    pass
                self.date_summary[col] = vals

        return df
