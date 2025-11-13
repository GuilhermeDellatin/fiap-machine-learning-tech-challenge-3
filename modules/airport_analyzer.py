from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature


class AirportAnalyzer:
    """
    Analyze and visualize airport and flight data.
    """

    def __init__(self, df_flights: pd.DataFrame, df_airports: pd.DataFrame):
        """
        Initialize the AirportAnalyzer with flight and airport datasets.
        """
        self.df_flights = df_flights.copy()
        self.df_airports = df_airports.copy()
        self.df_metrics = None
        self.df_routes = None

    # ------------------------------------------------------------------
    # ROUTES ANALYSIS
    # ------------------------------------------------------------------
    def compute_top_routes(self, top_n: int = 50):
        """
        Compute the top N busiest routes between airports.
        Returns a DataFrame with coordinates of origin and destination airports.
        """
        df_counts = (
            self.df_flights
            .value_counts(['origin_airport', 'destination_airport'])
            .reset_index(name='num_flights')
            .sort_values('num_flights', ascending=False)
        )

        df_top = df_counts.head(top_n)

        # Merge origin and destination coordinates
        df_origin = df_top.merge(
            self.df_airports,
            left_on='origin_airport', right_on='iata_code', how='left'
        ).rename(columns={'latitude': 'origin_lat', 'longitude': 'origin_lon'})

        df_full = df_origin.merge(
            self.df_airports,
            left_on='destination_airport', right_on='iata_code', how='left',
            suffixes=('_origin', '_dest')
        ).rename(columns={'latitude': 'dest_lat', 'longitude': 'dest_lon'})

        self.df_routes = df_full
        return df_full

    # ------------------------------------------------------------------
    # ROUTE PLOT
    # ------------------------------------------------------------------
    def plot_top_routes(self, top_n: int = 50):
        """
        Plot the top N busiest flight routes.
        """
        if self.df_routes is None:
            self.compute_top_routes(top_n)

        df_full = self.df_routes
        df_airports = self.df_airports

        plt.figure(figsize=(12, 8))
        ax = plt.axes(projection=ccrs.LambertConformal())
        ax.set_extent([-170, -50, 10, 70], crs=ccrs.PlateCarree())

        # Add base map
        ax.add_feature(cfeature.LAND, facecolor='lightgray')
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.add_feature(cfeature.STATES, linewidth=0.5)

        # Plot routes
        for _, row in df_full.iterrows():
            plt.plot(
                [row['origin_lon'], row['dest_lon']],
                [row['origin_lat'], row['dest_lat']],
                color='red',
                linewidth=0.5 + 3 * (row['num_flights'] / df_full['num_flights'].max()),
                alpha=0.5,
                transform=ccrs.PlateCarree()
            )

        # Plot airports
        plt.scatter(
            df_airports['longitude'],
            df_airports['latitude'],
            s=5, color='blue', transform=ccrs.PlateCarree(), label='Airports'
        )

        plt.title(f'Top {top_n} Busiest Flight Routes', fontsize=14)
        plt.legend(loc='lower left')
        plt.show()

    # ------------------------------------------------------------------
    # AIRPORT METRICS
    # ------------------------------------------------------------------
    def compute_airport_metrics(self):
        """
        Compute traffic (departures, arrivals), total flights, and delays per airport.
        Returns a merged DataFrame with all metrics and airport coordinates.
        """
        # Total flights per airport
        flights_origin = self.df_flights['origin_airport'].value_counts().rename('departures')
        flights_dest = self.df_flights['destination_airport'].value_counts().rename('arrivals')

        df_traffic = pd.concat([flights_origin, flights_dest], axis=1).fillna(0)
        df_traffic['total_flights'] = df_traffic['departures'] + df_traffic['arrivals']
        df_traffic = df_traffic.reset_index().rename(columns={'index': 'iata_code'})

        # Average delays
        arrival_delay = (
            self.df_flights.groupby('destination_airport')['arrival_delay']
            .mean().reset_index()
            .rename(columns={'destination_airport': 'iata_code', 'arrival_delay': 'avg_arrival_delay'})
        )

        departure_delay = (
            self.df_flights.groupby('origin_airport')['departure_delay']
            .mean().reset_index()
            .rename(columns={'origin_airport': 'iata_code', 'departure_delay': 'avg_departure_delay'})
        )

        # Merge with airport coordinates
        df_metrics = (
            self.df_airports.merge(df_traffic, on='iata_code', how='left')
            .merge(arrival_delay, on='iata_code', how='left')
            .merge(departure_delay, on='iata_code', how='left')
        )

        df_metrics = df_metrics.sort_values('total_flights', ascending=False).reset_index(drop=True)
        self.df_metrics = df_metrics
        return df_metrics

    # ------------------------------------------------------------------
    # AIRPORT PLOTS
    # ------------------------------------------------------------------
    def plot_airports(self, value_col: str, title: str, cmap: str = 'coolwarm'):
        """
        Plot airports on a North America map, color-encoded by the given metric column.
        Example value_col: 'total_flights', 'avg_arrival_delay', 'avg_departure_delay'
        """
        if self.df_metrics is None:
            self.compute_airport_metrics()

        df_plot = self.df_metrics.dropna(subset=[value_col, 'latitude', 'longitude'])

        plt.figure(figsize=(10, 7))
        ax = plt.axes(projection=ccrs.LambertConformal())
        ax.set_extent([-170, -50, 10, 70], crs=ccrs.PlateCarree())

        # Base map
        ax.add_feature(cfeature.LAND, facecolor='lightgray')
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.add_feature(cfeature.STATES, linewidth=0.4)

        # Normalize point sizes for visibility
        size = (
            (df_plot[value_col] - df_plot[value_col].min()) /
            (df_plot[value_col].max() - df_plot[value_col].min()) * 200 + 20
        )

        scatter = plt.scatter(
            df_plot['longitude'],
            df_plot['latitude'],
            s=size,
            c=df_plot[value_col],
            cmap=cmap,
            alpha=0.7,
            transform=ccrs.PlateCarree()
        )

        plt.title(title, fontsize=14)
        cbar = plt.colorbar(scatter, orientation='vertical', shrink=0.6, pad=0.05)
        cbar.set_label(value_col.replace('_', ' ').title())
        plt.show()
       
    # ------------------------------------------------------------------
    # TOP AIRPORTS
    # ------------------------------------------------------------------
    def get_top_airports(self, metric: str = 'total_flights', top_n: int = 10):
        """
        Return the top airports by a specific metric.
        metric options: 'total_flights', 'departures', 'arrivals',
                        'avg_arrival_delay', 'avg_departure_delay'
        """
        if self.df_metrics is None:
            self.compute_airport_metrics()

        if metric not in self.df_metrics.columns:
            raise ValueError(f"Invalid metric '{metric}'. Available: {list(self.df_metrics.columns)}")

        df_sorted = self.df_metrics.sort_values(metric, ascending=False).head(top_n)
        return df_sorted[['iata_code', 'airport', 'city', 'state', metric]].reset_index(drop=True)

