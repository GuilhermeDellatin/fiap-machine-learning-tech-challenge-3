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
        
    def plot_extreme_route_delays(self, top_n: int = 30):
        """
        Plot the routes with the most extreme average departure delays.
        """

        # Ensure metrics exist
        if not hasattr(self, "df_route_metrics") or self.df_route_metrics is None:
            self.compute_route_metrics()

        if self.df_metrics is None:
            self.compute_airport_metrics()

        # Work on a copy
        merged = self.df_route_metrics.copy()

        # Remove routes with missing coordinates (prevents posx/posy errors)
        merged = merged.dropna(subset=['origin_lat', 'origin_lon', 'dest_lat', 'dest_lon'])

        # Pick the top N worst delayed routes
        worst = merged.nlargest(top_n, 'avg_departure_delay').copy()

        # Map Setup
        fig = plt.figure(figsize=(14, 10))
        ax = plt.axes(projection=ccrs.LambertConformal())
        ax.set_extent([-170, -50, 10, 70], crs=ccrs.PlateCarree())

        # Map features
        ax.add_feature(cfeature.LAND, facecolor='lightgray')
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.add_feature(cfeature.STATES, linewidth=0.5)

        # Colors and sizes
        delays = worst['avg_departure_delay']
        norm = plt.Normalize(delays.min(), delays.max())
        cmap = plt.cm.Reds

        # Plot each delayed route
        for _, row in worst.iterrows():
            ax.plot(
                [row['origin_lon'], row['dest_lon']],
                [row['origin_lat'], row['dest_lat']],
                color=cmap(norm(row['avg_departure_delay'])),
                linewidth=1.0 + (row['avg_departure_delay'] / delays.max()) * 4,
                alpha=0.9,
                transform=ccrs.PlateCarree()
            )

        # Plot airports
        ax.scatter(
            worst['origin_lon'],
            worst['origin_lat'],
            s=25,
            color='blue',
            transform=ccrs.PlateCarree()
        )
        ax.scatter(
            worst['dest_lon'],
            worst['dest_lat'],
            s=25,
            color='blue',
            transform=ccrs.PlateCarree()
        )

        # Add labels for the worst 10 routes
        for _, row in worst.head(10).iterrows():
            ax.text(
                row['origin_lon'],
                row['origin_lat'],
                f"{row['origin_airport']}→{row['destination_airport']}",
                fontsize=8,
                transform=ccrs.PlateCarree(),
                weight='bold'
            )

        # Proper Cartopy colorbar — attach to the figure, not the axis
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        cbar_ax = fig.add_axes([0.92, 0.25, 0.02, 0.5])  # [left, bottom, width, height]
        cbar = plt.colorbar(sm, cax=cbar_ax)
        cbar.set_label("Route Avg Departure Delay (minutes)")

        plt.suptitle(f"Top {top_n} Routes With the Most Extreme Departure Delays", fontsize=16)
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
    # ROUTE METRICS
    # ------------------------------------------------------------------
    def compute_route_metrics(self, top_n: int = None):
        """
        Compute metrics for each route (origin → destination):
            - total number of flights
            - average arrival and departure delay
            - min/max delay
            - attach origin and destination coordinates

        If top_n is provided, returns only the top N busiest routes.
        """

        # Base route counts
        route_counts = (
            self.df_flights
            .value_counts(['origin_airport', 'destination_airport'])
            .reset_index(name='total_flights')
        )

        # Delay metrics per route
        route_delays = (
            self.df_flights
            .groupby(['origin_airport', 'destination_airport'])
            .agg(
                avg_departure_delay=('departure_delay', 'mean'),
                avg_arrival_delay=('arrival_delay', 'mean'),
                min_departure_delay=('departure_delay', 'min'),
                max_departure_delay=('departure_delay', 'max'),
                min_arrival_delay=('arrival_delay', 'min'),
                max_arrival_delay=('arrival_delay', 'max')
            )
            .reset_index()
        )

        # Merge counts + delay metrics
        df_routes = route_counts.merge(
            route_delays,
            on=['origin_airport', 'destination_airport'],
            how='left'
        )

        # -----------------------------
        # Add ORIGIN coordinates
        # -----------------------------
        df_routes = df_routes.merge(
            self.df_airports[['iata_code', 'latitude', 'longitude']],
            left_on='origin_airport',
            right_on='iata_code',
            how='left'
        ).rename(columns={
            'latitude': 'origin_lat',
            'longitude': 'origin_lon'
        }).drop(columns=['iata_code'])

        # -----------------------------
        # Add DESTINATION coordinates
        # -----------------------------
        df_routes = df_routes.merge(
            self.df_airports[['iata_code', 'latitude', 'longitude']],
            left_on='destination_airport',
            right_on='iata_code',
            how='left'
        ).rename(columns={
            'latitude': 'dest_lat',
            'longitude': 'dest_lon'
        }).drop(columns=['iata_code'])

        # Order by busiest
        df_routes = df_routes.sort_values('total_flights', ascending=False)

        # Optional filtering
        if top_n is not None:
            df_routes = df_routes.head(top_n)

        df_routes = df_routes.reset_index(drop=True)
        self.df_route_metrics = df_routes

        return df_routes
    
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

