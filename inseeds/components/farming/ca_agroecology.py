"""Agroecological clustering for cross-border social learning.

This module groups countries by climate similarity, enabling farmers to
learn from successful practices in climatically similar regions worldwide.

Overview
--------
Why cluster countries? A farmer in Spain might benefit from learning about
practices that work well in Morocco or California - regions with similar
Mediterranean climates. This module creates these "agroecological clusters".

Clustering Method
-----------------
Countries are clustered using k-means on 6 climate features:

    +-------------------+------------------------------------------+
    | Feature           | Description                              |
    +===================+==========================================+
    | temp_mean         | Annual mean temperature (°C)             |
    | temp_amplitude    | Seasonal temperature range (max - min)   |
    | prec_mean         | Annual mean precipitation (mm)           |
    | prec_amplitude    | Seasonal precipitation range             |
    | pet_mean          | Annual mean potential evapotranspiration |
    | pet_amplitude     | Seasonal PET range                       |
    +-------------------+------------------------------------------+

The number of clusters (k) is determined automatically using the elbow method,
or can be specified manually.

Data Flow
---------
    1. init_agroecological_clusters()
       - Runs at model initialization
       - Clusters countries and stores mappings in world.statistic

    2. cluster_management_performance()
       - Runs after all country updates each year
       - Aggregates country stores into cluster-level stores

Functions
---------
cluster_countries_by_agroecology
    Perform k-means clustering on climate features.
init_agroecological_clusters
    Initialize clusters and assign each country to a cluster.
cluster_management_performance
    Aggregate country statistics into cluster-level statistics.

See Also
--------
ca_country : Country-level aggregation (input to clusters)
ca_behaviour : How farmers use cluster statistics for social learning
ca_management : RegionManagementPerformanceStore definition
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def cluster_countries_by_agroecology(world, countries, n_clusters=None, k_range=(5, 10)):
    """Cluster countries by agroecological similarity: mean + seasonal amplitude.

    Uses 6 features per country:
    - temp_mean, temp_amplitude (max_month - min_month)
    - prec_mean, prec_amplitude
    - pet_mean, pet_amplitude

    Parameters
    ----------
    world : World
        World with LPJmL output data containing monthly variables.
    countries : list
        List of Country objects with indices attribute.
    n_clusters : int, optional
        Number of clusters. If None, use elbow method.
    k_range : tuple
        Range for elbow search when n_clusters is None.

    Returns
    -------
    dict
        {cluster_id: [country_code, ...]} mapping clusters to countries.
    dict
        {country_code: cluster_id} mapping countries to clusters.
    int
        Number of clusters used.

    Notes
    -----
    Countries with NaN values are excluded from clustering
    and will have cluster_id = -1.
    """
    output = world.from_earth

    country_features = []
    country_codes = []
    skipped_countries = []

    for country in countries:
        indices = country.indices

        if len(indices) == 0:
            skipped_countries.append(country.country_code)
            continue

        try:
            temp = output.temp.isel(cell=indices)
            prec = output.prec.isel(cell=indices)
            pet = output.pet.isel(cell=indices)
        except (KeyError, AttributeError):
            skipped_countries.append(country.country_code)
            continue

        temp_mean = float(temp.mean())
        prec_mean = float(prec.mean())
        pet_mean = float(pet.mean())

        if "band" in temp.dims and "band" in prec.dims and "band" in pet.dims:
            temp_monthly = temp.mean(dim=[d for d in temp.dims if d not in ["band"]])
            prec_monthly = prec.mean(dim=[d for d in prec.dims if d not in ["band"]])
            pet_monthly = pet.mean(dim=[d for d in pet.dims if d not in ["band"]])
        else:
            raise ValueError("temp, prec, and pet must have band (monthly) dimension")

        temp_amplitude = float(temp_monthly.max() - temp_monthly.min())
        prec_amplitude = float(prec_monthly.max() - prec_monthly.min())
        pet_amplitude = float(pet_monthly.max() - pet_monthly.min())

        features = [
            temp_mean,
            temp_amplitude,
            prec_mean,
            prec_amplitude,
            pet_mean,
            pet_amplitude,
        ]

        if not any(np.isnan(features)):
            country_features.append(features)
            country_codes.append(country.country_code)
        else:
            raise ValueError(
                f"Either temp, prec, or pet are invalid for country {country.country_code}"
            )

    if len(country_features) < 2:
        cluster_to_countries = {0: country_codes}
        country_to_cluster = {code: 0 for code in country_codes}
        for code in skipped_countries:
            country_to_cluster[code] = -1
        return cluster_to_countries, country_to_cluster, 1

    features_array = np.array(country_features)

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_array)

    if n_clusters is None:
        max_k = min(k_range[1], len(country_codes) - 1)
        min_k = min(k_range[0], max_k)
        if min_k < 2:
            min_k = 2
        if max_k < min_k:
            max_k = min_k
        n_clusters, _ = _find_optimal_k(features_scaled, (min_k, max_k))

    n_clusters = min(n_clusters, len(country_codes))

    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = kmeans.fit_predict(features_scaled)

    cluster_to_countries = {i: [] for i in range(n_clusters)}
    country_to_cluster = {}

    for code, label in zip(country_codes, labels):
        cluster_to_countries[int(label)].append(code)
        country_to_cluster[code] = int(label)

    for code in skipped_countries:
        country_to_cluster[code] = -1

    return cluster_to_countries, country_to_cluster, n_clusters


def _find_optimal_k(features, k_range=(3, 12)):
    """Find optimal number of clusters using elbow method.

    Uses the second derivative of inertia to find the "elbow" point
    where adding more clusters yields diminishing returns.

    Parameters
    ----------
    features : np.ndarray
        Scaled feature matrix (n_samples, n_features).
    k_range : tuple
        (min_k, max_k) range to search.

    Returns
    -------
    int
        Optimal number of clusters.
    list
        Inertia values for each k in range.
    """
    inertias = []
    k_values = list(range(k_range[0], k_range[1] + 1))

    for k in k_values:
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        kmeans.fit(features)
        inertias.append(kmeans.inertia_)

    if len(inertias) < 3:
        return k_range[0], inertias

    diffs = np.diff(inertias)
    diffs2 = np.diff(diffs)
    elbow_idx = np.argmax(diffs2) + 1
    optimal_k = k_values[elbow_idx]

    return optimal_k, inertias


def init_agroecological_clusters(world, countries, n_clusters=None, k_range=(5, 10)):
    """Initialize country-level agroecological clusters for tele-coupled social learning.

    Clusters countries by agroecological similarity (mean + seasonal amplitude of
    temperature, precipitation, and PET). Countries in the same cluster can
    share social learning information.

    Parameters
    ----------
    world : World
        The world object with statistic attribute.
    countries : list
        List of Country objects to cluster.
    n_clusters : int, optional
        Number of clusters. If None, automatically determined via elbow method.
    k_range : tuple
        (min_k, max_k) range for elbow search when n_clusters is None.

    Notes
    -----
    Stores results in world.statistic:
    - "cluster_to_countries": {cluster_id: [country_code, ...]}
    - "country_to_cluster": {country_code: cluster_id}
    - "n_agroecological_clusters": int

    Also sets `agroecological_cluster` attribute on each country for easy access.
    The cluster value is written to output via Country.output_variables.
    """
    if not countries:
        return

    cluster_to_countries, country_to_cluster, k = cluster_countries_by_agroecology(
        world, countries, n_clusters, k_range
    )

    world.statistic.set("cluster_to_countries", cluster_to_countries)
    world.statistic.set("country_to_cluster", country_to_cluster)
    world.statistic.set("n_agroecological_clusters", k)

    for country in countries:
        country.agroecological_cluster = country_to_cluster.get(
            country.country_code, -1
        )


def cluster_management_performance(world):
    """Aggregate country-level stats into agroecological cluster stats.

    Collects bundle performance and adoption data from all countries,
    groups by agroecological cluster, and stores aggregated stats in
    world.statistic for cross-border social learning.

    Call this AFTER all country updates complete each year.

    Parameters
    ----------
    world : World
        The world object with countries and statistic.

    Notes
    -----
    Stores in world.statistic:
    - "cluster_management_performance": {cluster_id: RegionManagementPerformanceStore}
    """
    from inseeds.components.farming.ca_management import (
        RegionManagementPerformanceStore,
    )

    cluster_to_countries = world.statistic.get("cluster_to_countries", {})

    if not cluster_to_countries:
        return

    cluster_stats = {}
    countries_by_code = {c.country_code: c for c in world.countries}

    for cluster_id, country_codes in cluster_to_countries.items():
        store = RegionManagementPerformanceStore(country_codes=list(country_codes))

        for code in country_codes:
            country = countries_by_code.get(code)
            if country is None:
                logger.warning(f"Country {code} not found in world.countries")
                continue

            country_store = country.statistic.get("management_performance")
            if not isinstance(country_store, RegionManagementPerformanceStore):
                logger.warning(
                    f"Country {code} has no management performance store, skipping"
                )
                continue

            if store.year < 0:
                store.year = country_store.year
            store.merge_store(country_store)

        store.finalize_store()
        cluster_stats[cluster_id] = store

    world.statistic.set("cluster_management_performance", cluster_stats)


def save_cluster_map_netcdf(world, output_path, countries=None):
    """Save agroecological cluster assignments as a NetCDF file.

    Creates a gridded map where each cell's value is the cluster ID
    of the country it belongs to. Cells not in any country get -1.

    Parameters
    ----------
    world : World
        The world object with grid, countries, and cluster assignments.
    output_path : str or Path
        Path to save the NetCDF file (e.g., "cluster_map.nc").
    countries : list, optional
        List of Country objects. If None, uses world.countries.

    Notes
    -----
    Output variables:
    - cluster: Agroecological cluster ID per cell (-1 for unassigned)
    - country: Country code per cell (for reference)

    Coordinates use the LPJmL grid (lon, lat).
    """
    import xarray as xr
    from pathlib import Path

    if countries is None:
        countries = list(world.countries)

    # Get grid from world
    grid = getattr(world, "grid", None) or getattr(world, "_grid", None)
    if grid is None:
        logger.warning("No grid found in world, cannot save cluster map")
        return

    # Get lon/lat coordinates
    if hasattr(grid, "lon") and hasattr(grid, "lat"):
        lon = grid.lon.values if hasattr(grid.lon, "values") else grid.lon
        lat = grid.lat.values if hasattr(grid.lat, "values") else grid.lat
    else:
        logger.warning("Grid has no lon/lat, cannot save cluster map")
        return

    ncell = len(lon)

    # Initialize arrays with -1 (unassigned)
    cluster_arr = np.full(ncell, -1, dtype=np.int32)
    country_arr = np.full(ncell, "", dtype=object)

    # Get cluster mapping
    country_to_cluster = world.statistic.get("country_to_cluster", {})

    # Fill arrays based on country assignments
    for country in countries:
        indices = country.indices
        code = country.country_code
        cluster_id = country_to_cluster.get(code, -1)

        for idx in indices:
            if 0 <= idx < ncell:
                cluster_arr[idx] = cluster_id
                country_arr[idx] = code

    # Create xarray Dataset
    ds = xr.Dataset(
        {
            "cluster": (["cell"], cluster_arr),
            "country": (["cell"], country_arr),
        },
        coords={
            "lon": (["cell"], lon),
            "lat": (["cell"], lat),
        },
        attrs={
            "title": "Agroecological Cluster Map",
            "description": "Cluster IDs assigned to each grid cell based on country agroecological similarity",
            "n_clusters": world.statistic.get("n_agroecological_clusters", -1),
        },
    )

    # Add variable attributes
    ds["cluster"].attrs = {
        "long_name": "Agroecological cluster ID",
        "units": "1",
        "_FillValue": -1,
    }
    ds["country"].attrs = {
        "long_name": "Country code (ISO 3166-1 alpha-3)",
    }

    # Save to NetCDF
    output_path = Path(output_path)
    ds.to_netcdf(output_path)
    logger.info(f"Saved agroecological cluster map to {output_path}")
