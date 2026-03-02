"""Dummy FAO data generator for development and testing.

This module provides realistic dummy data when the FAO API is unavailable.
Values are based on typical ranges from FAOSTAT data to ensure the model
behaves reasonably even without real data.

Typical value ranges (from FAOSTAT):
- Producer prices: 100-1000 USD/tonne (varies by crop)
- Depreciation rate: 0.03-0.08 (3-8% per year)
- Investment rate: 0.05-0.15 (5-15% per year)
- Net Capital Stocks: varies widely by country

References:
- FAOSTAT Producer Prices domain (PP)
- FAOSTAT Capital Stock domain (CS)
- OECD (2009). Measuring Capital - OECD Manual.
"""

from pathlib import Path
from typing import Literal

import numpy as np
import xarray as xr


# LPJmL crop functional types
LPJML_CFTS = [
    "temperate cereals", "rice", "maize", "tropical cereals",
    "pulses", "temperate roots", "tropical roots", "oil crops sunflower",
    "oil crops soybean", "oil crops groundnut", "oil crops rapeseed", "sugarcane",
    "others", "Others, annual", "Others, perennial", "grassland",
    "biomass_grass", "biomass_tree"
]

# Typical producer prices by crop type (USD/tonne dry matter)
# Based on FAOSTAT PP domain averages, adjusted for dry matter
TYPICAL_PRICES = {
    "temperate cereals": 250,    # Wheat, barley
    "rice": 400,                 # Paddy rice
    "maize": 200,                # Maize
    "tropical cereals": 300,     # Sorghum, millet
    "pulses": 600,               # Beans, lentils
    "temperate roots": 200,      # Potatoes (DM adjusted)
    "tropical roots": 150,       # Cassava (DM adjusted)
    "oil crops sunflower": 450,  # Sunflower seeds
    "oil crops soybean": 400,    # Soybeans
    "oil crops groundnut": 800,  # Groundnuts
    "oil crops rapeseed": 450,   # Rapeseed
    "sugarcane": 50,             # Sugarcane (low per tonne)
    "others": 300,               # Mixed
    "Others, annual": 300,
    "Others, perennial": 400,
    "grassland": 100,            # Hay/fodder
    "biomass_grass": 80,
    "biomass_tree": 60,
}


def generate_dummy_producer_prices(
    years: tuple[int, int] = (1990, 2020),
    countries: list[str] | None = None,
    output_path: str | Path | None = None,
    seed: int = 42,
) -> xr.Dataset:
    """Generate dummy producer prices dataset.

    Creates realistic price data with:
    - Crop-specific base prices from TYPICAL_PRICES
    - Country-specific random variation (±30%)
    - Year-to-year random variation (±10%)
    - Slight upward trend over time

    Parameters
    ----------
    years : tuple[int, int]
        Year range (start, end) inclusive.
    countries : list[str] | None
        List of ISO3 country codes. If None, uses common countries.
    output_path : str | Path | None
        If provided, saves dataset to this path.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    xr.Dataset
        Dataset with dimensions (time, area_code, npft) and variable "5532"
        (producer price in USD/tonne dry matter).
    """
    rng = np.random.default_rng(seed)

    if countries is None:
        countries = [
            "USA", "DEU", "FRA", "BRA", "IND", "CHN", "ARG", "AUS",
            "CAN", "MEX", "ESP", "ITA", "GBR", "POL", "NLD", "ZAF",
        ]

    year_list = list(range(years[0], years[1] + 1))
    n_years = len(year_list)
    n_countries = len(countries)
    n_crops = len(LPJML_CFTS)

    # Base prices for each crop
    base_prices = np.array([TYPICAL_PRICES[cft] for cft in LPJML_CFTS])

    # Create price array with shape (time, area_code, npft)
    prices = np.zeros((n_years, n_countries, n_crops))

    for i, country in enumerate(countries):
        # Country-specific factor (±30% variation)
        country_factor = 1.0 + rng.uniform(-0.3, 0.3)

        for j, year in enumerate(year_list):
            # Year-specific factor (±10% variation + slight trend)
            year_factor = 1.0 + rng.uniform(-0.1, 0.1)
            trend = 1.0 + 0.01 * (year - years[0])  # 1% per year increase

            # Apply factors
            prices[j, i, :] = base_prices * country_factor * year_factor * trend

    # Create xarray Dataset
    ds = xr.Dataset(
        {
            "5532": (["time", "area_code", "npft"], prices),
        },
        coords={
            "time": year_list,
            "area_code": countries,
            "npft": LPJML_CFTS,
        },
    )

    # Add metadata
    ds["5532"].attrs = {
        "units": "USD/tonne_dry_matter",
        "long_name": "Producer price per tonne dry matter (DUMMY DATA)",
        "source": "Generated dummy data - NOT real FAOSTAT",
        "warning": "This is synthetic data for testing only",
    }
    ds.time.attrs = {"units": "year", "long_name": "Year"}
    ds.area_code.attrs = {"code_standard": "ISO3"}

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        ds.to_netcdf(output_path)
        print(f"Saved dummy producer prices to {output_path}")

    return ds


def generate_dummy_capital_stock(
    years: tuple[int, int] = (1990, 2020),
    countries: list[str] | None = None,
    output_path: str | Path | None = None,
    seed: int = 42,
) -> xr.Dataset:
    """Generate dummy capital stock dataset.

    Creates realistic capital stock data with:
    - Depreciation rate: 3-8% (typical for agricultural capital)
    - Investment rate: 5-15% (typical GFCF/NCS ratio)
    - Net Capital Stocks: scaled by country size proxy
    - GFCF and CFC derived from rates

    Parameters
    ----------
    years : tuple[int, int]
        Year range (start, end) inclusive.
    countries : list[str] | None
        List of ISO3 country codes. If None, uses common countries.
    output_path : str | Path | None
        If provided, saves dataset to this path.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    xr.Dataset
        Dataset with dimensions (time, area_code) and variables:
        - 6184: Gross Fixed Capital Formation (million USD)
        - 6185: Consumption of Fixed Capital (million USD)
        - 6186: Net Capital Stocks (million USD)
        - depreciation_rate: CFC / NCS (1/year)
        - investment_rate: GFCF / NCS (1/year)
    """
    rng = np.random.default_rng(seed)

    if countries is None:
        countries = [
            "USA", "DEU", "FRA", "BRA", "IND", "CHN", "ARG", "AUS",
            "CAN", "MEX", "ESP", "ITA", "GBR", "POL", "NLD", "ZAF",
        ]

    year_list = list(range(years[0], years[1] + 1))
    n_years = len(year_list)
    n_countries = len(countries)

    # Country size proxies (relative agricultural sector size)
    # Larger values = larger agricultural capital stock
    country_sizes = {
        "USA": 500, "CHN": 800, "IND": 400, "BRA": 300, "ARG": 150,
        "AUS": 100, "CAN": 80, "MEX": 120, "FRA": 70, "DEU": 60,
        "ESP": 50, "ITA": 45, "GBR": 35, "POL": 40, "NLD": 25, "ZAF": 30,
    }

    # Initialize arrays
    ncs = np.zeros((n_years, n_countries))  # Net Capital Stocks
    gfcf = np.zeros((n_years, n_countries))  # Gross Fixed Capital Formation
    cfc = np.zeros((n_years, n_countries))  # Consumption of Fixed Capital
    dep_rate = np.zeros((n_years, n_countries))
    inv_rate = np.zeros((n_years, n_countries))

    for i, country in enumerate(countries):
        # Base capital stock (billion USD, converted to million later)
        base_ncs = country_sizes.get(country, 50) * 1000  # million USD

        # Country-specific rates
        base_dep_rate = rng.uniform(0.03, 0.08)  # 3-8% depreciation
        base_inv_rate = rng.uniform(0.05, 0.15)  # 5-15% investment

        for j, year in enumerate(year_list):
            # Slight growth over time
            growth = 1.0 + 0.02 * (year - years[0])  # 2% per year

            # Year-to-year variation
            year_var = 1.0 + rng.uniform(-0.05, 0.05)

            # Calculate values
            ncs[j, i] = base_ncs * growth * year_var
            dep_rate[j, i] = base_dep_rate * (1 + rng.uniform(-0.1, 0.1))
            inv_rate[j, i] = base_inv_rate * (1 + rng.uniform(-0.1, 0.1))

            # Derive GFCF and CFC from rates
            cfc[j, i] = dep_rate[j, i] * ncs[j, i]
            gfcf[j, i] = inv_rate[j, i] * ncs[j, i]

    # Create xarray Dataset
    ds = xr.Dataset(
        {
            "6184": (["time", "area_code"], gfcf),
            "6185": (["time", "area_code"], cfc),
            "6186": (["time", "area_code"], ncs),
            "depreciation_rate": (["time", "area_code"], dep_rate),
            "investment_rate": (["time", "area_code"], inv_rate),
        },
        coords={
            "time": year_list,
            "area_code": countries,
        },
    )

    # Add metadata
    ds["6184"].attrs = {
        "units": "million_USD",
        "long_name": "Gross Fixed Capital Formation (DUMMY DATA)",
        "source": "Generated dummy data - NOT real FAOSTAT",
    }
    ds["6185"].attrs = {
        "units": "million_USD",
        "long_name": "Consumption of Fixed Capital (DUMMY DATA)",
        "source": "Generated dummy data - NOT real FAOSTAT",
    }
    ds["6186"].attrs = {
        "units": "million_USD",
        "long_name": "Net Capital Stocks (DUMMY DATA)",
        "source": "Generated dummy data - NOT real FAOSTAT",
    }
    ds["depreciation_rate"].attrs = {
        "units": "1/year",
        "long_name": "Depreciation rate (DUMMY DATA)",
        "formula": "CFC / NCS",
    }
    ds["investment_rate"].attrs = {
        "units": "1/year",
        "long_name": "Investment rate (DUMMY DATA)",
        "formula": "GFCF / NCS",
    }
    ds.time.attrs = {"units": "year", "long_name": "Year"}
    ds.area_code.attrs = {"code_standard": "ISO3"}

    # Add global warning attribute
    ds.attrs["warning"] = "DUMMY DATA - Generated for testing, not real FAOSTAT data"

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        ds.to_netcdf(output_path)
        print(f"Saved dummy capital stock to {output_path}")

    return ds


def ensure_dummy_fao_data(
    sim_path: str | Path,
    data_type: Literal["producer_prices", "capital_stock", "both"] = "both",
    years: tuple[int, int] = (1990, 2020),
    countries: list[str] | None = None,
    force: bool = False,
) -> dict[str, Path]:
    """Ensure dummy FAO data files exist in simulation input folder.

    Creates dummy data files if they don't exist or if force=True.
    This is a convenience function for setting up test simulations.

    Note: Dummy files are named with '_DUMMY' suffix (e.g., fao_pft_prices_DUMMY.nc)
    to distinguish them from real FAO data.

    Parameters
    ----------
    sim_path : str | Path
        Simulation path (pycoupler sim_path).
    data_type : str
        Which data to generate: "producer_prices", "capital_stock", or "both".
    years : tuple[int, int]
        Year range for data generation.
    countries : list[str] | None
        Country codes to include. If None, uses defaults.
    force : bool
        If True, regenerate even if files exist.

    Returns
    -------
    dict[str, Path]
        Paths to the generated files.
    """
    sim_path = Path(sim_path)
    input_dir = sim_path / "input"
    input_dir.mkdir(parents=True, exist_ok=True)

    paths = {}

    if data_type in ("producer_prices", "both"):
        pp_path = input_dir / "fao_pft_prices_DUMMY.nc"
        if not pp_path.exists() or force:
            generate_dummy_producer_prices(
                years=years,
                countries=countries,
                output_path=pp_path,
            )
        paths["producer_prices"] = pp_path

    if data_type in ("capital_stock", "both"):
        cs_path = input_dir / "fao_capital_stock_DUMMY.nc"
        if not cs_path.exists() or force:
            generate_dummy_capital_stock(
                years=years,
                countries=countries,
                output_path=cs_path,
            )
        paths["capital_stock"] = cs_path

    return paths
