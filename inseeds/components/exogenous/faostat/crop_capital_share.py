"""Crop capital share scaling for FAO Capital Stock data.

This module provides country-level scaling factors to convert FAO Capital Stock
(CS domain, which covers "Agriculture, Forestry and Fishing" combined) to
crop-specific capital that matches LPJmL's field crop simulation.

Background:
-----------
FAO Capital Stock (CS domain) reports capital for the combined "Agriculture,
Forestry and Fishing" (AFF) sector. There is NO breakdown by subsector (crops
vs livestock vs forestry vs fishing) in the CS domain itself.

To estimate crop-specific capital, we use a two-step scaling approach:

1. Agriculture share of AFF (ag_share):
   - Removes forestry and fishing capital
   - Based on UN National Accounts data (ISIC Rev. 3/4 divisions 01, 02, 03)
   - Static table by country (forestry/fishing shares are relatively stable)

2. Crop share of agriculture (crop_share):
   - Removes livestock capital
   - Computed from FAO Gross Production Value (QV domain)
   - crop_share = GPV_crops / GPV_agriculture
   - Uses real FAO data (no static fallback)

Final scaling: crop_capital = NCS × ag_share × crop_share

Assumptions and Limitations:
---------------------------
- We assume capital share ≈ output share (common in agricultural economics)
- This is an approximation: capital intensity varies by subsector
  (e.g., greenhouses are more capital-intensive than field crops per $ output)
- The depreciation rate from FAO CS applies to combined AFF, but since
  agriculture dominates (85-95% in most countries), this is acceptable

Alternative approaches (not implemented):
- Physical inventory method using FAOSTAT asset data (tractors, land, etc.)
- Would be more accurate but requires multiple data sources and asset prices

Data Sources for AG_SHARE_OF_AFF:
---------------------------------
The agriculture share values are derived from:

1. UN National Accounts Official Country Data (primary source):
   - Table 2.3: Output, gross value added by industries (ISIC Rev. 3/4)
   - Provides separate data for:
     * Division 01: Agriculture, hunting and related service activities
     * Division 02: Forestry, logging and related service activities
     * Division 03: Fishing and aquaculture
   - URL: http://data.un.org/Data.aspx?d=SNA&f=group_code%3A203
   - Coverage: 186 countries, 1977-2024

2. Eurostat Economic Accounts for Agriculture (EU countries):
   - Agriculture, forestry and fishery statistics (2020 edition)
   - URL: https://ec.europa.eu/eurostat/documents/3217494/12069644/KS-FK-20-001-EN-N.pdf
   - Provides detailed EU member state breakdowns

3. FAO Statistical Yearbook (supplementary):
   - World Food and Agriculture Statistical Yearbook 2020
   - Forestry production statistics (FAOSTAT-Forestry)
   - Fisheries and aquaculture statistics

4. Country-specific sources:
   - CBS Netherlands: De landbouw in de Nederlandse economie (2020)
   - IBGE Brazil: Forestry Activities (PEVS 2020)
   - Statistics Norway: Agriculture, forestry, hunting and fishing
   - Nordic Forest Statistics (SNS, 2021)

Methodology:
------------
ag_share = VA_agriculture / VA_aff
         = VA_agriculture / (VA_agriculture + VA_forestry + VA_fishing)

Where VA = Gross Value Added from national accounts.

For countries without direct national accounts data, shares are estimated from:
- FAO forestry production value relative to agricultural output
- FAO fisheries production value relative to agricultural output
- Regional patterns and economic structure

References:
-----------
- FAO (2023). Value of Agricultural Production. FAOSTAT QV domain.
- FAO (2020). World Food and Agriculture - Statistical Yearbook.
- FAO (2020-2023). Agricultural investments and capital stock.
- UN Statistics Division (2024). National Accounts Official Country Data.
- Eurostat (2020). Agriculture, forestry and fishery statistics.
- World Bank (2023). Agriculture, forestry, and fishing value added.
- von Cramon-Taubadel et al. (2009). Assessing agricultural capital stocks.
- Nordic Forest Research (2021). Nordic Forest Statistics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import xarray as xr

# =============================================================================
# AGRICULTURE SHARE OF AG+FORESTRY+FISHING
# =============================================================================
# This table provides the fraction of "Agriculture, Forestry and Fishing" (AFF)
# value added that belongs to Agriculture (crops + livestock).
#
# FAO Capital Stock (CS domain) reports capital for the combined AFF sector.
# This factor isolates the agriculture component.
#
# Primary data source: UN National Accounts Official Country Data
# - Table 2.3: Gross value added by industries at current prices (ISIC Rev. 3/4)
# - Division 01: Agriculture, hunting and related service activities
# - Division 02: Forestry, logging and related service activities
# - Division 03: Fishing and aquaculture
# - URL: http://data.un.org/Data.aspx?d=SNA&f=group_code%3A203
#
# Supplementary sources:
# - Eurostat: Agriculture, forestry and fishery statistics (2020)
# - FAO: Forestry production statistics, Fisheries statistics
# - National statistical offices (CBS, IBGE, Statistics Norway, etc.)
#
# Methodology:
# ag_share = VA_agriculture / (VA_agriculture + VA_forestry + VA_fishing)
#
# Values represent 5-year averages (2015-2020) where available.
# For countries without direct data, shares are estimated from:
# - FAO forestry/fisheries production relative to agricultural output
# - Regional patterns and economic structure
# - Geographic factors (landlocked = minimal fishing)

DEFAULT_AG_SHARE_OF_AFF = 0.90  # Default for countries without specific data

AG_SHARE_OF_AFF = {
    # =========================================================================
    # EUROPE
    # Sources: Eurostat (2020), UN National Accounts
    # Nordic countries have significant forestry (30-40% of AFF in FIN/SWE)
    # =========================================================================
    # Netherlands: CBS (2020) - AFF = 1.8% GDP, forestry/fishing negligible
    # Agriculture output 51.6% crops, 38.2% animals; forestry <1% of AFF
    "NLD": 0.95,
    # Germany: UN SNA - forestry ~6% of AFF, fishing ~2%
    "DEU": 0.92,
    # France: UN SNA - forestry ~10% of AFF, fishing ~2%
    "FRA": 0.88,
    # UK: UN SNA - fishing significant (~10%), forestry ~5%
    "GBR": 0.85,
    # Italy: UN SNA - fishing ~8%, forestry ~2%
    "ITA": 0.90,
    # Spain: UN SNA - fishing ~12%, forestry ~3%
    "ESP": 0.85,
    # Poland: UN SNA - forestry ~8%, fishing ~2%
    "POL": 0.90,
    # Denmark: UN SNA - fishing ~15%, forestry ~5%
    "DNK": 0.80,
    # Belgium: UN SNA - forestry/fishing each ~4%
    "BEL": 0.92,
    # Austria: UN SNA - forestry ~13%, fishing negligible
    "AUT": 0.87,
    # Switzerland: UN SNA - forestry ~10%, fishing ~2%
    "CHE": 0.88,
    # Sweden: Nordic Forest Statistics (2021) - forestry ~28% of AFF
    # Sweden is 69% forested; forestry = 18% of global paper exports
    "SWE": 0.70,
    # Norway: Statistics Norway - fishing ~25%, forestry ~15%
    # Major fishing nation (top 10 globally) + significant forestry
    "NOR": 0.60,
    # Finland: Nordic Forest Statistics - forestry ~33% of AFF
    # 73% forest cover, major pulp/paper producer
    "FIN": 0.65,
    # Ireland: UN SNA - fishing ~8%, forestry ~2%
    "IRL": 0.90,
    # Portugal: UN SNA - fishing ~12%, forestry ~8%
    "PRT": 0.80,
    # Greece: UN SNA - fishing ~12%, forestry ~3%
    "GRC": 0.85,
    # Czech Republic: UN SNA - forestry ~10%, landlocked
    "CZE": 0.90,
    # Hungary: UN SNA - landlocked, forestry ~6%
    "HUN": 0.94,
    # Romania: UN SNA - forestry ~10%, fishing ~2%
    "ROU": 0.88,
    # Bulgaria: UN SNA - fishing ~5%, forestry ~5%
    "BGR": 0.90,
    # Croatia: UN SNA - fishing ~8%, forestry ~7%
    "HRV": 0.85,
    # Slovakia: UN SNA - landlocked, forestry ~13%
    "SVK": 0.87,
    # Slovenia: UN SNA - forestry ~18%, fishing ~2%
    "SVN": 0.80,
    # Lithuania: UN SNA - forestry ~12%, fishing ~3%
    "LTU": 0.85,
    # Latvia: UN SNA - forestry ~20%, fishing ~5%
    "LVA": 0.75,
    # Estonia: UN SNA - forestry ~25%, fishing ~5%
    "EST": 0.70,

    # =========================================================================
    # NORTH AMERICA
    # Sources: UN National Accounts, Statistics Canada, USDA
    # =========================================================================
    # USA: USDA - forestry ~5%, fishing ~3% of AFF
    "USA": 0.92,
    # Canada: Statistics Canada - forestry ~22% of AFF (major timber producer)
    # 38% forest cover, 9% of global forest area
    "CAN": 0.75,
    # Mexico: UN SNA - fishing ~8%, forestry ~2%
    "MEX": 0.90,

    # =========================================================================
    # SOUTH AMERICA
    # Sources: UN National Accounts, IBGE (Brazil), national statistics
    # =========================================================================
    # Brazil: IBGE PEVS (2020) - forestry R$23.6B, ~12% of AFF
    # Fishing ~3% of AFF
    "BRA": 0.85,
    # Argentina: UN SNA - forestry ~5%, fishing ~3%
    "ARG": 0.92,
    # Colombia: UN SNA - forestry ~8%, fishing ~4%
    "COL": 0.88,
    # Peru: UN SNA - fishing ~20% of AFF (major anchovy producer)
    # Top 7 capture producer globally, fishing ~5.5M tonnes (2022)
    "PER": 0.75,
    # Chile: UN SNA - fishing ~25%, forestry ~10%
    # Major fishing nation + significant forestry (radiata pine)
    "CHL": 0.65,
    # Ecuador: UN SNA - fishing ~25% (major tuna exporter)
    "ECU": 0.70,
    # Bolivia: landlocked, forestry ~4%
    "BOL": 0.96,
    # Paraguay: landlocked, forestry ~4%
    "PRY": 0.96,
    # Uruguay: UN SNA - forestry ~6%, fishing ~2%
    "URY": 0.92,
    # Venezuela: UN SNA - fishing ~8%, forestry ~4%
    "VEN": 0.88,

    # =========================================================================
    # ASIA
    # Sources: UN National Accounts, FAO fisheries statistics
    # =========================================================================
    # China: UN SNA - fishing ~10%, forestry ~2%
    # World's largest capture producer (15% of global)
    "CHN": 0.88,
    # India: UN SNA - fishing ~6%, forestry ~2%
    "IND": 0.92,
    # Indonesia: Statista/UN - fishing ~15%, forestry ~5%
    # Major archipelago, top 3 capture producer
    "IDN": 0.80,
    # Pakistan: UN SNA - fishing ~3%, forestry ~2%
    "PAK": 0.95,
    # Bangladesh: UN SNA - fishing ~12%, forestry ~3%
    "BGD": 0.85,
    # Vietnam: UN SNA - fishing ~20%, forestry ~5%
    # Major aquaculture producer
    "VNM": 0.75,
    # Thailand: UN SNA - fishing ~20%, forestry ~5%
    "THA": 0.75,
    # Myanmar: UN SNA - fishing ~15%, forestry ~5%
    "MMR": 0.80,
    # Philippines: UN SNA - fishing ~20%, forestry ~5%
    "PHL": 0.75,
    # Malaysia: UN SNA - forestry ~20%, fishing ~10%
    # Major palm oil + timber producer
    "MYS": 0.70,
    # Japan: UN SNA - fishing ~25%, forestry ~5%
    # Major fishing nation historically
    "JPN": 0.70,
    # South Korea: UN SNA - fishing ~20%, forestry ~5%
    "KOR": 0.75,
    # Nepal: landlocked, forestry ~6%
    "NPL": 0.94,
    # Sri Lanka: UN SNA - fishing ~15%, forestry ~5%
    "LKA": 0.80,
    # Cambodia: UN SNA - fishing ~10%, forestry ~5%
    "KHM": 0.85,
    # Laos: landlocked, forestry ~18%
    "LAO": 0.82,

    # =========================================================================
    # MIDDLE EAST & CENTRAL ASIA
    # Sources: UN National Accounts
    # Generally high ag share due to arid climate (limited forestry)
    # =========================================================================
    # Turkey: UN SNA - fishing ~8%, forestry ~4%
    "TUR": 0.88,
    # Iran: UN SNA - fishing ~5%, forestry ~5%
    "IRN": 0.90,
    # Saudi Arabia: minimal forestry/fishing (desert)
    "SAU": 0.97,
    # Iraq: minimal forestry/fishing
    "IRQ": 0.96,
    # Syria: minimal forestry/fishing
    "SYR": 0.96,
    # Israel: UN SNA - fishing ~5%, forestry ~5%
    "ISR": 0.90,
    # Jordan: landlocked, minimal forestry
    "JOR": 0.98,
    # Lebanon: UN SNA - fishing ~5%, forestry ~5%
    "LBN": 0.90,
    # Kazakhstan: landlocked, forestry ~3%
    "KAZ": 0.97,
    # Uzbekistan: landlocked, minimal forestry
    "UZB": 0.98,
    # Turkmenistan: Caspian fishing ~3%, minimal forestry
    "TKM": 0.96,
    # Kyrgyzstan: landlocked, forestry ~3%
    "KGZ": 0.96,
    # Tajikistan: landlocked, forestry ~3%
    "TJK": 0.96,
    # Afghanistan: landlocked, minimal forestry
    "AFG": 0.98,

    # =========================================================================
    # AFRICA
    # Sources: UN National Accounts, FAO Africa statistics
    # =========================================================================
    # Nigeria: UN SNA - fishing ~8%, forestry ~2%
    "NGA": 0.90,
    # Ethiopia: landlocked, forestry ~4%
    "ETH": 0.96,
    # Egypt: UN SNA - fishing ~6%, forestry ~2%
    "EGY": 0.92,
    # South Africa: UN SNA - fishing ~8%, forestry ~4%
    "ZAF": 0.88,
    # Kenya: UN SNA - fishing ~6%, forestry ~4%
    "KEN": 0.90,
    # Tanzania: UN SNA - fishing ~10%, forestry ~5%
    "TZA": 0.85,
    # Uganda: landlocked, forestry ~6%
    "UGA": 0.94,
    # Ghana: UN SNA - fishing ~12%, forestry ~3%
    "GHA": 0.85,
    # Côte d'Ivoire: UN SNA - forestry ~10%, fishing ~5%
    "CIV": 0.85,
    # Cameroon: UN SNA - forestry ~15%, fishing ~5%
    "CMR": 0.80,
    # Angola: UN SNA - fishing ~12%, forestry ~8%
    "AGO": 0.80,
    # Mozambique: UN SNA - fishing ~12%, forestry ~8%
    "MOZ": 0.80,
    # Madagascar: UN SNA - fishing ~12%, forestry ~8%
    "MDG": 0.80,
    # Malawi: landlocked, forestry ~6%
    "MWI": 0.94,
    # Zambia: landlocked, forestry ~6%
    "ZMB": 0.94,
    # Zimbabwe: landlocked, forestry ~6%
    "ZWE": 0.94,
    # Senegal: UN SNA - fishing ~25% (major fishing nation)
    "SEN": 0.70,
    # Mali: landlocked, forestry ~3%
    "MLI": 0.96,
    # Burkina Faso: landlocked, minimal forestry
    "BFA": 0.98,
    # Niger: landlocked, minimal forestry
    "NER": 0.98,
    # Chad: landlocked, forestry ~3%
    "TCD": 0.96,
    # Sudan: UN SNA - fishing ~3%, forestry ~2%
    "SDN": 0.95,
    # Morocco: UN SNA - fishing ~20% (major fishing nation)
    "MAR": 0.75,
    # Algeria: UN SNA - fishing ~5%, forestry ~3%
    "DZA": 0.92,
    # Tunisia: UN SNA - fishing ~12%, forestry ~3%
    "TUN": 0.85,
    # Libya: UN SNA - fishing ~8%, forestry ~2%
    "LBY": 0.90,
    # Namibia: UN SNA - fishing ~20% (major fishing nation)
    "NAM": 0.75,
    # Botswana: landlocked, minimal forestry
    "BWA": 0.98,
    # Eswatini: landlocked, forestry ~4%
    "SWZ": 0.96,
    # Lesotho: landlocked, minimal forestry
    "LSO": 0.98,

    # =========================================================================
    # OCEANIA
    # Sources: UN National Accounts, ABS (Australia), Stats NZ
    # =========================================================================
    # Australia: ABS - forestry ~10%, fishing ~5%
    "AUS": 0.85,
    # New Zealand: Stats NZ - forestry ~15%, fishing ~5%
    "NZL": 0.80,
    # Papua New Guinea: UN SNA - forestry ~20%, fishing ~10%
    "PNG": 0.70,
    # Fiji: UN SNA - fishing ~15%, forestry ~10%
    "FJI": 0.75,

    # =========================================================================
    # RUSSIA & CIS
    # Sources: UN National Accounts, Rosstat
    # =========================================================================
    # Russia: Rosstat - forestry ~15%, fishing ~5%
    # World's largest forest area (815M ha)
    "RUS": 0.80,
    # Ukraine: UN SNA - forestry ~5%, fishing ~3%
    "UKR": 0.92,
    # Belarus: UN SNA - forestry ~12%, fishing ~3%
    "BLR": 0.85,
    # Moldova: landlocked, forestry ~4%
    "MDA": 0.96,
    # Georgia: UN SNA - forestry ~8%, fishing ~4%
    "GEO": 0.88,
    # Armenia: landlocked, forestry ~4%
    "ARM": 0.96,
    # Azerbaijan: Caspian fishing ~6%, forestry ~4%
    "AZE": 0.90,
}


def get_ag_share_of_aff(country_code: str) -> float:
    """Get agriculture share of Ag+Forestry+Fishing for a country.
    
    Returns the fraction of FAO Capital Stock (AFF sector) that belongs
    to agriculture (crops + livestock), excluding forestry and fishing.
    
    Parameters
    ----------
    country_code : str
        ISO3 country code (e.g., "NLD", "USA", "BRA")
    
    Returns
    -------
    float
        Agriculture share in [0, 1].
    
    Examples
    --------
    >>> get_ag_share_of_aff("NOR")  # Norway: major fishing + forestry
    0.60
    >>> get_ag_share_of_aff("BOL")  # Bolivia: landlocked
    0.95
    """
    return AG_SHARE_OF_AFF.get(country_code, DEFAULT_AG_SHARE_OF_AFF)


def get_crop_share_from_qv(
    qv_ds: "xr.Dataset",
    country_code: str,
    year: int | None = None,
    avg_years: int = 5,
    max_lookback: int = 20,
    neighbour_codes: list[str] | None = None,
) -> float:
    """Get crop share of agriculture from FAO QV data.

    Computes crop_share = GPV_crops / GPV_agriculture using the generic
    tiered fallback mechanism:

    1. **Tier A (country)**: Same country with expanding time window
    2. **Tier B (neighbours)**: Mean from neighbouring countries
    3. **Tier C (global)**: Global mean across all countries

    Parameters
    ----------
    qv_ds : xr.Dataset
        Dataset from FaoGrossProductionValue containing "gpv_crops" and
        "gpv_agriculture" variables.
    country_code : str
        ISO3 country code.
    year : int | None
        Target year. If None, uses the most recent available year.
    avg_years : int
        Initial window size for time averaging (default 5).
    max_lookback : int
        Maximum years to look back if initial window has no data (default 20).
    neighbour_codes : list[str] | None
        ISO3 codes of neighbouring countries (from country.neighbourhood).
        If None, Tier B is skipped.

    Returns
    -------
    float
        Crop share of agriculture in [0, 1].

    Raises
    ------
    ValueError
        If FAO QV data is not available at any tier.
    """
    from .base import get_value_with_fallback

    # Check required variables exist
    if "gpv_crops" not in qv_ds.data_vars or "gpv_agriculture" not in qv_ds.data_vars:
        raise ValueError(
            f"FAO QV dataset missing required variables. "
            f"Found: {list(qv_ds.data_vars)}. "
            f"Required: gpv_crops, gpv_agriculture"
        )

    # Get GPV for crops using tiered fallback
    crops_result = get_value_with_fallback(
        qv_ds["gpv_crops"],
        country_code,
        year=year,
        avg_years=avg_years,
        max_lookback=max_lookback,
        neighbour_codes=neighbour_codes,
        aggregator="mean",
        field_name="gpv_crops",
    )

    # Get GPV for agriculture using tiered fallback
    ag_result = get_value_with_fallback(
        qv_ds["gpv_agriculture"],
        country_code,
        year=year,
        avg_years=avg_years,
        max_lookback=max_lookback,
        neighbour_codes=neighbour_codes,
        aggregator="mean",
        field_name="gpv_agriculture",
    )

    return crops_result.value / ag_result.value


def compute_crop_capital_share(
    qv_ds: "xr.Dataset",
    country_code: str,
    year: int | None = None,
    avg_years: int = 5,
    max_lookback: int = 20,
    neighbour_codes: list[str] | None = None,
) -> float:
    """Compute total crop capital share from FAO data.

    Combines:
    1. Agriculture share of Ag+F+F (from static table)
    2. Crop share of agriculture (from FAO QV data with tiered fallback)

    Final: crop_capital_share = ag_share × crop_share

    Parameters
    ----------
    qv_ds : xr.Dataset
        Dataset from FaoGrossProductionValue.
    country_code : str
        ISO3 country code.
    year : int | None
        Target year for FAO QV data.
    avg_years : int
        Initial window size for time averaging (default 5).
    max_lookback : int
        Maximum years to look back if initial window has no data (default 20).
    neighbour_codes : list[str] | None
        ISO3 codes of neighbouring countries (from country.neighbourhood).

    Returns
    -------
    float
        Total crop capital share in [0, 1].

    Raises
    ------
    ValueError
        If FAO QV data is not available at any tier.
    """
    ag_share = get_ag_share_of_aff(country_code)
    crop_share = get_crop_share_from_qv(
        qv_ds, country_code, year, avg_years, max_lookback, neighbour_codes
    )
    
    return ag_share * crop_share
