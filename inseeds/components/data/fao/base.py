"""Base class for FAO dataset handlers.

This module provides the abstract base class for downloading, transforming,
and caching FAOSTAT data for InSEEDS simulations.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Self

import pandas as pd
import requests
import xarray as xr

from copan_eval.fao import FaoData, fao_definitions, FaoApiAdapter, FaoCropTranslator


def check_fao_api_available(timeout: float = 5.0) -> bool:
    """Check if FAO API is available and accessible.

    Performs a quick test request to the FAO API definitions endpoint.
    This fails fast if the API requires authentication or is down.

    Parameters
    ----------
    timeout : float
        Request timeout in seconds.

    Returns
    -------
    bool
        True if API is accessible, False otherwise.
    """
    try:
        response = requests.get(
            "https://faostatservices.fao.org/api/v1/en/definitions/types/area",
            timeout=timeout,
        )
        return response.status_code == 200
    except Exception:
        return False


class FaoDataset(ABC):
    """Abstract base class for FAO dataset handlers.

    Subclasses must implement:
        - domain: FAOSTAT domain code (e.g., "PP", "CS")
        - elements: List of element codes to download
        - name: Human-readable name for the dataset
        - output_filename: Filename for the output NetCDF file
        - _get_items(): Returns item codes to download
        - _post_process(ds): Domain-specific transformations

    The base class provides shared functionality for:
        - Chunked download from FAO API (handling 500-record limit)
        - Coordinate transformations (FAO→ISO3 area codes, cftime→int years)
        - Caching and lazy loading
    """

    @property
    @abstractmethod
    def domain(self) -> str:
        """FAOSTAT domain code (e.g., 'PP', 'CS', 'QCL')."""
        ...

    @property
    @abstractmethod
    def elements(self) -> list[str]:
        """List of FAOSTAT element codes to download."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this dataset."""
        ...

    @property
    @abstractmethod
    def output_filename(self) -> str:
        """Filename for the output NetCDF file."""
        ...

    @abstractmethod
    def _get_items(self) -> pd.Series:
        """Return item codes to download from FAOSTAT.

        Returns
        -------
        pd.Series
            Series of item codes appropriate for this domain.
        """
        ...

    @abstractmethod
    def _post_process(self, ds: xr.Dataset) -> xr.Dataset:
        """Apply domain-specific post-processing to the dataset.

        Parameters
        ----------
        ds : xr.Dataset
            Dataset after standard transformations.

        Returns
        -------
        xr.Dataset
            Dataset with domain-specific transformations applied.
        """
        ...

    def _translate_to_lpjml(self) -> bool:
        """Whether to translate item codes to LPJmL CFT names.

        Override to return False for datasets that don't use crop items
        (e.g., capital stock which uses sector codes).
        """
        return True

    def get_path(self, sim_path: str | Path) -> Path:
        """Get the path to the real FAO data file in simulation input folder.

        Parameters
        ----------
        sim_path : str | Path
            Simulation path (pycoupler sim_path).

        Returns
        -------
        Path
            Path to output file in {sim_path}/input/
        """
        return Path(sim_path) / "input" / self.output_filename

    def get_dummy_path(self, sim_path: str | Path) -> Path:
        """Get the path to the dummy data file in simulation input folder.

        Dummy files have '_DUMMY' suffix to distinguish them from real data.

        Parameters
        ----------
        sim_path : str | Path
            Simulation path (pycoupler sim_path).

        Returns
        -------
        Path
            Path to dummy file in {sim_path}/input/
        """
        base = self.output_filename.replace(".nc", "_DUMMY.nc")
        return Path(sim_path) / "input" / base

    def download(
        self,
        adapter: FaoApiAdapter,
        years: list[str],
        countries: pd.Series,
    ) -> pd.DataFrame:
        """Download data from FAOSTAT API with chunking.

        The FAO API has a limit of 500 records per request. This method
        handles chunking by country and year to stay under the limit.

        Parameters
        ----------
        adapter : FaoApiAdapter
            FAO API adapter instance.
        years : list[str]
            List of years to download.
        countries : pd.Series
            Country codes to download.

        Returns
        -------
        pd.DataFrame
            Downloaded data from FAOSTAT.
        """
        items = self._get_items()

        year_chunk_size = 1
        country_chunk_size = 2

        country_list = list(countries)
        year_chunks = [
            years[i : i + year_chunk_size]
            for i in range(0, len(years), year_chunk_size)
        ]
        country_chunks = [
            country_list[i : i + country_chunk_size]
            for i in range(0, len(country_list), country_chunk_size)
        ]

        total_chunks = len(year_chunks) * len(country_chunks) * len(self.elements)
        print(
            f"Downloading {self.name} from FAOSTAT ({self.domain} domain)..."
        )
        print(
            f"  {total_chunks} chunks "
            f"({len(year_chunks)} years × {len(country_chunks)} countries "
            f"× {len(self.elements)} elements)"
        )

        dfs = []
        chunk_count = 0
        for element in self.elements:
            for year_chunk in year_chunks:
                for country_chunk in country_chunks:
                    chunk_count += 1
                    if chunk_count % 50 == 0:
                        print(f"  Progress: {chunk_count}/{total_chunks} chunks...")
                    try:
                        df = adapter.download_data(
                            domain=self.domain,
                            element=element,
                            year=year_chunk,
                            items=items,
                            areas=country_chunk,
                            item_code_format="FAO",
                            area_code_format="FAO",
                        )
                        if len(df) > 0:
                            dfs.append(df)
                    except Exception as e:
                        print(
                            f"Warning: Failed chunk {element}/{year_chunk}/"
                            f"{country_chunk}: {e}"
                        )

        if not dfs:
            raise RuntimeError(f"No data downloaded for {self.name}")

        print(
            f"Downloaded {len(dfs)} chunks with data, "
            f"total {sum(len(df) for df in dfs)} records"
        )
        return pd.concat(dfs, ignore_index=True)

    def transform(self, df: pd.DataFrame) -> xr.Dataset:
        """Transform downloaded DataFrame to xarray Dataset.

        Applies standard transformations:
        - Convert to FaoData via from_dataframe()
        - Translate crop codes to LPJmL CFT names (if applicable)
        - Convert area codes from FAO to ISO3
        - Convert time coordinates from cftime to integer years

        Parameters
        ----------
        df : pd.DataFrame
            Downloaded FAOSTAT data.

        Returns
        -------
        xr.Dataset
            Transformed dataset.
        """
        print(f"Converting {self.name} to xarray format...")

        fao_data_list = FaoData.from_dataframe(
            df, multi_element=len(self.elements) > 1
        )
        fao_data_list = [d for d in fao_data_list if d is not None]

        if not fao_data_list:
            raise RuntimeError(f"Failed to convert {self.name} to FaoData")

        if self._translate_to_lpjml():
            print("  Translating to LPJmL CFT format...")
            translated = []
            for fao_data in fao_data_list:
                fao_data_npft = fao_data.translate_dimension(
                    FaoCropTranslator(
                        dim="item_code",
                        new_standard="LPJmL",
                        reducer="mean",
                        fail_on_partial=False,
                        dim_rename="npft",
                    )
                )
                translated.append(fao_data_npft.dataset)

            if len(translated) == 1:
                ds = translated[0]
            else:
                ds = xr.merge(translated)
        else:
            if len(fao_data_list) == 1:
                ds = fao_data_list[0].dataset
            else:
                ds = xr.merge([d.dataset for d in fao_data_list])

        ds = self._convert_area_codes(ds)
        ds = self._convert_time_coords(ds)

        return ds

    def _convert_area_codes(self, ds: xr.Dataset) -> xr.Dataset:
        """Convert area codes from FAO to ISO3 format."""
        if "area_code" not in ds.dims:
            return ds

        print("  Converting area codes to ISO3...")
        fao_to_iso3 = fao_definitions.get_area_code_dict(
            code_standard_out="ISO3", code_standard_in="FAO"
        )
        current_codes = ds.area_code.values
        iso3_codes = [fao_to_iso3.get(str(c), str(c)) for c in current_codes]

        valid_mask = [bool(c) for c in iso3_codes]
        if not all(valid_mask):
            valid_indices = [i for i, v in enumerate(valid_mask) if v]
            ds = ds.isel(area_code=valid_indices)
            iso3_codes = [iso3_codes[i] for i in valid_indices]

        ds = ds.assign_coords(area_code=iso3_codes)
        return ds

    def _convert_time_coords(self, ds: xr.Dataset) -> xr.Dataset:
        """Convert time coordinates from cftime to integer years."""
        if "time" not in ds.dims:
            return ds

        print("  Converting time coordinates to integer years...")
        time_values = ds.time.values
        if hasattr(time_values[0], "year"):
            years = [int(t.year) for t in time_values]
        else:
            years = [int(t) for t in time_values]

        ds = ds.assign_coords(time=years)
        ds.time.attrs["units"] = "year"
        ds.time.attrs["long_name"] = "Year"
        return ds

    def prepare(
        self,
        cache_path: str | Path | None = None,
        output_path: str | Path | None = None,
        years: tuple[int, int] = (1990, 2020),
    ) -> xr.Dataset:
        """Prepare FAO data: download, transform, and optionally save.

        Parameters
        ----------
        cache_path : str | Path | None
            Path to cache downloaded data (parquet). If exists, loads from cache.
        output_path : str | Path | None
            Path to save output NetCDF. If None, returns dataset without saving.
        years : tuple[int, int]
            Year range (start, end) inclusive.

        Returns
        -------
        xr.Dataset
            Prepared dataset.

        Raises
        ------
        RuntimeError
            If FAO API is unavailable and no cache exists.
        """
        year_list = [str(y) for y in range(years[0], years[1] + 1)]

        # Check cache first
        if cache_path and Path(cache_path).exists():
            print(f"Loading {self.name} from cache: {cache_path}")
            df = pd.read_parquet(cache_path)
        else:
            # Check API availability before attempting download
            print(f"Checking FAO API availability...")
            if not check_fao_api_available():
                raise RuntimeError(
                    "FAO API is unavailable (requires authentication or is down). "
                    "Use dummy data fallback or provide cached data."
                )

            all_countries = fao_definitions.get_all_country_codes()
            adapter = FaoApiAdapter()
            df = self.download(adapter, year_list, all_countries)
            if cache_path:
                print(f"Saving to cache: {cache_path}")
                Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
                df.to_parquet(cache_path)

        ds = self.transform(df)
        ds = self._post_process(ds)

        if output_path:
            print(f"Saving to {output_path}...")
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            ds.to_netcdf(output_path)

        print(f"Done preparing {self.name}!")
        print(f"  Dimensions: {dict(ds.dims)}")
        print(f"  Variables: {list(ds.data_vars)}")

        return ds

    def ensure(
        self,
        sim_path: str | Path,
        years: tuple[int, int] = (1990, 2020),
        force_download: bool = False,
        use_dummy_on_failure: bool = True,
    ) -> Path:
        """Ensure FAO data is available in simulation input folder.

        Priority order:
        1. Real data file exists → use silently
        2. Dummy data file exists → use with warning
        3. Download from FAO API → save as real data
        4. API fails + use_dummy_on_failure → generate dummy data

        Parameters
        ----------
        sim_path : str | Path
            Simulation path (pycoupler sim_path).
        years : tuple[int, int]
            Year range for FAO data (start, end) inclusive.
        force_download : bool
            If True, re-download even if file exists.
        use_dummy_on_failure : bool
            If True (default), generate dummy data when FAO API fails.
            This allows the model to run for testing/development even
            without FAO API access.

        Returns
        -------
        Path
            Path to the FAO data file (real or dummy).

        Raises
        ------
        RuntimeError
            If data cannot be downloaded/prepared and use_dummy_on_failure=False.
        """
        output_path = self.get_path(sim_path)
        dummy_path = self.get_dummy_path(sim_path)

        # Priority 1: Real data exists - use silently
        if output_path.exists() and not force_download:
            return output_path

        # Priority 2: Dummy data exists - use with warning (once per session)
        if dummy_path.exists() and not force_download:
            return dummy_path

        # Priority 3 & 4: Download or generate
        print(f"Downloading {self.name} from FAOSTAT...")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path = output_path.parent / f"{self.name}_cache.parquet"

        try:
            self.prepare(
                cache_path=cache_path,
                output_path=output_path,
                years=years,
            )
            return output_path
        except Exception as e:
            if use_dummy_on_failure:
                print(f"WARNING: FAO API failed: {e}")
                print(f"Generating DUMMY data for {self.name}...")
                self._generate_dummy_fallback(dummy_path, years)
                return dummy_path
            else:
                raise RuntimeError(
                    f"Failed to download/prepare {self.name}: {e}\n"
                    f"Please check your internet connection and try again, or "
                    f"manually prepare the data and place it at: {output_path}"
                ) from e

    def _generate_dummy_fallback(
        self,
        output_path: Path,
        years: tuple[int, int],
    ) -> None:
        """Generate dummy data as fallback when FAO API fails.

        Override in subclasses to provide domain-specific dummy data.

        Parameters
        ----------
        output_path : Path
            Where to save the dummy data.
        years : tuple[int, int]
            Year range for data generation.
        """
        raise NotImplementedError(
            f"Dummy data generation not implemented for {self.name}. "
            f"Please provide data manually at: {output_path}"
        )

    def is_available(self, sim_path: str | Path) -> bool:
        """Check if FAO data (real or dummy) is available.

        Parameters
        ----------
        sim_path : str | Path
            Simulation path (pycoupler sim_path).

        Returns
        -------
        bool
            True if real or dummy data file exists.
        """
        return self.get_path(sim_path).exists() or self.get_dummy_path(sim_path).exists()

    def is_dummy(self, sim_path: str | Path) -> bool:
        """Check if only dummy data is available (no real data).

        Parameters
        ----------
        sim_path : str | Path
            Simulation path (pycoupler sim_path).

        Returns
        -------
        bool
            True if using dummy data (real data not available).
        """
        return not self.get_path(sim_path).exists() and self.get_dummy_path(sim_path).exists()
