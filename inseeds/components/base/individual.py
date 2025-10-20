import os
import sys
import pandas as pd

from . import Entity


class Individual(Entity):
    """Define properties.
    Inherits from I.World as the interface with all necessary variables
    and parameters.
    """

    @property
    def output_table(self):
        variables = self.get_defined_outputs()

        if not variables:
            return pd.DataFrame()
        else:
            df = super().output_table

            # Get cell index - use stored _cell_index if available, otherwise extract from grid
            if hasattr(self.cell, "_cell_index"):
                cell_id = self.cell._cell_index
            else:
                # Fallback: extract from grid.cell
                cell_coord = self.cell.grid.cell
                if hasattr(cell_coord, "values"):
                    cell_vals = cell_coord.values
                    cell_id = int(
                        cell_vals.item()
                        if cell_vals.shape == ()
                        else cell_vals[0]
                    )
                else:
                    cell_id = int(cell_coord)

            df.insert(1, "cell", [cell_id] * len(variables))

            # Get longitude and latitude from grid coordinates
            if (
                hasattr(self.cell.grid, "coords")
                and "lon" in self.cell.grid.coords
            ):
                lon_val = self.cell.grid.coords["lon"].values
                lat_val = self.cell.grid.coords["lat"].values
                lon = lon_val.item() if lon_val.shape == () else lon_val[0]
                lat = lat_val.item() if lat_val.shape == () else lat_val[0]
            elif hasattr(self.cell.grid, "data"):
                # Fallback: try to get from grid data directly
                grid_data = self.cell.grid.data
                if grid_data.ndim == 2:
                    lon = grid_data[0, 0]
                    lat = grid_data[0, 1]
                elif grid_data.ndim == 1:
                    # Single cell case - data might be [lon, lat] or just one value
                    lon = grid_data[0] if len(grid_data) > 0 else 0.0
                    lat = grid_data[1] if len(grid_data) > 1 else 0.0
                else:
                    lon = lat = 0.0
            else:
                lon = lat = 0.0

            df.insert(2, "lon", [lon] * len(variables))
            df.insert(3, "lat", [lat] * len(variables))

            # Add country information - prefer country_code if available, otherwise country name
            if (
                hasattr(self.cell, "country_code")
                and self.cell.country_code is not None
            ):
                country_val = self.cell.country_code
                # Handle case where country_code might be an array or list
                if hasattr(country_val, "item"):
                    country_val = country_val.item()
                elif hasattr(country_val, "__iter__") and not isinstance(
                    country_val, str
                ):
                    country_val = (
                        country_val[0] if len(country_val) > 0 else None
                    )
                # Ensure we have a string, not a list representation
                if (
                    isinstance(country_val, str)
                    and country_val.startswith("['")
                    and country_val.endswith("']")
                ):
                    # Extract from string representation like "['NLD']"
                    country_val = country_val[2:-2]  # Remove "['" and "']"
                df.insert(4, "country", [country_val] * len(variables))
            elif (
                hasattr(self.cell, "country") and self.cell.country is not None
            ):
                df.insert(
                    4, "country", [self.cell.country.name] * len(variables)
                )
            if hasattr(self.cell, "area"):
                df.insert(
                    5,
                    "area [km2]",
                    [round(self.cell.area.item() * 1e-6, 4)] * len(variables),
                )
            return df
