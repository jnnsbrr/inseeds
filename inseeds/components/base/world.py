import os
import sys
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


class World:
    """Define properties.
    Inherits from I.World as the interface with all necessary variables
    and parameters.
    """

    def __init__(self, **kwargs):
        """Initialize an instance of World."""
        super(World, self).__init__(**kwargs)

    def update_output_table(self, t, init=False):
        df = self.create_output_table(t)
        if not hasattr(sys, "_called_from_test"):
            self.write_output_parquet(df, init)
            self.write_output_csv(df, init)

    def create_output_table(self, t):
        """Initialize output data"""
        # create sample time and cell data

        entities = {
            "World": "world",
            "Cell": "cell",
            "Farmer": "farmer",
            "Individual": "individual",
            "SocialSystem": "social_system",
        }
        taxa = {
            "Environment": "environment",
            "Metabolism": "metabolism",
            "Culture": "culture",
        }
        core_classes = {
            key: value
            for key, value in {**entities, **taxa}.items()
            if hasattr(self.model, key)
            and hasattr(self.lpjml.config.coupled_config.output, value)
        }

        df = pd.DataFrame()

        for copan_interface, core_class in core_classes.items():
            for var in getattr(self.lpjml.config.coupled_config.output, core_class):
                df_data = {
                    "year": [t] * len(getattr(self, f"{core_class}s")),
                }

                if core_class in ["cell", "individual", "farmer"]:
                    if core_class in ["individual", "farmer"]:
                        call = ".cell"
                    else:
                        call = ""

                    df_data["cell"] = [
                        eval(f"attr{call}.grid.cell.item()")
                        for attr in getattr(self, f"{core_class}s")
                    ]

                    if (
                        self.lpjml.config.coupled_config.output_settings.write_lon_lat  # noqa
                    ):  # noqa
                        df_data["lon"] = [
                            eval(f"attr{call}.grid.lon.item()")
                            for attr in getattr(self, f"{core_class}s")
                        ]
                        df_data["lat"] = [
                            eval(f"attr{call}.grid.lat.item()")
                            for attr in getattr(self, f"{core_class}s")
                        ]

                    if hasattr(self.lpjml, "country"):
                        df_data["country"] = [
                            eval(f"attr{call}.country.item()")
                            for attr in getattr(self, f"{core_class}s")
                        ]
                    if hasattr(self.lpjml, "terr_area"):
                        df_data["area [km2]"] = [
                            eval(f"attr{call}.area.item()") * 1e-6
                            for attr in getattr(self, f"{core_class}s")
                        ]

                variable = [
                    eval(f"self.model.{copan_interface}.{var}.name")
                ] * len(  # noqa
                    getattr(self, f"{core_class}s")
                )

                if core_class == "world":
                    df_data["class"] = [core_class]
                    df_data["variable"] = variable
                    df_data["value"] = [eval(f"self.{var}")]

                else:
                    df_data["class"] = [core_class] * len(
                        getattr(self, f"{core_class}s")
                    )
                    df_data["variable"] = variable
                    df_data["value"] = [
                        eval(f"attr.{var}")
                        for attr in getattr(self, f"{core_class}s")  # noqa
                    ]

                if hasattr(
                    eval(f"self.model.{copan_interface}.{var}.unit"), "symbol"
                ):  # noqa
                    df_data["unit"] = [
                        eval(f"self.model.{copan_interface}.{var}.unit.symbol")  # noqa
                    ] * len(getattr(self, f"{core_class}s"))

                df = pd.concat([df, pd.DataFrame(df_data)])

        return df

    def write_output_csv(self, df, init=False):
        """Write output data"""
        mode = (
            "w"
            if (
                self.lpjml.sim_year == self.lpjml.config.start_coupling and init
            )  # noqa
            else "a"
        )

        # define the file name and header row
        file_name = f"{self.lpjml.config.sim_path}/output/{self.lpjml.config.sim_name}/inseeds_data.csv"  # noqa

        if not os.path.isfile(file_name) or mode == "w":
            header = True
        else:
            header = False

        df.to_csv(file_name, mode=mode, header=header, index=False)

    def write_output_parquet(self, df, init=False):
        """Write output data to Parquet file"""
        file_name = f"{self.lpjml.config.sim_path}/output/{self.lpjml.config.sim_name}/inseeds_data.parquet"  # noqa

        # Append mode: write new data without rewriting the file.
        if not os.path.isfile(file_name) or (
            self.lpjml.sim_year == self.lpjml.config.start_coupling and init
        ):
            df.to_parquet(file_name, engine="pyarrow", index=False)
        else:
            # Read the existing data
            existing_data = pd.read_parquet(file_name)

            # Concatenate the existing data with the new data
            combined_data = pd.concat([existing_data, df], ignore_index=True)

            # Write the combined data back
            combined_data.to_parquet(file_name, engine="pyarrow", index=False)
