import os
import sys
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import shutil


class Component:
    """Model mixin class."""

    def __init__(self, **kwargs):
        """Initialize the model mixin."""
        pass

    @property
    def output_table(self):
        # get all world outputs
        df = self.world.output_table

        # get all cell outputs
        if hasattr(self.world, "cells"):
            df = pd.concat(
                [df] + [cell.output_table for cell in self.world.cells]
            )

        # get all farmer outputs
        if hasattr(self.world, "farmers"):
            df = pd.concat(
                [df] + [farmer.output_table for farmer in self.world.farmers]
            )

        return df

    def write_output_table(self, init=False, file_format="parquet"):
        if hasattr(sys, "_called_from_test"):
            return
        if file_format == "parquet":
            self.write_output_parquet(self.output_table, init)
        elif file_format == "csv":
            self.write_output_csv(self.output_table, init)
        else:
            raise ValueError(f"Output file format {file_format} not supported")

    def write_output_csv(self, df, init=False):
        """Write output data to CSV.

        Parameters
        ----------
        df : pandas.DataFrame
            Table to be written.
        init : bool, optional
            If True and at the first coupling year, overwrite any existing
            file; otherwise append. Default is False.
        """
        if df is None or df.empty:
            return
        mode = (
            "w"
            if (
                self.lpjml.sim_year == self.config.start_coupling and init
            )  # noqa
            else "a"
        )

        # define the file name and header row
        file_name = (
            f"{self.config.sim_path}/output/"
            f"{self.config.sim_name}/inseeds_data.csv"
        )

        # ensure output directory exists
        os.makedirs(os.path.dirname(file_name), exist_ok=True)

        if not os.path.isfile(file_name) or mode == "w":
            header = True
        else:
            header = False

        # optional CSV compression from config, e.g., "gzip", "bz2"
        csv_compression = None
        try:
            csv_compression = getattr(
                self.config.coupled_config.output_settings,
                "csv_compression",
                None,
            )
        except Exception:
            pass

        df.to_csv(
            file_name,
            mode=mode,
            header=header,
            index=False,
            compression=csv_compression,
        )

    def write_output_parquet(self, df, init=False):
        """Write output data to a Parquet dataset with partitioning.

        Appends efficiently by writing a new Parquet file per partition instead
        of reading and rewriting a single large file.

        Parameters
        ----------
        df : pandas.DataFrame
            Table to be written.
        init : bool, optional
            If True and at the first coupling year, the dataset directory is
            reset; otherwise new data are appended as additional files within
            partitions. If the dataset directory does not exist, it is
            created. Default is False.
        """
        if df is None or df.empty:
            return

        # Base directory that will contain a Parquet dataset (folder)
        base_dir = (
            f"{self.config.sim_path}/output/"
            f"{self.config.sim_name}/inseeds_data.parquet"
        )

        # Ensure base directory exists
        # If this is the initial write at start year, reset the dataset dir
        try:
            if (
                self.lpjml.sim_year == self.config.start_coupling
                and init
                and os.path.isdir(base_dir)
            ):
                shutil.rmtree(base_dir)
        except Exception:
            # Non-fatal; proceed to create/append
            pass
        os.makedirs(base_dir, exist_ok=True)

        # Retrieve optional settings
        partition_by = ["year", "entity"]
        compression = "zstd"
        try:
            settings = getattr(
                self.config.coupled_config, "output_settings", None
            )
            if settings is not None:
                p = getattr(settings, "partition_by", None)
                if isinstance(p, (list, tuple)) and len(p) > 0:
                    partition_by = list(p)
                c = getattr(settings, "compression", None)
                if isinstance(c, str) and len(c) > 0:
                    compression = c
        except Exception:
            pass

        # Optimize dtypes for storage (categoricals enable dictionary encoding)
        df_to_write = self._optimize_df_for_storage(df)

        # Convert to Arrow Table and write to a partitioned dataset
        table = pa.Table.from_pandas(df_to_write, preserve_index=False)

        # Only partition by columns that actually exist in the data
        present_partition_cols = [
            col for col in partition_by if col in df_to_write.columns
        ]

        # Use write_to_dataset for broad PyArrow compatibility
        pq.write_to_dataset(
            table,
            root_path=base_dir,
            partition_cols=present_partition_cols or None,
            compression=compression,
            use_dictionary=True,
        )

    def _optimize_df_for_storage(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of ``df`` with storage-friendly dtypes.

        - Convert common string columns to pandas ``category`` to enable Arrow
          dictionary encoding (smaller Parquet files, faster IO)
        - Downcast integer columns where safe
        - Optionally downcast float precision if configured

        Parameters
        ----------
        df : pandas.DataFrame
            Input table.

        Returns
        -------
        pandas.DataFrame
            Optimized copy.
        """
        if df is None or df.empty:
            return df

        optimized = df.copy()

        categorical_candidates = [
            col
            for col in ["entity", "variable", "unit", "country"]
            if col in optimized.columns and optimized[col].dtype == object
        ]
        for col in categorical_candidates:
            optimized[col] = optimized[col].astype("category")

        if "year" in optimized.columns:
            optimized["year"] = pd.to_numeric(
                optimized["year"], errors="coerce"
            ).astype("Int32")
        if "cell" in optimized.columns:
            optimized["cell"] = pd.to_numeric(
                optimized["cell"], errors="coerce"
            ).astype("Int32")

        # Optional float downcast
        float_precision = None
        try:
            settings = getattr(
                self.config.coupled_config, "output_settings", None
            )
            if settings is not None:
                float_precision = getattr(settings, "float_precision", None)
        except Exception:
            pass
        if float_precision in (16, 32):
            float_type = {16: "float16", 32: "float32"}[float_precision]
            for col in optimized.select_dtypes(
                include=["float", "float64"]
            ).columns:
                optimized[col] = optimized[col].astype(float_type)

        return optimized

    def update(self, t):
        """Update the model."""
        pass
