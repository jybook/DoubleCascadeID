"""Example of converting Parquet files to SQLite database."""

import os.path
from typing import List

from graphnet.constants import EXAMPLE_OUTPUT_DIR, TEST_PARQUET_DATA
from graphnet.data.utilities.parquet_to_sqlite import ParquetToSQLiteConverter
from graphnet.data.extractors.internal import ParquetExtractor
from graphnet.utilities.argparse import ArgumentParser
from graphnet.utilities.logging import Logger

PARENT_DIR = '/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/jbook/double_cascade_identification/'
MODEL_CONFIG_DIR = PARENT_DIR + 'model_configs'
DATASETS_CONFIG_DIR = PARENT_DIR + 'dataset_configs/'
OUTPUT_DIR = PARENT_DIR + 'sqlite'

def main(parquet_path: str, tables: List[str]) -> None:
    """Run example."""
    # Construct Logger
    logger = Logger()

    # Path to where you want the database to be stored
    outdir = f"{OUTPUT_DIR}/convert_parquet_to_sqlite/"

    if os.path.exists(outdir):
        logger.error(
            f"Output database {outdir} already exists. The conversion "
            "will likely fail. If so, please remove the output database "
            "before running this script again."
        )

    extractors = [ParquetExtractor(table) for table in tables]
    converter = ParquetToSQLiteConverter(
        extractors=extractors,
        outdir=outdir,
        num_workers=1,
        index_column="event_no",
    )
    converter(parquet_path)


if __name__ == "__main__":

    # Parse command-line arguments
    parser = ArgumentParser(
        description="""
Convert Parquet files to SQLite database.
"""
    )

    parser.add_argument(
        "--parquet-path",
        action="store",
        help="Path to Parquet file (default: %(default)s)",
        default=PARENT_DIR+'parquet',
    )

    parser.add_argument(
        "--tables",
        action="store",
        help="Parquet tables to convert (default: %(default)s)",
        default=["truth", "SRTInIcePulses"],
    )
    args, unknown = parser.parse_known_args()

    main(args.parquet_path, args.tables)
