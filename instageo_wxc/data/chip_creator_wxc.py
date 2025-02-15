# ------------------------------------------------------------------------------
# This code is licensed under the Attribution-NonCommercial-ShareAlike 4.0
# International (CC BY-NC-SA 4.0) License.
#
# You are free to:
# - Share: Copy and redistribute the material in any medium or format
# - Adapt: Remix, transform, and build upon the material
#
# Under the following terms:
# - Attribution: You must give appropriate credit, provide a link to the license,
#   and indicate if changes were made. You may do so in any reasonable manner,
#   but not in any way that suggests the licensor endorses you or your use.
# - NonCommercial: You may not use the material for commercial purposes.
# - ShareAlike: If you remix, transform, or build upon the material, you must
#   distribute your contributions under the same license as the original.
#
# For more details, see https://creativecommons.org/licenses/by-nc-sa/4.0/
# ------------------------------------------------------------------------------

"""InstaGeo Chip Creator Module."""

import bisect
import json
import os
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from absl import app, flags, logging
from shapely.geometry import Point
from tqdm import tqdm

import instageo_wxc.data.merra2_utils_wxc as m2_utils
from instageo_wxc.data.geo_utils_wxc import open_mf_tiff_dataset

logging.set_verbosity(logging.INFO)

FLAGS = flags.FLAGS

flags.DEFINE_string("dataframe_path", None, "Path to the DataFrame CSV file.")
flags.DEFINE_integer("chip_size", 224, "Size of each chip.")
flags.DEFINE_integer("src_crs", 4326, "CRS of the geo-coordinates in `dataframe_path`")
flags.DEFINE_string(
    "output_directory",
    None,
    "Directory where the chips and segmentation maps will be saved.",
)
flags.DEFINE_integer(
    "no_data_value", -1, "Value to use for no data areas in the segmentation maps."
)
flags.DEFINE_integer("min_count", 100, "Minimum observation counts per tile")
flags.DEFINE_integer("num_steps", 3, "Number of temporal steps")
flags.DEFINE_integer("temporal_step", 30, "Temporal step size.")
flags.DEFINE_integer(
    "temporal_tolerance", 5, "Tolerance used when searching for the closest tile"
)
flags.DEFINE_boolean(
    "download_only", False, "Downloads MERRA-2 dataset without creating chips."
)
flags.DEFINE_boolean("mask_cloud", False, "Perform Cloud Masking")


def check_required_flags() -> None:
    """Check if required flags are provided."""
    required_flags = ["dataframe_path", "output_directory"]
    for flag_name in required_flags:
        if not getattr(FLAGS, flag_name):
            raise app.UsageError(f"Flag --{flag_name} is required.")


def create_segmentation_map(
    chip: Any,
    df: pd.DataFrame,
    no_data_value: int,
) -> np.ndarray:
    """Create a segmentation map for the chip using the DataFrame.

    Args:
        chip (Any): The chip (subset of the original data) for which the segmentation
            map is being created.
        df (pd.DataFrame): DataFrame containing the data to be used in the segmentation
            map.
        no_data_value (int): Value to be used for pixels with no data.

    Returns:
        np.ndarray: The created segmentation map as a NumPy array.
    """
    seg_map = chip.isel(band=0).assign(
        {
            "band_data": (
                ("y", "x"),
                no_data_value * np.ones((chip.sizes["x"], chip.sizes["y"])),
            )
        }
    )
    df = df[
        (chip["x"].min().item() <= df["geometry"].x)
        & (df["geometry"].x <= chip["x"].max().item())
        & (chip["y"].min().item() <= df["geometry"].y)
        & (df["geometry"].y <= chip["y"].max().item())
    ]
    # Use a tolerance of 30 meters
    for _, row in df.iterrows():
        nearest_index = seg_map.sel(
            x=row["geometry"].x, y=row["geometry"].y, method="nearest", tolerance=30
        )
        seg_map.loc[
            dict(x=nearest_index["x"].values.item(), y=nearest_index["y"].values.item())
        ] = row["label"]
    return seg_map.band_data.squeeze()


def get_chip_coords(
    df: gpd.GeoDataFrame, tile: xr.DataArray, chip_size: int
) -> list[tuple[int, int]]:
    """Get Chip Coordinates.

    Given a list of x,y coordinates tuples of a point and an xarray dataarray, this
    function returns the corresponding x,y indices of the grid where each point will fall
    when the dataarray is gridded such that each grid has size `chip_size`
    indices where it will fall.

    Args:
        gdf (gpd.GeoDataFrame): GeoPandas dataframe containing the point.
        tile (xr.DataArray): Tile DataArray.
        chip_size (int): Size of each chip.

    Returns:
        List of chip indices.
    """
    coords = []
    for _, row in df.iterrows():
        x = bisect.bisect_left(tile["x"].values, row["geometry"].x)
        y = bisect.bisect_left(tile["y"].values[::-1], row["geometry"].y)
        y = tile.sizes["y"] - y - 1
        x = int(x // chip_size)
        y = int(y // chip_size)
        coords.append((x, y))
    return coords


def create_and_save_chips_with_seg_maps(
    m2_tile_dict: dict[str, dict[str, str]],
    df: pd.DataFrame,
    chip_size: int,
    output_directory: str,
    no_data_value: int,
    src_crs: int,
    mask_cloud: bool,
) -> tuple[list[str], list[str | None]]:
    """Chip Creator.

    Create chips and corresponding segmentation maps from a MERRA-2 tile and save them to
    an output directory.

    Args:
        m2_tile_dict (Dict): A dict mapping band names to MERRA-2 tile filepath.
        df (pd.DataFrame): DataFrame containing the data for segmentation maps.
        chip_size (int): Size of each chip.
        output_directory (str): Directory where the chips and segmentation maps will be
            saved.
        no_data_value (int): Value to use for no data areas in the segmentation maps.
        src_crs (int): CRS of points in `df`
        mask_cloud (bool): Perform cloud masking if True.

    Returns:
        A tuple conatinging the lists of created chips and segmentation maps.
    """
    ds, crs = open_mf_tiff_dataset(m2_tile_dict, mask_cloud)
    df = gpd.GeoDataFrame(df, geometry=[Point(xy) for xy in zip(df.x, df.y)])
    df.set_crs(epsg=src_crs, inplace=True)
    df = df.to_crs(crs=crs)

    df = df[
        (ds["x"].min().item() <= df["geometry"].x)
        & (df["geometry"].x <= ds["x"].max().item())
        & (ds["y"].min().item() <= df["geometry"].y)
        & (df["geometry"].y <= ds["y"].max().item())
    ]
    os.makedirs(output_directory, exist_ok=True)
    tile_name_splits = m2_tile_dict["tiles"]["B02_0"].split(".")
    tile_id = f"{tile_name_splits[1]}_{tile_name_splits[2]}_{tile_name_splits[3]}"
    date_id = df.iloc[0]["date"].strftime("%Y%m%d")
    chips = []
    seg_maps: list[str | None] = []
    n_chips_x = ds.sizes["x"] // chip_size
    n_chips_y = ds.sizes["y"] // chip_size
    chip_coords = list(set(get_chip_coords(df, ds, chip_size)))
    for x, y in chip_coords:
        if (x >= n_chips_x) or (y >= n_chips_y):
            continue
        chip_id = f"{date_id}_{tile_id}_{x}_{y}"
        chip_name = f"chip_{chip_id}.tif"
        seg_map_name = f"seg_map_{chip_id}.tif"

        chip_filename = os.path.join(output_directory, "chips", chip_name)
        seg_map_filename = os.path.join(output_directory, "seg_maps", seg_map_name)
        if os.path.exists(chip_filename) or os.path.exists(seg_map_filename):
            continue

        chip = ds.isel(
            x=slice(x * chip_size, (x + 1) * chip_size),
            y=slice(y * chip_size, (y + 1) * chip_size),
        )
        if chip.count().values == 0:
            continue
        seg_map = create_segmentation_map(chip, df, no_data_value)
        if seg_map.where(seg_map != -1).count().values == 0:
            continue
        seg_maps.append(seg_map_name)
        seg_map.rio.to_raster(seg_map_filename)
        chip = chip.fillna(no_data_value)
        chips.append(chip_name)
        chip.band_data.rio.to_raster(chip_filename)
    return chips, seg_maps


def create_m2_dataset(
    data_with_tiles: pd.DataFrame, outdir: str
) -> tuple[dict[str, dict[str, dict[str, str]]], set[str]]:
    """Creates MERRA-2 Dataset.

    A MERRA-2 dataset is list of dictionary mapping band names to corresponding GeoTiff
    filepath. It is required for creating chips.

    Args:
        data_with_tiles (pd.DataFrame): A dataframe containing observations that fall
            within a dense tile. It also has `m2_tiles` column that contains a temporal
            series of MERRA-2 granules.
        outdir (str): Output directory where tiles dould be downloaded to.

    Returns:
        A tuple containing MERRA-2 dataset and a list of tiles that needs to be downloaded.
    """
    data_with_tiles = data_with_tiles.drop_duplicates(subset=["m2_tiles"])
    data_with_tiles = data_with_tiles[
        data_with_tiles["m2_tiles"].apply(
            lambda granule_lst: all("HLS" in str(item) for item in granule_lst)
        )
    ]
    assert not data_with_tiles.empty, "No observation record with valid HLS tiles"
    m2_dataset = {}
    granules_to_download = []
    s30_bands = ["B02", "B03", "B04", "B8A", "B11", "B12", "Fmask"]
    l30_bands = ["B02", "B03", "B04", "B05", "B06", "B07", "Fmask"]
    for m2_tiles, obsv_date in zip(
        data_with_tiles["m2_tiles"], data_with_tiles["date"]
    ):
        band_id, band_path = [], []
        mask_id, mask_path = [], []
        for idx, tile in enumerate(m2_tiles):
            tile = tile.strip(".")
            if "HLS.S30" in tile:
                for band in s30_bands:
                    if band == "Fmask":
                        mask_id.append(f"{band}_{idx}")
                        mask_path.append(
                            os.path.join(outdir, "m2_tiles", f"{tile}.{band}.tif")
                        )
                    else:
                        band_id.append(f"{band}_{idx}")
                        band_path.append(
                            os.path.join(outdir, "m2_tiles", f"{tile}.{band}.tif")
                        )
                    granules_to_download.append(
                        f"https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/HLSS30.020/{tile}/{tile}.{band}.tif"  # noqa
                    )
            else:
                for band in l30_bands:
                    if band == "Fmask":
                        mask_id.append(f"{band}_{idx}")
                        mask_path.append(
                            os.path.join(outdir, "m2_tiles", f"{tile}.{band}.tif")
                        )
                    else:
                        band_id.append(f"{band}_{idx}")
                        band_path.append(
                            os.path.join(outdir, "m2_tiles", f"{tile}.{band}.tif")
                        )
                    granules_to_download.append(
                        f"https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/HLSL30.020/{tile}/{tile}.{band}.tif"  # noqa
                    )

        m2_dataset[f'{obsv_date.strftime("%Y-%m-%d")}_{tile.split(".")[2]}'] = {
            "tiles": {k: v for k, v in zip(band_id, band_path)},
            "fmasks": {k: v for k, v in zip(mask_id, mask_path)},
        }
    return m2_dataset, set(granules_to_download)


def main(argv: Any) -> None:
    """CSV Chip Creator.

    Given a csv file containing geo-located point observations and labels, the Chip
    Creator creates small chip from large HLS tiles which is suitable for training
    segmentation models.
    """
    del argv
    data = pd.read_csv(FLAGS.dataframe_path)
    data["date"] = pd.to_datetime(data["date"]) - pd.offsets.MonthBegin(1)
    data["input_features_date"] = data["date"] - pd.DateOffset(months=1)
    sub_data = m2_utils.get_m2_tiles(data, min_count=FLAGS.min_count)

    if not (
        os.path.exists(os.path.join(FLAGS.output_directory, "m2_dataset.json"))
        and os.path.exists(
            os.path.join(FLAGS.output_directory, "granules_to_download.csv")
        )
    ):
        logging.info("Creating HLS dataset JSON.")
        logging.info("Retrieving HLS tile ID for each observation.")
        sub_data_with_tiles = m2_utils.add_m2_granules(
            sub_data,
            num_steps=FLAGS.num_steps,
            temporal_step=FLAGS.temporal_step,
            temporal_tolerance=FLAGS.temporal_tolerance,
        )
        logging.info("Retrieving HLS tiles that will be downloaded.")
        m2_dataset, granules_to_download = create_m2_dataset(
            sub_data_with_tiles, outdir=FLAGS.output_directory
        )
        with open(
            os.path.join(FLAGS.output_directory, "m2_dataset.json"), "w"
        ) as json_file:
            json.dump(m2_dataset, json_file, indent=4)
        pd.DataFrame({"tiles": list(granules_to_download)}).to_csv(
            os.path.join(FLAGS.output_directory, "granules_to_download.csv")
        )
    else:
        logging.info("HLS dataset JSON already created")
        with open(
            os.path.join(FLAGS.output_directory, "m2_dataset.json")
        ) as json_file:
            m2_dataset = json.load(json_file)
        granules_to_download = pd.read_csv(
            os.path.join(FLAGS.output_directory, "granules_to_download.csv")
        )["tiles"].tolist()
    os.makedirs(os.path.join(FLAGS.output_directory, "m2_tiles"), exist_ok=True)
    logging.info("Downloading HLS Tiles")
    m2_utils.parallel_download(
        granules_to_download,
        outdir=os.path.join(FLAGS.output_directory, "m2_tiles"),
    )
    if FLAGS.download_only:
        return
    logging.info("Creating Chips and Segmentation Maps")
    all_chips = []
    all_seg_maps = []
    os.makedirs(os.path.join(FLAGS.output_directory, "chips"), exist_ok=True)
    os.makedirs(os.path.join(FLAGS.output_directory, "seg_maps"), exist_ok=True)
    for key, m2_tile_dict in tqdm(m2_dataset.items(), desc="Processing HLS Dataset"):
        obsv_date_str, tile_id = key.split("_")
        obsv_data = sub_data[
            (sub_data["date"] == pd.to_datetime(obsv_date_str))
            & (sub_data["mgrs_tile_id"].str.contains(tile_id.strip("T")))
        ]
        try:
            chips, seg_maps = create_and_save_chips_with_seg_maps(
                m2_tile_dict,
                obsv_data,
                chip_size=FLAGS.chip_size,
                output_directory=FLAGS.output_directory,
                no_data_value=FLAGS.no_data_value,
                src_crs=FLAGS.src_crs,
                mask_cloud=FLAGS.mask_cloud,
            )
            all_chips.extend(chips)
            all_seg_maps.extend(seg_maps)
        except rasterio.errors.RasterioIOError as e:
            logging.error(f"Error {e} when reading dataset containing: {m2_tile_dict}")
        except IndexError as e:
            logging.error(f"Error {e} when processing {key}")
    logging.info("Saving dataframe of chips and segmentation maps.")
    pd.DataFrame({"Input": all_chips, "Label": all_seg_maps}).to_csv(
        os.path.join(FLAGS.output_directory, "m2_chips_dataset.csv")
    )


if __name__ == "__main__":
    app.run(main)