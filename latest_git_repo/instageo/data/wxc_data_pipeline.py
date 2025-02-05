#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
# Ce code est sous licence CC BY-NC-SA 4.0
#
# Vous êtes libre de :
# - Partager : copier et redistribuer le matériel dans n'importe quel format
# - Adapter : remixer, transformer et construire sur ce matériel
#
# Sous les conditions suivantes :
# - Attribution : Vous devez mentionner le crédit approprié et fournir un lien vers la licence.
# - NonCommercial : Vous ne pouvez pas utiliser le matériel à des fins commerciales.
# - ShareAlike : Si vous modifiez ou transformez le matériel, vous devez distribuer vos contributions
#   sous la même licence que l'original.
#
# Pour plus de détails, voir https://creativecommons.org/licenses/by-nc-sa/4.0/
# ------------------------------------------------------------------------------

"""
Module de pipeline de segmentation pour données .nc4.
Ce module lit un fichier .nc4 (contenant par exemple des données météo/climat sous forme de tenseur),
découpe l'ensemble spatial en "chips" (extraits de taille fixe) et génère pour chacun une carte
de segmentation en fonction d'observations fournies dans un DataFrame.
"""

import os
import logging
from functools import partial
from typing import Any, Callable

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from shapely.geometry import box
from pyproj import Transformer
import rioxarray  # permet de sauvegarder les DataArrays en raster (GeoTIFF)

# Valeur de "no data" pour les segmentation maps
NO_DATA_SEG_MAP = -9999

def data_reader_nc4(nc4_path: str, load_mask: bool = False) -> tuple[xr.DataArray, xr.DataArray, str]:
    """
    Lit un fichier .nc4 et renvoie :
      - data_array : le tenseur (xarray.DataArray) contenant les données (par défaut la première variable trouvée),
      - mask_array : un tenseur de masque (None ici, à adapter si vos .nc4 incluent un masque),
      - crs : le système de coordonnées (ici supposé EPSG:4326).
    """
    ds = xr.open_dataset(nc4_path)
    # Choix de la variable d'intérêt : ici on prend la première variable trouvée
    if ds.data_vars:
        data_array = list(ds.data_vars.values())[0]
    else:
        raise ValueError("Aucune variable n'a été trouvée dans le fichier .nc4")
    mask_array = None  # Adapter si un masque est présent dans le fichier
    crs = "EPSG:4326"  # Supposé
    # Assurez-vous que le DataArray possède un CRS via rioxarray
    data_array = data_array.rio.write_crs(crs)
    return data_array, mask_array, crs

def apply_mask_nc4(chip: xr.DataArray,
                   mask: xr.DataArray,
                   no_data_value: int = NO_DATA_SEG_MAP) -> xr.DataArray:
    """
    Applique un masque à un chip.
    Si un masque (binaire, avec 0 pour les pixels valides et 1 pour masqués) est fourni,
    on remplace les pixels masqués par no_data_value.
    """
    if mask is None:
        return chip
    chip_masked = chip.where(mask == 0, other=no_data_value)
    return chip_masked

def create_segmentation_map_nc4(chip: xr.DataArray,
                                df: pd.DataFrame,
                                window_size: int) -> xr.DataArray:
    """
    Crée une carte de segmentation pour un chip.
    On part d'un chip (DataArray) et d'un DataFrame d'observations (colonnes 'x', 'y', 'label').
    Pour chaque observation présente dans l'étendue du chip, on affecte dans une fenêtre (définie par window_size)
    la valeur du label correspondant.
    """
    # Créer une segmentation map de même forme que le premier canal du chip
    seg_map = xr.full_like(chip.isel({list(chip.dims)[-1]: 0}), fill_value=NO_DATA_SEG_MAP, dtype=np.int16)
    
    # Convertir le DataFrame en GeoDataFrame
    df_gpd = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.x, df.y))
    # On suppose que df est dans le même CRS que chip
    xmin, ymin, xmax, ymax = chip.rio.bounds()
    df_filtered = df_gpd[
        (df_gpd.geometry.x >= xmin) &
        (df_gpd.geometry.x <= xmax) &
        (df_gpd.geometry.y >= ymin) &
        (df_gpd.geometry.y <= ymax)
    ]
    
    # Récupérer la transformation affine du chip
    transform = chip.rio.transform()
    # Pour chaque observation, convertir les coordonnées en indices de pixel
    for idx, row in df_filtered.iterrows():
        x_coord, y_coord = row.geometry.x, row.geometry.y
        # Calcul des indices (arrondis à l'entier inférieur)
        x_idx, y_idx = ~transform * (x_coord, y_coord)
        x_idx = int(np.floor(x_idx))
        y_idx = int(np.floor(y_idx))
        # Définir une fenêtre autour du point
        offsets = np.arange(-window_size, window_size + 1)
        off_x, off_y = np.meshgrid(offsets, offsets)
        win_x = np.clip(x_idx + off_x, 0, chip.sizes['x'] - 1)
        win_y = np.clip(y_idx + off_y, 0, chip.sizes['y'] - 1)
        # Affecter la valeur de label dans la segmentation map
        seg_map.values[win_x, win_y] = row['label']
    return seg_map

def get_chip_coords_nc4(gdf: gpd.GeoDataFrame,
                        tile: xr.DataArray,
                        chip_size: int) -> np.array:
    """
    Calcule et renvoie les indices uniques (sous forme d'un tableau [i, j]) permettant d'extraire des chips
    de taille chip_size à partir du tile (DataArray).
    """
    transform = tile.rio.transform()
    # Convertir chaque point de gdf en indice (x, y)
    coords = np.array([~transform * (pt.x, pt.y) for pt in gdf.geometry])
    coords = np.floor(coords).astype(int)
    # Déduire les indices de chips (division entière)
    chip_indices = np.unique((coords // chip_size), axis=0)
    return chip_indices

def create_and_save_chips_with_seg_maps_nc4(
    nc4_path: str,
    df: pd.DataFrame,
    chip_size: int,
    output_directory: str,
    no_data_value: int = NO_DATA_SEG_MAP,
    window_size: int = 2
) -> tuple[list[str], list[str]]:
    """
    Crée des chips et leurs cartes de segmentation à partir d'un fichier .nc4 et d'un DataFrame d'observations.
    Les chips et segmentation maps sont enregistrés dans des sous-dossiers 'chips' et 'seg_maps' du répertoire
    de sortie.

    Args:
        nc4_path (str): Chemin vers le fichier .nc4.
        df (pd.DataFrame): DataFrame contenant les observations (colonnes 'x', 'y', 'label', éventuellement 'date').
        chip_size (int): Taille en pixels de chaque chip.
        output_directory (str): Répertoire où enregistrer les chips et segmentation maps.
        no_data_value (int): Valeur à utiliser pour les pixels sans données.
        window_size (int): Taille de la fenêtre (en pixels) autour de chaque observation pour affecter le label.

    Returns:
        Tuple de listes contenant les noms des fichiers chips et segmentation maps créés.
    """
    # Lecture des données .nc4
    ds, mask_ds, crs = data_reader_nc4(nc4_path, load_mask=False)
    
    # On suppose que les dimensions spatiales du DataArray sont 'lon' et 'lat'
    # On renomme ces dimensions en 'x' et 'y' pour être cohérent avec le reste du code.
    if "lon" in ds.dims and "lat" in ds.dims:
        ds = ds.rename({'lon': 'x', 'lat': 'y'})
    ds = ds.rio.write_crs(crs)
    
    # Convertir le DataFrame d'observations en GeoDataFrame et filtrer selon l'étendue du tile
    xmin, ymin, xmax, ymax = ds.rio.bounds()
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.x, df.y))
    gdf = gdf.set_crs(crs)
    gdf = gdf[(gdf.geometry.x >= xmin) & (gdf.geometry.x <= xmax) &
              (gdf.geometry.y >= ymin) & (gdf.geometry.y <= ymax)]
    
    # Création des répertoires de sortie
    chips_dir = os.path.join(output_directory, "chips")
    seg_maps_dir = os.path.join(output_directory, "seg_maps")
    os.makedirs(chips_dir, exist_ok=True)
    os.makedirs(seg_maps_dir, exist_ok=True)
    
    # Nombre de chips possibles selon la taille du tile
    n_chips_x = ds.sizes["x"] // chip_size
    n_chips_y = ds.sizes["y"] // chip_size
    
    # Obtenir les indices de chips à partir des observations
    chip_coords = get_chip_coords_nc4(gdf, ds, chip_size)
    
    chips = []
    seg_maps = []
    
    for coord in chip_coords:
        i, j = coord  # indices du chip dans les directions x et y
        if (i >= n_chips_x) or (j >= n_chips_y):
            continue
        chip_id = f"{i}_{j}"
        chip_name = f"chip_{chip_id}.tif"
        seg_map_name = f"seg_map_{chip_id}.tif"
        chip_filepath = os.path.join(chips_dir, chip_name)
        seg_map_filepath = os.path.join(seg_maps_dir, seg_map_name)
        # Si les fichiers existent déjà, on passe
        if os.path.exists(chip_filepath) or os.path.exists(seg_map_filepath):
            continue
        # Extraire le chip par découpage
        chip = ds.isel(
            x=slice(i * chip_size, (i + 1) * chip_size),
            y=slice(j * chip_size, (j + 1) * chip_size)
        ).compute()
        # Appliquer le masque si disponible
        if mask_ds is not None:
            mask_chip = mask_ds.isel(
                x=slice(i * chip_size, (i + 1) * chip_size),
                y=slice(j * chip_size, (j + 1) * chip_size)
            ).compute()
            chip = apply_mask_nc4(chip, mask_chip, no_data_value)
        # Créer la segmentation map
        seg_map = create_segmentation_map_nc4(chip, gdf, window_size)
        # Vérifier que le chip et la segmentation map contiennent des données valides
        if (chip.where(chip != no_data_value).count().values == 0) or (seg_map.where(seg_map != no_data_value).count().values == 0):
            continue
        # Sauvegarder en GeoTIFF
        chip.rio.to_raster(chip_filepath)
        seg_map.rio.to_raster(seg_map_filepath)
        chips.append(chip_name)
        seg_maps.append(seg_map_name)
        
    return chips, seg_maps

# Exemple d'utilisation
if __name__ == "__main__":
    # Exemple de DataFrame d'observations.
    # Il doit contenir au moins les colonnes 'x', 'y' et 'label'. La colonne 'date' est optionnelle.
    df_obs = pd.DataFrame({
        "x": [10.0, 20.0, 30.0],
        "y": [50.0, 55.0, 60.0],
        "label": [1, 2, 3],
        "date": pd.to_datetime(["2023-01-01", "2023-01-01", "2023-01-01"])
    })
    
    # Chemin vers votre fichier .nc4
    nc4_file_path = "chemin/vers/votre_fichier.nc4"
    # Répertoire de sortie
    output_dir = "output"
    # Taille des chips en pixels
    chip_size = 256
    # Taille de la fenêtre autour de chaque point (pour affecter le label)
    window_size = 3
    
    chips_created, seg_maps_created = create_and_save_chips_with_seg_maps_nc4(
        nc4_path=nc4_file_path,
        df=df_obs,
        chip_size=chip_size,
        output_directory=output_dir,
        no_data_value=NO_DATA_SEG_MAP,
        window_size=window_size
    )
    
    print("Chips créés :", chips_created)
    print("Segmentation maps créées :", seg_maps_created)
