#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 09:09:43 2018

@author: Charlotte Smith
@contact: c.smith25@lancaster.ac.uk

@summary: Split large arrays into smaller sections
@version: 1.0
@updated: 02/05/2019
"""

#from tifffile import imsave, imread
import numpy as np
import pandas as pd
import os
import gdal

# %%
TASK_ID = os.getenv("SGE_TASK_ID")
driver = gdal.GetDriverByName('Gtiff')

year = str(TASK_ID)
regions = ['para', 'amazonas']

# %%
# Split to 6
for region in regions:
    base_path = INSERT_ROOT_PATH
    classified = "{path}{region}/classified/".format(path=base_path, region=region)

    # %%
    # Get list of mapbiomas image names
    images = []                             # Empty list for storing image names
    for i in os.listdir(classified):        # Extract tif file names
        if ".tif" in i and i[0] != '.':     # Ensure only tif files are selected
            images.append(i)                # Add image file name to list

    # Paths for saving divided tiffs too
    nw_path = "{path}{region}{area}/classified/{year}_{region}{area}_classified.tif".format(path=base_path, region=region, area='NW', year=year)
    ne_path = "{path}{region}{area}/classified/{year}_{region}{area}_classified.tif".format(path=base_path, region=region, area='NE', year=year)
    nc_path = "{path}{region}{area}/classified/{year}_{region}{area}_classified.tif".format(path=base_path, region=region, area='NC', year=year)
    sw_path = "{path}{region}{area}/classified/{year}_{region}{area}_classified.tif".format(path=base_path, region=region, area='SW', year=year)
    se_path = "{path}{region}{area}/classified/{year}_{region}{area}_classified.tif".format(path=base_path, region=region, area='SE', year=year)
    sc_path = "{path}{region}{area}/classified/{year}_{region}{area}_classified.tif".format(path=base_path, region=region, area='SC', year=year)

    for image in images:
        if year in image:
            im = gdal.Open(classified + image).ReadAsArray()    # read tiff to split
            im = im.astype('int8')

            arrays = np.array_split(im, 2)  # Split array horizontally
            im = None
            north = arrays[0]
            south = arrays[1]

            north_split = np.array_split(north, 3, axis=1)  # Split the northern half into 3 chunks
            north = None
            north_west = north_split[0]
            north_centre = north_split[1]
            north_east = north_split[2]

            # Save chunks to new tiffs
            im = driver.Create(nw_path, north_west.shape[1], north_west.shape[0], 1, gdal.GDT_Byte , options = ['COMPRESS=LZW'])
            im.GetRasterBand(1).WriteArray(north_west)
            im.FlushCache()
            im = driver.Create(nc_path, north_centre.shape[1], north_centre.shape[0], 1, gdal.GDT_Byte , options = ['COMPRESS=LZW'])
            im.GetRasterBand(1).WriteArray(north_centre)
            im.FlushCache()
            im = driver.Create(ne_path, north_east.shape[1], north_east.shape[0], 1, gdal.GDT_Byte , options = ['COMPRESS=LZW'])
            im.GetRasterBand(1).WriteArray(north_east)
            im.FlushCache()

            north_west = None
            north_east = None

            south_split = np.array_split(south, 3, axis=1)   # Split the southern half into 3 chunks
            south = None
            south_west = south_split[0]
            south_centre = south_split[1]
            south_east = south_split[2]

            # Save chunks to new tiffs
            im = driver.Create(sw_path, south_west.shape[1], south_west.shape[0], 1, gdal.GDT_Byte , options = ['COMPRESS=LZW'])
            im.GetRasterBand(1).WriteArray(south_west)
            im.FlushCache()
            im = driver.Create(sc_path, south_centre.shape[1], south_centre.shape[0], 1, gdal.GDT_Byte , options = ['COMPRESS=LZW'])
            im.GetRasterBand(1).WriteArray(south_centre)
            im.FlushCache()
            im = driver.Create(se_path, south_east.shape[1], south_east.shape[0], 1, gdal.GDT_Byte , options = ['COMPRESS=LZW'])
            im.GetRasterBand(1).WriteArray(south_east)
            im.FlushCache()

            south_west = None
            south_east = None
