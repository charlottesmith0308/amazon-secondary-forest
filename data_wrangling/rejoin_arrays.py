# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 15:12:25 2019

@author: Charlotte Smith
@contact: c.smith25@lancaster.ac.uk

@summary: Rejoin states which were split for processing
@version: 1.0
@updated: 04/05/2019
"""


import gdal
import numpy as np
import os
import pandas as pd

from glob import glob
from sys import argv

# %%
TASK_ID = os.getenv("SGE_TASK_ID")

year = str(TASK_ID)
print(year)

base_path = INSERT_ROOT_PATH

# %%

#########################################
### Functions for saving output tiffs ###
#########################################
def get_geotransform(region):
    """ Get geotransformation data from original files to apply when saving new tiff files.
    :region: name of region to get georeference data for
    """
    base_image = glob(classified + '*1985.tif') # Get path to image with georeference metadata
    if not base_image:
        base_image = glob(classified + '1985*classified.tif')
    if not base_image:
        base_image = glob(classified + '1985*raw_crop.tif')
    base = gdal.Open(base_image[0])
    geo_t = base.GetGeoTransform()
    proj = base.GetProjection()

    geo_ref = [geo_t, proj]
    base = None

    return geo_ref


def save_to_tif(image, out_path, geo_ref, bands=1):
    """ Save numpy array to LZW compressed tiff file
    :image: numpy array to be saved
    :out_path: destination path for output file
    """
    driver = gdal.GetDriverByName('Gtiff')

    if bands == 1:
        im = driver.Create(out_path, image.shape[1], image.shape[0], bands, gdal.GDT_Byte , options = ['COMPRESS=LZW'])
        im.SetGeoTransform(geo_ref[0])
        im.SetProjection(geo_ref[1])
        im.GetRasterBand(1).WriteArray(image)
    else:
        im = driver.Create(out_path, image.shape[2], image.shape[1], bands, gdal.GDT_Byte , options = ['COMPRESS=LZW'])
        im.SetGeoTransform(geo_ref[0])
        im.SetProjection(geo_ref[1])
        for i in range(bands):
            im.GetRasterBand(i+1).WriteArray(image[i])
    im.FlushCache()
    im = None


#####################
### Rejoin arrays ###
#####################
regions = ['amazonas', 'para']    
areas = ['NW', 'NC', 'NE', 'SW', 'SC', 'SE']  
map_types = ['A', 'CE', 'CEA', 'CT', 'AG', 'P', 'FD']

for region in regions:
    print(region)
    classified = "{path}{region}/classified/".format(path=base_path, region=region)
    reclassified = "{path}{region}/reclassified/".format(path=base_path, region=region)
    transitions = "{path}{region}/transitions/".format(path=base_path, region=region)

    geo_ref = get_geotransform(region=region)

    for map_ in map_types:
        print(map_)
        # Create name for output tif
        output_file = "{path}{year}_{region}_{map_type}.tif".format(path=transitions, year=year, region=region, map_type=map_)
        
        # Get tifs to be joined
        images = glob("{path}{region}??/transitions/{year}*_{map_type}.tif".format(path=base_path, region=region, year=year, map_type=map_))
        images = sorted(images)

        print(images)
        print(len(images))

        # Open northern chunks
        imNC = gdal.Open(images[0]).ReadAsArray()
        imNE = gdal.Open(images[1]).ReadAsArray()
        imNW = gdal.Open(images[2]).ReadAsArray()

        # Join northern chunks
        north = np.concatenate((imNW, imNC, imNE), axis=1)

        imNW = None
        imNC = None
        imNE = None

        # Open southern chunks
        imSC = gdal.Open(images[3]).ReadAsArray()
        imSE = gdal.Open(images[4]).ReadAsArray()
        imSW = gdal.Open(images[5]).ReadAsArray()

        # Join southern chunks
        south = np.concatenate((imSW, imSC, imSE), axis=1)

        imSW = None
        imSC = None
        imSE = None

        # Join north and south chunks into complete image
        final = np.concatenate((north, south), axis=0)

        save_to_tif(image=final, out_path=output_file, geo_ref=geo_ref)

        final = None
