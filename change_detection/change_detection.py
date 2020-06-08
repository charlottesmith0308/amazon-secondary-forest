#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 10:41:18 2018

@author: Charlotte Smith
@contact: c.smith25@lancaster.ac.uk

@summary: Change detection
@version: 1.0
@updated: 04/05/2019
"""
import gdal
import cv2
import numpy as np
import pandas as pd
import os

from glob import glob


# %%
#####################
### Set Variables ###
#####################
TASK_ID = os.getenv("SGE_TASK_ID")



REGIONS = {'1':'acre',
               '2':'amapa',
               '3':'amazonasNC',
               '4':'amazonasNE',
               '5':'amazonasNW',
               '6':'amazonasSC',
               '7':'amazonasSE',
               '8':'amazonasSW',
               '9':'roraima',
               '10':'paraNC',
               '11':'paraNE',
               '12':'paraNW',
               '13':'paraSC',
               '14':'paraSE',
               '15':'paraSW',
               '16':'rondonia',
               '17':'maranhao',
               '18':'matogrosso',
               '19':'tocantins'}


region = REGIONS[TASK_ID]

print(region)

first_year = 1985
last_year = 2017

# %%
cropped = ['matogrosso', 'tocantins', 'maranhao'] # list of cropped states to enable special handling

# %%
#####################
### Setting paths ###
#####################
base_path = INSERT_ROOT_PATH_HERE + region +'/'
classified = base_path + "classified/"
reclassified = base_path + "reclassified/"
transitions = base_path + "transitions/"

if region in cropped:
    classified = base_path + "raw_crop/"


# %%
#########################################
### Functions for saving output tiffs ###
#########################################
def get_geotransform(region=region):
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

geo_ref = get_geotransform()


def save_to_tif(image, out_path, bands=1):
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


# %%
#########################
#### Reclassification ###
#########################

#-------------------------#
# Make file names uniform #
#-------------------------#
for year in range(1985, 2018):
    year = str(year)
    name = glob(classified + '*' + year + ".tif")
    if region in cropped:
        name = glob(classified + year + '*' + ".tif")
    print(name)
    if name:
        im = gdal.Open(name[0]).ReadAsArray()
        save_to_tif(im, classified + year + '_' + region + '_classified.tif')

# %%
#-------------------#
# Reclassify Images #
#-------------------#
# Reclassification groups - lists of mapbiomas pixel IDs
forest = [1.0, 2.0, 3.0, 4.0, 5.0]
agriculture = [9.0, 14.0, 18.0, 19.0, 20.0, 21.0]
pasture = [15.0]
other = [10.0, 11.0, 12.0, 13.0, 22.0, 23.0, 24.0, 25.0, 29.0, 30.0, 32.0]
water = [26.0, 31.0, 33.0]
na = [0.0, 27.0, -128, 128]

def reclassify_image(image):
    """ Reassign pixel values based on reclassification groups
    :image: image file name
    """
    im = gdal.Open(classified + image).ReadAsArray()
    im = im.astype('int8')

    for i in na:
        im[im == i] = -99
    for i in forest:
        im[im == i] = 3
    for i in water:
        im[im == i] = 0
    for i in other:
        im[im == i] = 0
    for i in agriculture:
        im[im == i] = 1
    for i in pasture:
        im[im == i] = 2

    im = im**2  # Square values (ESSENTIAL for later processing)
    im[im==73] = -99  # Reset NA values after squaring
    im = im.astype('int8')

    return im


# Get list of mapbiomas image names
images = []
for i in os.listdir(classified):
    if ".tif" in i and i[0] != '.' and 'classified' in i:
        images.append(i)

# Run the function reclassify_image() for each year and save the output
for i in range(first_year,last_year+1):
    year = str(i)
    for image in images:
        if year in image:
            print('reclassify: ' + str(year))
            name = reclassified + year + "_" + region + "_reclassified.tif"  # Create output file name in format YEAR_REGION_reclassified.tif
            save_to_tif(reclassify_image(image), name)

# %%
#------------------------------------------#
# Remove water errors and apply water mask #
#------------------------------------------#
def remove_water_errors(year, final_mask, na_mask):
    """ Corrects pixels which have been misclassified as water

    :year: year to process
    """
    # Setting values for year variables
    year1 = str(year)
    year2 = str(year+1)
    year3 = str(year+2)

    print('water errors: ' + year1)

    # Setting path names for files
    y1_name = reclassified + year1 + "_" + region + "_reclassified.tif"
    y2_name = reclassified + year2 + "_" + region + "_reclassified.tif"
    y3_name = reclassified + year3 + "_" + region + "_reclassified.tif"

    # Reading image files to arrays
    y1 = gdal.Open(y1_name).ReadAsArray()
    y1 = y1.astype('int8')
    y2 = gdal.Open(y2_name).ReadAsArray()
    y2 = y2.astype('int8')
    y3 = gdal.Open(y3_name).ReadAsArray()
    y3 = y3.astype('int8')

    # Create empty final water and na masks on first run
    if year == 1985:
        final_mask = np.zeros((y2.shape[0], y2.shape[1]), dtype=bool)
        na_mask = np.zeros((y2.shape[0], y2.shape[1]), dtype=bool)

    # Removing random water pixels from middle year
    y1_mask = y1!=0  # Create mask of non-water pixels
    y2_mask = y2==0  # Create mask of water pixels
    y3_mask = y3!=0  # Create mask of non-water pixels

    water_mask = y1_mask & y2_mask & y3_mask  # Combine masks

    y2[water_mask] = y1[water_mask]  # Where mask, fill y2 with y1 values

    # Make water if is water before and after
    water_mask = ~y1_mask & ~y3_mask  # Where y1 and y3 water, fill y2 as water
    y2[water_mask] = 0
    save_to_tif(y2, y2_name)

    # Remove random water from first pixel if it is the first year
    if year == first_year:
        y1_mask = y1==0                     # Create mask of water pixels
        y2_mask = y2!=0                     # Create mask of non-water pixels
        y3_mask = y3!=0                     # Create mask of non-water pixels

        water_mask = y1_mask & y2_mask & y3_mask   # Combine masks
        y1[water_mask] = y2[water_mask]   # Where mask, fill y1 with y2 values
        save_to_tif(y1, y1_name)

    # Remove random water pixels from last pixel if it is the last year
    if year+2 == last_year:
        y1_mask = y1!=0                     # Create mask of non-water pixels
        y2_mask = y2!=0                     # Create mask of non-water pixels
        y3_mask = y3==0                     # Create mask of water pixels

        water_mask = y1_mask & y2_mask & y3_mask   # Combine masks
        y3[water_mask] = y2[water_mask]  # Where mask, fill y3 with y2 values
        save_to_tif(y3, y3_name)

    water_mask = y2==0      # create mask of all water pixels in y2

    if year3 == str(last_year):    # If y3 is the last year in the time series
        last_mask = y3==0                       # Mask where y3 is water
        water_mask = last_mask | water_mask     # Add last year water to water mask
        last_mask = None

    final_mask = final_mask | water_mask  # Combine water mask with the cumulative 'final' mask for return
    water_mask = None

    # Create iterative mask of NA values such that the maximum can be applied across all years
    na_mask_y1 = y1==-99
    na_mask_y2 = y2==-99
    na_mask_y3 = y3==-99
    na_mask = na_mask | na_mask_y1 | na_mask_y2 | na_mask_y3
    na_mask_y1 = None
    na_mask_y2 = None
    na_mask_y3 = None

    return final_mask, na_mask

final_mask = None
na_mask = None
for year in range(first_year, last_year-1):
    final_mask, na_mask = remove_water_errors(year, final_mask, na_mask)


#-------------------------------#
# Apply final mask to each year #
#-------------------------------#
def apply_final_masks(year, final_mask, na_mask):
    """ Apply final water and na masks so they are consistent across all years

    :year: year to process
    :final_mask: first output of remove_water_errors
    :na_mask: second output of remove_water_errors
    """
    print(year)
    image = gdal.Open(reclassified + str(year) + "_" + region + "_reclassified.tif").ReadAsArray()
    image = image.astype('int8')
    image[final_mask] = 0
    image[na_mask] = -99
    image = image.astype('int8')

    save_to_tif(image, reclassified + str(year) + "_" + region + "_reclassified.tif") # Save to file


for year in range(first_year,last_year+1):
    apply_final_masks(year, final_mask, na_mask)

# %%
########################
### Change Detection ###
########################
#---------------------------------------------#
# Define first year change detection function #
#---------------------------------------------#
def change_detection_y0(image):
    """ Create dynamics array for first year in timeseries

    :image: path to reclassified image
    """
    im = gdal.Open(image).ReadAsArray()
    im = im.astype('int8')

    # Create dynamics map - empty 7 band array
    dynamics_map = np.zeros((7, im.shape[0], im.shape[1]), dtype=np.int8)
    # band 0 - cover type
    # band 1 - length of agriculture
    # band 2 - length of pasture
    # band 3 - clearance events
    # band 4 - years since clearance event
    # band 5 - age of current land cover
    # band 6 - time since first disturbance

    # Create masks of land cover types
    na = im==-99
    water = im==0
    agriculture = im==1
    pasture = im == 4
    forest = im==9

    # Set values based on landcover type
    # NA
    dynamics_map[0][na] = -99
    dynamics_map[1][na] = -99
    dynamics_map[2][na] = -99
    dynamics_map[3][na] = -99
    dynamics_map[4][na] = -99
    dynamics_map[5][na] = -99
    dynamics_map[6][na] = -99

    # Water
    dynamics_map[1][water] = -99
    dynamics_map[2][water] = -99
    dynamics_map[3][water] = -99
    dynamics_map[4][water] = -99
    dynamics_map[5][water] = -99
    dynamics_map[6][water] = -99

    # Agriculture
    dynamics_map[0][agriculture] = 1
    dynamics_map[1][agriculture] = 1
    dynamics_map[2][agriculture] = 0
    dynamics_map[3][agriculture] = 1
    dynamics_map[4][agriculture] = 1
    dynamics_map[5][agriculture] = 1
    dynamics_map[6][agriculture] = 1

    # Pasture
    dynamics_map[0][pasture] = 2
    dynamics_map[1][pasture] = 0
    dynamics_map[2][pasture] = 1
    dynamics_map[3][pasture] = 1
    dynamics_map[4][pasture] = 1
    dynamics_map[5][pasture] = 1
    dynamics_map[6][pasture] = 1

    # Forest
    dynamics_map[0][forest] = 3
    dynamics_map[1][forest] = 0
    dynamics_map[2][forest] = 0
    dynamics_map[3][forest] = 0
    dynamics_map[4][forest] = -99
    dynamics_map[5][forest] = 1
    dynamics_map[6][forest] = -99

    dynamics_map = dynamics_map.astype('int8')
    name = transitions + str(first_year) + "_" + region + "_dynamics.tif"

    # Save combined file and save each band seperately
    save_to_tif(dynamics_map, name, bands=7)
    save_to_tif(dynamics_map[0], transitions + str(first_year) + '_' + region + '_CT.tif')
    save_to_tif(dynamics_map[1], transitions + str(first_year) + '_' + region + '_AG.tif')
    save_to_tif(dynamics_map[2], transitions + str(first_year) + '_' + region + '_P.tif')
    save_to_tif(dynamics_map[3], transitions + str(first_year) + '_' + region + '_CE.tif')
    save_to_tif(dynamics_map[4], transitions + str(first_year) + '_' + region + '_CEA.tif')
    save_to_tif(dynamics_map[5], transitions + str(first_year) + '_' + region + '_A.tif')
    save_to_tif(dynamics_map[6], transitions + str(first_year) + '_' + region + '_FD.tif')

# %%
#---------------------------------#
# Run first year change detection #
#---------------------------------#
first_year_im_path = reclassified + str(first_year) + "_" + region + "_reclassified.tif"
change_detection_y0(first_year_im_path)

# %%
#---------------------------------------#
# Define main change detection function #
#---------------------------------------#
def change_detection(year, previous_year):
    """ Run change detection algorithm for provided year

    :year: year for current year
    :previous_year: year for previous year
    """

    # Read in images
    c_year = gdal.Open("{path}{year}_{region}_reclassified.tif".format(path=reclassified, year=year, region=region)).ReadAsArray()
    p_year = gdal.Open("{path}{year}_{region}_reclassified.tif".format(path=reclassified, year=previous_year, region=region)).ReadAsArray()

    if year == 1986:
        p_year = gdal.Open("{path}{year}_{region}_reclassified.tif".format(path=reclassified, year=previous_year, region=region)).ReadAsArray()       # Read image for previous year to Numpy Array
    else:
        p_year = gdal.Open("{path}{year}_{region}_CT.tif".format(path=transitions, year=previous_year, region=region)).ReadAsArray()
        p_year = p_year**2
        p_year[p_year==16] = 9
        p_year[p_year==73] = -99

    c_year = c_year.astype('int8')
    p_year = p_year.astype('int8')

    # Apply minimum clearance event size
    ce = np.zeros(c_year.shape)         # Create empty array for storing SF bool
    transition = p_year - c_year        # Calculate transion values
    ce[transition==8] = 1
    ce[transition==5] = 1
    no_change = (transition == 0) & ((c_year == 1)|(c_year == 4))
    ce[no_change] = 1
    transition = None
    no_change = None
    ce = ce.astype(np.uint8)

    ret, patches, stats, centroids = cv2.connectedComponentsWithStats(ce, connectivity=8)
    ret = None
    centroids = None
    ce = None
    patch_data = pd.DataFrame(data=stats, columns=['left', 'top', 'width', 'height', 'area'])
    stats = None
    patch_data.reset_index(inplace=True)
    small_patches = patch_data[patch_data.area<4].index
    patch_data = None
    small_patches_mask = np.isin(patches, small_patches)
    small_patches = None
    c_year[small_patches_mask] = p_year[small_patches_mask]

    correction_counts = np.unique(p_year[small_patches_mask], return_counts=True)
    correction_counts_df = pd.DataFrame(data=[correction_counts[1]], columns=correction_counts[0])
    correction_counts_df.to_csv('{path}{year}_{region}_min_clearance_size_corrections_4.csv'.format(path=transitions, year=year, region=region))

    small_patches_mask = None

    # Claculate transitions
    transition = p_year - c_year
    p_year = None
    c_year = None

    # Get previous year dynamics map
    prev_dynamics = gdal.Open("{path}{year}_{region}_dynamics.tif".format(path=transitions, year=previous_year, region=region)).ReadAsArray()
    prev_dynamics = prev_dynamics.astype('int8')

    # Duplicated previous year dynamics
    dynamics_map = prev_dynamics

    # BANDS
    # band 0 - cover type
    # band 1 - length of agriculture
    # band 2 - length of pasture
    # band 3 - clearance events
    # band 4 - years since clearance event
    # band 5 - age of current land cover

    # TRANSITION VALUES
    # -9    W/O  to F
    # -8    Agri to F
    # -5    Past to F
    # -4    W/O  to Past
    # -3    Agri to Past
    # -1    W/O  to Agri
    #  0     No Change
    #  1    Agri to W/O
    #  3    Past to Agri
    #  4    Past to W/O
    #  5    F    to Past
    #  8    F    to Agri
    #  9    F    to W/O


    # Set Values based on transition value
    # Agriculture to forest
    dynamics_map[0][transition==-8] = 4
   # dynamics_map[1][transition==-8] =
   # dynamics_map[2][transition==-8] =
   # dynamics_map[3][transition==-8] =
    dynamics_map[4][transition==-8] += 1
    dynamics_map[5][transition==-8] = 1
    dynamics_map[6][transition==-8] += 1

    # Pasture to forest
    dynamics_map[0][transition==-5] = 4
   # dynamics_map[1][transition==-5] =
   # dynamics_map[2][transition==-5] =
   # dynamics_map[3][transition==-5] =
    dynamics_map[4][transition==-5] += 1
    dynamics_map[5][transition==-5] = 1
    dynamics_map[6][transition==-5] += 1

    # Agriculture to Pasture
    dynamics_map[0][transition==-3] = 2
   # dynamics_map[1][transition==-3] =
    dynamics_map[2][transition==-3] += 1
   # dynamics_map[3][transition==-3] =
    dynamics_map[4][transition==-3] += 1
    dynamics_map[5][transition==-3] = 1
    dynamics_map[6][transition==-3] += 1

    # No change PF
    pf = dynamics_map[0]==3
    no_change = transition==0
    pf_no_change = pf & no_change
   # dynamics_map[0][pf_no_change] =
   # dynamics_map[1][pf_no_change] =
   # dynamics_map[2][pf_no_change] =
   # dynamics_map[3][pf_no_change] =
   # dynamics_map[4][pf_no_change] =
    dynamics_map[5][pf_no_change] += 1
   # dynamics_map[6][pf_no_change] =
    pf_no_change = None

    # No change SF
    sf = dynamics_map[0]==4
    sf_no_change = sf & no_change
   # dynamics_map[0][sf_no_change] =
   # dynamics_map[1][sf_no_change] =
   # dynamics_map[2][sf_no_change] =
   # dynamics_map[3][sf_no_change] =
    dynamics_map[4][sf_no_change] += 1
    dynamics_map[5][sf_no_change] += 1
    dynamics_map[6][sf_no_change] += 1
    sf_no_change = None

    # No change Agriculture
    ag = dynamics_map[0]==1
    ag_no_change = ag & no_change
    ag = None
   # dynamics_map[0][ag_no_change] =
    dynamics_map[1][ag_no_change] += 1
   # dynamics_map[2][ag_no_change] =
   # dynamics_map[3][ag_no_change] =
    dynamics_map[4][ag_no_change] += 1
    dynamics_map[5][ag_no_change] += 1
    dynamics_map[6][ag_no_change] += 1
    ag_no_change = None

    # No change Pasture
    p = dynamics_map[0]==2
    p_no_change = p & no_change
    p = None
    no_change = None
   # dynamics_map[0][p_no_change] =
   # dynamics_map[1][p_no_change] =
    dynamics_map[2][p_no_change] += 1
   # dynamics_map[3][p_no_change] =
    dynamics_map[4][p_no_change] += 1
    dynamics_map[5][p_no_change] += 1
    dynamics_map[6][p_no_change] += 1
    p_no_change = None

    # Pasture to Agriculture
    dynamics_map[0][transition==3] = 1
    dynamics_map[1][transition==3] += 1
   # dynamics_map[2][transition==3] =
   # dynamics_map[3][transition==3] =
    dynamics_map[4][transition==3] += 1
    dynamics_map[5][transition==3] = 1
    dynamics_map[6][transition==3] += 1

    # Forest to Pasture
    f_to_p = transition==5
    pf_to_p = pf & f_to_p
    sf_to_p = sf & f_to_p
    dynamics_map[0][transition==5] = 2
    dynamics_map[1][transition==5] = 0
    dynamics_map[2][transition==5] = 1
    dynamics_map[3][transition==5] +=1
    dynamics_map[4][transition==5] = 1
    dynamics_map[5][transition==5] = 1
    dynamics_map[6][pf_to_p] = 1
    dynamics_map[6][sf_to_p] += 1
    f_to_p = None
    sf_to_p = None
    pf_to_p = None

    # Forest to Agriculture
    f_to_ag = transition==8
    pf_to_ag = pf & f_to_ag
    sf_to_ag = sf & f_to_ag
    dynamics_map[0][transition==8] = 1
    dynamics_map[1][transition==8] = 1
    dynamics_map[2][transition==8] = 0
    dynamics_map[3][transition==8] +=1
    dynamics_map[4][transition==8] = 1
    dynamics_map[5][transition==8] = 1
    dynamics_map[6][pf_to_ag] = 1
    dynamics_map[6][sf_to_ag] += 1
    f_to_ag = None
    sf_to_ag = None
    pf_to_ag = None

    # Save combined file and save each band seperately

    out_path = "{path}{year}_{region}".format(path=transitions, year=year, region=region)

    save_to_tif(dynamics_map, "{path}_{variable}.tif".format(path=out_path, variable='dynamics'), bands=7)
    save_to_tif(dynamics_map[0], "{path}_{variable}.tif".format(path=out_path, variable='CT'))
    save_to_tif(dynamics_map[1], "{path}_{variable}.tif".format(path=out_path, variable='AG'))
    save_to_tif(dynamics_map[2], "{path}_{variable}.tif".format(path=out_path, variable='P'))
    save_to_tif(dynamics_map[3], "{path}_{variable}.tif".format(path=out_path, variable='CE'))
    save_to_tif(dynamics_map[4], "{path}_{variable}.tif".format(path=out_path, variable='CEA'))
    save_to_tif(dynamics_map[5], "{path}_{variable}.tif".format(path=out_path, variable='A'))
    save_to_tif(dynamics_map[6], "{path}_{variable}.tif".format(path=out_path, variable='FD'))

# %%
#---------------------------#
# Run main change detection #
#---------------------------#
for i in range(first_year+1, last_year+1):
    print('change detection: ' + str(i))
    change_detection(i, i-1)
