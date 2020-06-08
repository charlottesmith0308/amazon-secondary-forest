# The Amazon 2.0
Mapping secondary forest in the Amazon

## Description
Creating maps of secondary forest in the Brazilian Amazon using the Mapbiomas 3.1 land cover dataset.


## Scope
The following document outlines the steps taken by Smith et. al to produce maps of secondary forest in the Brazilian Amazon from 1986 - 2017 using the Mapbiomas 3.1 land cover dataset. Scripts were written for use with the SGE based Lancaster Univeristy HEC system and have not been modified for wider use. 

For each pixel scripts also calculate: 
- age 
- years as cropland or pasture
- number of clearance events
- years since last clearance event
- years since first clearance. 


## Setup / Usage / How To

### Required Packages
Packages were managed using Anaconda and were installed with conda or conda-forge.

- cv2
- gdal
- glob
- numpy
- os
- pandas
- sys


### Running Scripts
1. Upload raw mapbiomas data using winscp or equivalent

2. Split big arrays
```
qsub data_wrangling/split_big_arrays_launch.txt
```

3. Change Detection
```
qsub change_detection/change_detection_launch.txt
qsub -t 17-19:2 -l h_vmem=50G change_detection/change_detection_launch.txt
qsub -t 18 -l h_vmem=60G change_detection/change_detection_launch.txt
```

4. Rejoin Large States
```
qsub data_wrangling/rejoin_arrays_launch.txt
```

## Contribute
If you have any suggestions on how to improve the efficiency of this project please get in contact!

## Credit
Everything within this repo was written by myself (Charlotte C. Smith) and forms the basis of my PhD research. If you are making use of any of the code or methodology, please cite Smith et. al 2020.

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons Licence" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br /><span xmlns:dct="http://purl.org/dc/terms/" property="dct:title">The Amazon 2.0</span> by <span xmlns:cc="http://creativecommons.org/ns#" property="cc:attributionName">Charlotte C. Smith</span> is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.<br />Based on a work at <a xmlns:dct="http://purl.org/dc/terms/" href="https://github.com/charlottesmith0308/amazon-secondary-forest.git" rel="dct:source">https://github.com/charlottesmith0308/amazon-secondary-forest.git</a>.
