# Stellar-Classification
A deep learning project to classify celestial objects (stars, galaxies, and quasars) based on their spectral characteristics using an Artificial Neural Network (ANN) trained on data from the Sloan Digital Sky Survey (SDSS).

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)


# Overview 
This project aims to classify astronomical objects into three main categories :

- **Stars** : A star is a massive, luminous spheroid of plasma held together by its own gravity. Stars are fundamental building blocks of galaxies and can exist alone, in pairs, or as part of larger groups like clusters
- **Galaxies** : A galaxy is a vast system composed of stars, stellar remnants, interstellar gas, dust, and dark matter, all bound together by gravity.
- **Quasars** : A quasar is an extremely luminous active galactic nucleus (AGN), powered by a supermassive black hole at the center of a distant galaxy.

The classification is based on spectral characteristics and photometric data.

# Dataset
The dataset has been taken from [Kaggle](https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17). It consists of 100,000 observations of space taken by the SDSS (Sloan Digital Sky Survey).

- `obj_ID` : Object Identifier, the unique value that identifies the object in the image catalog used by the CAS. 
- `alpha` : Right Ascension angle (at J2000 epoch).
- delta :  Declination angle (at J2000 epoch).
- `u` : Ultraviolet filter in the photometric system.
- `g` : Green filter in the photometric system.
- `r` : Red filter in the photometric system.
- `i` : Near Infrared filter in the photometric system.
- `z` : Infrared filter in the photometric system.
- `run_ID` : Run Number used to identify the specific scan.
- `rerun_ID` : Rerun Number to specify how the image was processed.
- `cam_col` : Camera column to identify the scanline within the run.
- `field_ID` : Field number to identify each field.
- `spec_obj_ID` : Unique ID used for optical spectroscopic objects. 
- `class` : object class (galaxy, star or quasar object).
- `redshift` : redshift value based on the increase in wavelength.
- `plate` : plate ID, identifies each plate in SDSS.
- `MJD` : Modified Julian Date, used to indicate when a given piece of SDSS data was taken.
- `fiber_ID` : fiber ID that identifies the fiber that pointed the light at the focal plane in each observation.

>[!IMPORTANT] 
>fedesoriano. (January 2022). Stellar Classification Dataset - SDSS17. 

>Data originally from the Sloan Digital Sky Survey (SDSS), [Data Release 17](https://www.sdss.org/dr17/).


