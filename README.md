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

#### CITATIONS
>fedesoriano. (January 2022). Stellar Classification Dataset - SDSS17. 

>Data originally from the Sloan Digital Sky Survey (SDSS), [Data Release 17](https://www.sdss.org/dr17/).

# A Small Explanation
To determine whether a system is a star, galaxy or a quasar using the SDSS parameters, the photometric colors (u, g, r, i, z) and the redshift are the most critical factors. 

- **Photometric Magnitudes (u, g, r, i, z)** 

	These define the object's colour profile.
	- **Stars** : Have specific color patterns (e.g., hotter stars appear bluer, cooler stars redder). For example, a star with u−g≈0 might be a blue star, while r−i>0.5 could indicate a red star.
	- **Galaxies** : Tend to be redder (dominated by older stellar populations) or have complex color profiles depending on star formation activity. For instance, a galaxy with g−r>0.7 might be a red elliptical, while i−z<0.3 could indicate a star-forming spiral.
	- **Quasars** : Often exhibit unusual colors due to emission lines (e.g., excess blue light from accretion disks). A quasar might show u−g≈−0.5 (very blue) or r−i>1.0 (unusual redness from dust reddening).
	- **Key Colour Indices** : u−g, g−r, r−i, i−z are commonly used to build colour-colour diagrams for classification.

- **Redshift(z)** 

	The Redshift indicates the distance and nature of the object.
	- **Stars** : Have negligible redshift (z ≈ 0) because they are within the Milky Way.
	- **Galaxies** : Span a wide range of redshifts (z = 0.01 to 2.0+), with higher z indicating more distant galaxies.
	- **Quasars** : Typically have high redshifts (z > 0.5), often exceeding z = 2.0, due to their extreme luminosity and distance.

The other parameters, like observational metadata (run_ID, rerun_ID, cam_col, field_ID), Spectroscopic ID (spec_obj_ID) and the Right ascension (α) and Declination (δ) do not directly influence the classification.  

#### Summary
| Parameter | Role in Classification | Example Use Case |
| :-- | :-- | :-- |
| **u, g, r, i, z** | Defines color profile to distinguish stars, galaxies, and quasars. | A quasar with u - g = -0.6, r - i = 1.2 suggests blue excess and dust reddening. |
| **Redshift (z)** | Indicates distance and type (stars: z≈0; galaxies/quasars: z>0.1). | A z=2.5 object is likely a quasar, while z=0.01 is a nearby galaxy. |
| **spec_obj_ID** | Ensures cross-matching between photometric and spectroscopic data. | Same ID confirms a galaxy’s redshift and emission-line features. |
| **run/rerun** | Affects photometric accuracy but not intrinsic type. | Poor run conditions may blur colours but won’t misclassify a star as a quasar. |
| **α/δ** | Celestial coordinates that specify the position of an astronomical object in the sky. | Alpha and delta themselves do not determine whether an object is a star, galaxy, or quasar. They only provide the object's location on the celestial sphere. |


# Installation

1. Clone the repository:

```bash
git clone https://github.com/sayanjit082805/Stellar-Classification.git
cd Stellar-Classification
```

2. Create a virtual environment (recommended):

```bash
python -m venv stellar_classification_env
source stellar_classification_env/bin/activate  # On Windows: stellar_classification_env\Scripts\activate
```

3. Install required dependencies:

```bash
pip install -r requirements.txt
```


# Neural Network Architecture
The ANN consists of :

- **Input Layer** : 6 neurons (matching the number of input features after scaling)
- **Hidden Layers** : 2 hidden layers with the RelU activation function, with 12 and 11 neurons, respectively. 
- **Dropout Layers** : 2 dropout layers (after each of the hidden layers), to prevent overfitting. 
- **Output Layer** : 3 neurons, with softmax activation function for multi-class classification.

The optimiser used is the [Adam](https://keras.io/api/optimizers/adam/) optimiser, and the loss function is the sparse categorical cross-entropy. 

# Metrics
The model achieves an overall accuracy of ~97%. For more detailed metrics, please consult the classification report in the jupyter notebook.  

Due to the use of Early fitting, the model convergences at 78 epochs.

# License 
This project is licensed under The Unlicense License, see the LICENSE file for details. 

>[!NOTE]
> The License does not cover the dataset. It has been taken from Kaggle and is sourced from SDSS and is in the public domain; please see the SDSS data policy for details.