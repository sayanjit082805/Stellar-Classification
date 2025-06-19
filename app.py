import streamlit as st
import pandas as pd
from tensorflow.keras.models import load_model
import joblib
import numpy as np

model = load_model("stellar_classification_model.keras")
model_columns = joblib.load("stellar_classification_model_columns.pkl")
scaler = joblib.load("stellar_scaler.pkl")

st.title("Stellar Classifier")

st.subheader(
    "Classify stars, galaxies and quasars, based on spectral characteristics.",
    divider="gray",
)

st.write(
    """
   A deep learning project to classify celestial objects (stars, galaxies, and quasars) based on their spectral characteristics using an Artificial Neural Network (ANN) trained on data from the Sloan Digital Sky Survey (SDSS).     
"""
)

st.sidebar.header("Input Parameters")


def input_features():
    u = st.sidebar.slider(
        "Ultaviolet (u)", min_value=12.0, max_value=25.0, value=20.0, step=0.1
    )
    g = st.sidebar.slider(
        "Green (g)", min_value=12.0, max_value=25.0, value=20.0, step=0.1
    )
    r = st.sidebar.slider(
        "Red (r)", min_value=12.0, max_value=25.0, value=20.0, step=0.1
    )
    i = st.sidebar.slider(
        "Near Infrared (i)", min_value=12.0, max_value=25.0, value=20.0, step=0.1
    )
    z = st.sidebar.slider(
        "Infrared (z)", min_value=12.0, max_value=25.0, value=20.0, step=0.1
    )
    redshift = float(
        st.sidebar.text_input("Redshift", placeholder="0.00004", value="0.00001")
    )
    # features = ['airline', 'stops', 'source_city', 'departure_time', 'arrival_time', 'destination_city', 'class', 'days_left']

    features = pd.DataFrame(
        [
            {
                "u": u,
                "g": g,
                "r": r,
                "i": i,
                "z": z,
                "redshift": float(redshift),
            }
        ]
    )

    features_scaled = scaler.transform(features)

    return features_scaled


input = input_features()


if st.button("Predict"):
    prediction = model.predict(input)
    prediction_indices = np.argmax(prediction, axis=1)[0]
    prediction_class = model_columns[prediction_indices]
    if prediction_class == "QSO":
        prediction_class = "Quasar"
    st.success(
        f"The class of the celestial object is: **{prediction_class.title()}**",
    )


st.header("A Small Explanation", divider="gray")

st.markdown(
    """
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
"""
)

summary = pd.DataFrame(
    {
        "Parameter": [
            "u, g, r, i, z",
            "Redshift(z)",
            "spec_obj_ID",
            "run/rerun",
            "α/δ",
        ],
        "Role in Classification": [
            "Defines color profile to distinguish stars, galaxies, and quasars.",
            "Indicates distance and type (stars: z≈0; galaxies/quasars: z>0.1).",
            "Ensures cross-matching between photometric and spectroscopic data.",
            "Affects photometric accuracy but not intrinsic type.",
            "Celestial coordinates that specify the position of an astronomical object in the sky.",
        ],
        "Example Use Case": [
            "A quasar with u - g = -0.6, r - i = 1.2 suggests blue excess and dust reddening.",
            "A z=2.5 object is likely a quasar, while z=0.01 is a nearby galaxy.",
            "Same ID confirms a galaxy’s redshift and emission-line features.",
            "Poor run conditions may blur colours but won’t misclassify a star as a quasar.",
            "Alpha and delta themselves do not determine whether an object is a star, galaxy, or quasar. They only provide the object's location on the celestial sphere.",
        ],
    }
)

st.table(summary)

st.header("About the Model", divider="gray")
st.subheader("Dataset Details")
st.markdown(
    """
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

#### Citations
>fedesoriano. (January 2022). Stellar Classification Dataset - SDSS17. 

>Data originally from the Sloan Digital Sky Survey (SDSS), [Data Release 17](https://www.sdss.org/dr17/).
"""
)

st.subheader("Model Performance")

st.write(
    """ 
    The ANN achieved an accuracy of ~97%. For more detailed metrics, please consult the classification report in the jupyter notebook.    
"""
)


st.divider()
