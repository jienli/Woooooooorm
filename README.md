# Woooooooorm
CSCI2470 Final Project

This directory only contains scripts (due to size limitations), the data should be placed in the `../data` directory, described as follows:

Data (not included here, but can be found here: https://www.dropbox.com/scl/fo/42zi674xd0mk206brzoyd/AKW8ZWKjXNpcPnuaidOwLfI?rlkey=5cknvgmwn42z9ks0btoj8n236&st=q0aervm7&dl=0)
should be organized as follows:

`../data`   (includes all metadata files, such as filenames_mapped_ages.csv)

`../data/exported_data` (should include the extracted features files from the 60s motion videos: features_summary_tierpsy_plate_20241205_033129.csv and filenames_summary_tierpsy_plate_20241205_033129.csv)

`../data/exported_data/images` (should include all images)

`../data/exported_data/Videos_head_all` (should include all head videos)

(60s motion videos are not provided even in Dropbox, due to size limitations(~300G))


Pumping rate-related scripts can be found in `attention-based_1D_CNN.py` and `attention-based_3D_CNN.py`; head video prediction scripts are located in `Video_3D_ResNet.ipynb`; Image CNN and CvT are located in the `/Image CNN and CvT` folder (please run `preprocess.py` first before running `main`; Image resnet, feature-based model, and multimodal learning can be found in `\Oscar scripts` (please run `preprocess.py` and `preprocess_MotionFeatures.py` before running `main_resnet.py`, `main_MotionFeatures.py`, or `main_multimodal.py`)
