# Celebrity Voice Match

Audio Processing and Indexing project by Giulia Rivetti, Yağmur Doğan, Evan Meltz, Kacper Kadziolka  
Note: All the `.ipynb` notebooks are imported from the Google Colab environment.

## Installation

Install project dependencies by running the following command:

```bash
pip install -r requirements.txt
```

## Demo

The file `inference.py` contains the code to run the demo. It uses the pre-trained CNN model from the `/model` directory, 
where we store various iterations of the CNN model. The code allows you to run test examples located in the 
`/inference_test_audios `directory by using command-line arguments. The argument 2 enables matching the recorded audio 
samples to the celebrity, as demonstrated in the university's demo.

## Dataset

We have significantly normalized and pre-processed the data, as explained in the report. Additionally, to further 
improve the dataset quality, we downloaded, collected, and labeled a substantial number of samples from various sources, 
such as YouTube. The dataset collection code is located in the `/data_scripts` directory.
The final version of the dataset is accessible via the following link: [release_in_the_wild.zip](https://drive.google.com/file/d/1J14RA2mJMlUWO_J_i-VIi3IJhlI39BTZ/view)

## Training

We trained our flagship CNN model using the notebook located in the `/training` directory. The notebook contains code 
for data normalization and pre-processing, various proof-of-concept experiments conducted along the way, and the final 
model training process. After training and evaluating the model, we plotted confusion matrices and saved the trained 
model to the `/model `directory.

Additionally, we experimented with the Wav2Vec architecture using a notebook located in the same `/training` directory.

Before initiating our first efforts to create baselines and develop more sophisticated solutions, we conducted 
Exploratory Data Analysis (EDA). The EDA code is located in the `/data_analysis directory`.

## Baselines

To create baselines, we used the following methods:
* We first created a simple SVM model using the scikit-learn library. The code is located in the `/baselines/SVM `directory.
Note: This method requires direct access to the dataset for training.
* We experimented with Random Forest models using various feature extraction methods. The code is located in the 
`/baselines/feature_engineering` directory. We tested combinations of delta MFCC, spectral bandwidth, spectral contrast, 
prosodic features, LPC, zero-crossing rate, tempo, harmonic-percussive source separation, and mel spectrograms with 
dimensionality reduction methods. Note: This method also requires direct access to the dataset for training.
* The main baseline, which we will later use for comparison with the CNN and Wav2Vec models, is located in the 
`/baselines/random_forest directory`. This is the Random Forest classifier from the scikit-learn package. The training 
iterations were performed in the Google Colab environment, and the trained model is utilized in the 
`inference_RandomForest.py` script.