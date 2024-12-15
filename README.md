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
`/inference_test_audios `directory by using command-line arguments. The argument `2` enables matching the recorded audio 
samples to the celebrity, as demonstrated in the university's demo.

## Dataset

We have significantly normalized and pre-processed the data, as explained in the report. Additionally, to further 
improve the dataset quality, we downloaded, collected, and labeled a substantial number of samples from various sources, 
such as YouTube. The dataset collection code is located in the `/data_scripts` directory.
The final version of the dataset is accessible via the following link: [release_in_the_wild.zip](https://drive.google.com/file/d/1J14RA2mJMlUWO_J_i-VIi3IJhlI39BTZ/view)