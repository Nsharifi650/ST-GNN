# Spatio-Temperoal Graph Neural Networks 

This repository provides a framework and model for training spatio-temporal graph neural networks (ST-GNN) using PyTorch Geometric, currently it is built using stock data as the default but this can be easily modified to use different dataset which has an inherent graphical realtionship.

Even on stock data  - it has an impressive performance - significantly better than traditional GRU/LSTM methods alone.

1. Setup
Clone the Repository

2. Install Dependencies
pip install -r requirements.txt

3. Configuration
The config.yaml file in (config/config.yaml) holds key settings for data sources, model parameters, and inference options. Edit this file to:
a. Specify the data path and stock market data parameters.
b. Adjust model and inference settings.

4. Running:
python main.py

This will read config.yaml and run the training and testing pipeline with the current settings.

5. Adapting for Other Data Types
If you wish to work with data other than stock market information then you'll need to modify config.yaml: update the data source and any relevant parameters.
Adjust Data Processing: Change preprocessing steps in the data loader or related modules to suit your new data format.

If necessary, tweak model architecture to better fit your new data type.
