# Disaster Response Pipeline Project

### Purpose of this project:
The purpose of the project is to build a model that classifies disaster messages. Using the web app an emergency worker can input a new message and get classification results in several categories so to have an idea what kind of help is needed.

### Install
1. NumPy
2. Pandas
3. Json
4. Plotly
5. Nltk
6. Flask
7. Sklearn
8. Sqlalchemy
9. Sys
10. Re
11. Pickle

### Files:
- process_data.py: This code extracts data from both CSV files: messages.csv and categories.csv and creates an SQLite database containing a merged and cleaned version of this data.
- train_classifier.py: This code takes the SQLite database produced by process_data.py as an input and uses the data contained within it to train and tune a ML model for categorizing messages. The output is a pickle file containing the fitted model. Test evaluation metrics are also printed as part of the training process.
- run.py: contains the visualization code and the connection with the web page. 
- disaster_messages.csv, disaster_categories.csv contain sample messages (real messages that were sent during disaster events) and categories datasets in csv format.
- templates folder: This folder contains all of the files necessary to run and render the web app.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python train_classifier.py DisasterResponse.db classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
