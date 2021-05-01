import sys
# import libraries
import pandas as pd 
import numpy as np
from sqlalchemy import create_engine
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import pickle
from sklearn.metrics import classification_report
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')



def load_data(database_filepath):
    """
    Function that loads messages and categories cleaned from database using database_filepath as a filepath and sqlalchemy as library
    Returns two numpy arrays X and y
    """
    engine = create_engine('sqlite:///{}'.format(database_filepath))   
    df=pd.read_sql_query('SELECT * FROM CleanedData',engine)
    X = df['message']
    Y = df.drop(['id', 'message','original','genre'], axis=1)
    X=X.to_numpy()
    Y=Y.to_numpy()
    category_names = df.drop(columns = ['id', 'message', 'original', 'genre']).columns.values

    return X,Y,category_names


def tokenize(text):
    """Tokenization function. Receives as input raw text which afterwards normalized, stop words removed, lemmatized.
    Returns tokenized text"""
    
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenize text
    tokens = word_tokenize(text)
    
    # lemmatize and remove stop words
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    
    return tokens


def build_model():
    """
    Function that builds a ML pipeline using Tfidf and OneVsRestClassifier with LogisticRegression and then uses GridsearchCV to optimize the model then return the model.
    """
    LR=LogisticRegression(max_iter=10000)

    # Pipline that include Tfidf and OVR
    pipeline = Pipeline([('tfidf', TfidfVectorizer(tokenizer=tokenize)), ('OvR', OneVsRestClassifier(LR))])
    
    parameters = {
        'OvR__estimator__C': [0.1, 1, 10, 100, 1000]
    }   
    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv

def evaluate_model(model, X_test, Y_test,category_names):
    """
    Evaluate the model with precision, Recall, F1-score and support.
    """
    y_pred=model.predict(X_test)
    report = classification_report(Y_test, y_pred,target_names=category_names,output_dict=True)
    report = pd.DataFrame(report).transpose()
    
    print(report)
    
def save_model(model, model_filepath):
    """
    Saving the model using pickle
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y,category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test,category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()