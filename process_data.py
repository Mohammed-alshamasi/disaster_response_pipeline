import sys
import pandas as pd
import numpy as np

#
#"~/Desktop/disaster_response_pipeline_project/data/disaster_categories.csv"
def load_data(messages_filepath, categories_filepath):
    '''
    INPUT
    file paths of the message and categories files in csv format
    
    OUTPUT
    a dataframe contains both dataset merged
    '''
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages,categories,on='id',how='inner')
    
    return df

def clean_data(df):
    '''
    INPUT
    a dataframe with both messages and categories for data cleaning
    
    OUTPUT
    cleaned dataframe, with new expanding columns for each message category
    '''
    
    # Split `categories` into separate category columns
    categories = df.categories.str.split(';',expand=True)
   
    row = categories.loc[0]
    category_colnames = row.apply(lambda y:y[:-2]) 
    # Rename the new splitted columns
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x:x[-1])
    
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    
    # Drop the old categories column
    df.drop('categories',axis=1,inplace=True)
    
    # Concat the newly cleaned columns
    df=pd.concat([df,categories],axis=1)
    
    # Changing some values that were 2 to 0
    df.loc[df['related'] > 1,'related'] = 0
    
    # Drop duplicates
    df=df.drop_duplicates()
    
    return df


def save_data(df, database_filename):
    '''
    INPUT
    cleaned dataframe and the filepath for the SQL database for saving the dataframe
    OUTPUT
    None
    '''
    from sqlalchemy import create_engine
    
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('CleanedData', engine,if_exists = 'replace', index=False)  
    engine.dispose()

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()