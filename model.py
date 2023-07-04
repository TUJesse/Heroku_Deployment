# Importing the libraries
import numpy as np
import pandas as pd
import pickle

dataset = pd.read_csv('ice_cream_flavors.csv')

dataset['price'].fillna(0, inplace=True)

X = dataset.iloc[:, :2]

#Converting words to integer values
def convert_to_int(word):
    word_dict = {'Vanilla':1.5, 'Chocolate':2, 'Strawberry':2.5, 'MintChip':3, 'CookieDough':3.5, 'Small':0.5, 'Medium':0.75, 'Large':1}
    return word_dict[word]

X['flavor'] = X['flavor'].apply(lambda x : convert_to_int(x))
X['size'] = X['size'].apply(lambda x : convert_to_int(x))


y = dataset.iloc[:, 2:3]

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

#Fitting model with trainig data
regressor.fit(X, y)

# Saving model to disk
pickle.dump(regressor, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[convert_to_int('MintChip'), convert_to_int('Small')]]))