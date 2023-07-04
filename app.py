from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

def convert_to_int(word):
    word_dict = {'Vanilla':1.5, 'Chocolate':2, 'Strawberry':2.5, 'MintChip':3, 'CookieDough':3.5, 'Small':0.5, 'Medium':0.75, 'Large':1}
    return word_dict[word]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict.html', methods=['POST'])
def predict():
    flavor = request.form['flavors']
    size = request.form['sizes']
    
    prediction = model.predict([[convert_to_int(flavor),convert_to_int(size)]])

    # language = detect(text)


    return f"The predicted price is: {prediction}"
    

if __name__ == '__main__':
    app.run()