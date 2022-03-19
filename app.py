from flask import Flask, render_template, request
from transformers import BertTokenizer
import torch

from transformers import BertForSequenceClassification

import numpy as np
import joblib
import pickle

app = Flask(__name__)

@app.route('/')


def home():
    return render_template('home.html')

@app.route('/predict/', methods=['GET', 'POST'])
def predict():
    if request.method == "POST":
        if request.form['submit_button'] == 'Logistische Regression':
            text = request.form.get('text')
            prediction = preprocessDataAndPredict(text)  # pass prediction to template
            return render_template('predict.html', prediction=prediction)

        elif request.form['submit_button'] == 'Bert-Model':
            text = request.form.get('text')
            prediction = preprocessDataAndPredictBert(text)  # pass prediction to template
            return render_template('predict.html', prediction=prediction)
        else:
            pass
    pass

label_dict={'welt': 0,
 'derspiegel': 1,
 'Der_Postillon': 2,
 'WELT_GLASAUGE': 3,
 'DIEZEIT': 4,
 'BILD': 5,
 'titanic': 6,
 'SZ': 7}

def preprocessDataAndPredict(text):
  
    test_data = [text]
    print(test_data)


    # open file
    file = open("Model/text_classification.joblib", "rb")

    # load trained model
    trained_model = joblib.load(file)

    # predict

    prediction = trained_model.predict(test_data)

    return prediction


def preprocessDataAndPredictBert(text):
  
    test_data = [text]
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                      num_labels=len(label_dict),
                                                      output_attentions=False,
                                                      output_hidden_states=False)

    model.load_state_dict(torch.load('BertModel/finetuned_BERT_epoch_3.model', map_location=torch.device('cpu')))




  

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',  
                                          truncation=True,
                                          do_lower_case=True)

    tokens = tokenizer.encode(test_data, return_tensors="pt")

  
    result = model(tokens)
    prediction= [k for k, v in label_dict.items() if v == int(torch.argmax(result.logits))]

    return prediction

if __name__ == '__main__':
    app.run(debug=True)