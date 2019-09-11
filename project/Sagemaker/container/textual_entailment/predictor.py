# This is the file that implements a flask server to do inferences. It's the file that you will modify to
# implement the scoring for your own algorithm.

import os
import json
import torch
import flask
from flask import jsonify
from torchtext import data
from torchtext import datasets

#Define the path
prefix = '/opt/ml/'
model_path = os.path.join(prefix, 'model')

#Torch use GPU if available
if torch.cuda.is_available():
    torch.cuda.set_device(0)
    device = torch.device('cuda:{}'.format(0))
else:
    device = torch.device('cpu')

# Load the model components
inputs = data.Field(lower=True, tokenize='spacy')
answers = data.Field(sequential=False)
text_fields = {'sentence1': ('premise', inputs),
             'sentence2': ('hypothesis', inputs)}
inputs.vocab = torch.load(os.path.join(model_path, 'inputs_vocab.pt'))
model = torch.load(os.path.join(model_path, 'model.pt'), map_location=device)

# The flask app for serving predictions
app = flask.Flask(__name__)
@app.route('/ping', methods=['GET'])
def ping():
    # Check if the classifier was loaded correctly
    try:
        model
        status = 200
    except:
        status = 400
    return flask.Response(response= json.dumps(' '), status=status, mimetype='application/json' )

@app.route('/invocations', methods=['POST'])
def transformation():
    # Get input JSON data and convert it to a DF
    input_json = flask.request.get_json()
    json_input = json.dumps(input_json)
    example = data.example.Example.fromJSON(json_input, text_fields)
    example_list = [example]
    # Tokenize data and predict
    if isinstance(text_fields, dict):
        fields, field_dict=[],text_fields
        for field in field_dict.values():
            if isinstance(field, list):
                fields.extend(field)
            else:
                fields.append(field)

    predict = data.dataset.Dataset(examples = example_list, fields = fields)
    predict_iter = data.iterator.Iterator(dataset = predict, batch_size = 1, device=device)
    predict_item = next(iter(predict_iter))
    model.eval()
    answer = model(predict_item)
    xmax = int(torch.max(answer, 1)[1])
    
    # Transform predicted labels (1, 2, and 3) to easier to understand as (Entailment, Contradiction, and Neutral)
    prediction = lambda x: 'Entailment' if x == 1 else ('Contradiction' if
                       x == 2 else 'Neutral')
    label = prediction(xmax)
    percentage = float(torch.exp(answer[0][xmax])/torch.sum(torch.exp(answer)))
    percentage = '{:.2%}'.format(percentage)

    # Transform predictions to JSON
    result = {'label': label, 'probability': percentage}
    result = json.dumps(result)
    return flask.Response(response=result, status=200, mimetype='application/json')
