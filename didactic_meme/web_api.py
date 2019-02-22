import flask
import numpy as np
import pandas as pd
import torch
import logging
logger = logging.getLogger(__name__)


app = flask.Flask(__name__, template_folder='templates')

@app.route('/')
def index():
    print('visitor')
    return flask.render_template('form.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    print('prediction')
    data_json = {
        key: np.array([float(flask.request.form.get(key))]) for key in ['feature1', 'feature2']
    }
    # data_json = flask.request.get_json()
    data = pd.DataFrame(data=data_json)
    dist = my_model.predict(data, model, config)
    return flask.jsonify(dist.mean.numpy().tolist())


if __name__ == '__main__':
    print('loading model and starting flask...')
    model_dir = 'runs/decay'
    config = my_model.model.Config(model_dir)
    model = my_model.load(config.model_dir)

    app.run()
