from crypt import methods
from flask import Flask, render_template, request
import numpy as np
from joblib import load
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import uuid

### THIS IS A TEST CHANGE TO SEE IF IT IS REFELCTED IN GITHUB 

def make_picture(training_data_filename, model_in, new_input_arr, output_file):
  # Plot training data with model
  data = pd.read_pickle(training_data_filename)
  data = data[data['Age'] > 0]
  ages = data['Age']
  heights = data['Height']
  x_new = np.array(list(range(19))).reshape(19,1)
  preds = model_in.predict(x_new)

  fig = px.scatter(x=ages, y=heights, title="Height vs Age", labels={'x': 'Age (Years)',
                                                                   'y': 'Height (Inches)'})
  fig.add_trace(
      go.Scatter(x=x_new.reshape(19), y=preds, mode='lines', name='Model'))

  new_preds = model_in.predict(new_input_arr)
  # Plot new predictions
  fig.add_trace(go.Scatter(x=new_input_arr.reshape(len(new_input_arr)), 
                          y=new_preds,
                          name='New Outputs',
                          mode='markers', 
                          marker=dict(color='purple', size=20, line=dict(color='purple', width=2))) ) 


  fig.write_image(output_file, width=800, engine='kaleido')
  fig.show()        



def floats_string_to_input_arr(floats_str):
  floats = [float(x) for x in [x for x in floats_str.split(',')] if x != '']
  as_np_arr = np.array(floats).reshape(len(floats), 1)
  return as_np_arr


app = Flask(__name__)

@app.route("/", methods=['GET', 'POST'])
def hello_world():
    request_type_str = request.method
    if request_type_str == 'GET':
        return render_template('index.html', href='static/base_pic.svg')
    else:
        text = request.form['text']
        random_str = uuid.uuid4().hex
        #path= f'application/static/{random_str}.svg'
        path= f'/Users/delgadonoriega/Desktop/FlaskTutorialTwo/live-demo-flask-app/application/static/{random_str}.svg'
        #model = load('live-demo-flask/application/model.joblib')
        model= load('/Users/delgadonoriega/Desktop/FlaskTutorialTwo/live-demo-flask-app/application/model.joblib')
        np_arr = floats_string_to_input_arr(text)
        #make_picture('live-demo-flask/application/AgesAndHeights.pkl', model, np_arr, path)
        make_picture('/Users/delgadonoriega/Desktop/FlaskTutorialTwo/live-demo-flask-app/application/AgesAndHeights.pkl', model, np_arr, path)
        return render_template('index.html', href=path[12:])




if __name__ == '__main__':
    app.run()

    #sdfwerfwer 