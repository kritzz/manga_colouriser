from flask import Flask, render_template, request, redirect, url_for
import os
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
from colourise import process_colours, process_hints

app = Flask(__name__)

cwd = os.getcwd()
UPLOAD_LOCATION = os.path.join(cwd, 'static/upload/')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_LOCATION


@app.route('/', methods = ["GET","POST"])
def testing():
    return render_template('index2.html')

@app.route("/upload",methods=["GET","POST"])
def upload():
    errorMessage = None
    if request.method == 'POST':
        # if 'process' in request.form:
        #     return redirect(url_for('process'))
        # else:
        if 'img' not in request.files:
            errorMessage = "No file selected"
        else:
            file = request.files['img']
            if file.filename == '':
                errorMessage = "No file selected"
            else:
                filename = "img_input.jpg"
                file.save(app.config['UPLOAD_FOLDER'] + filename)
                image_url = url_for('static', filename='upload/' + filename)
                return redirect(url_for('display', image_url=image_url))
        return render_template('noInput.html', errorMessage=errorMessage)
    return render_template('error.html')

@app.route("/process",methods=["GET","POST"])
def process():
    original_url = url_for('static', filename='upload/' + "img_input.jpg")
    if "greyscale" in request.form:
        process_colours("greyscale")
        image_url = url_for('static', filename='colours/' + "output.jpg")
    elif "xDoG" in request.form:
        process_colours("xdog")
        image_url = url_for('static', filename='colours/' + "output.jpg")
    elif "hint" in request.form:
        process_hints()
        image_url = url_for('static', filename='colours/' + "output.jpg")
    return render_template('content.html', content=image_url, original = original_url)

@app.route("/display",methods=["GET","POST"])
def display():
    image_url = request.args.get('image_url')
    return render_template('display.html', content=image_url)

 
# main driver function
if __name__ == '__main__':
    app.run(debug=True)