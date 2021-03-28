import os
from flask import Flask, flash, request, redirect, url_for, render_template, render_template_string
from werkzeug.utils import secure_filename
import pathlib

import PIL
import PIL.Image

import recognition

UPLOAD_FOLDER = 'static/images/'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

app = Flask('Flower recognition')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

"""[check uploaded file has allowed extension]
"""
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

"""[Upload file]

Returns:
        [template]: [returns to index for further processing of uploaded file]
"""
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file_input']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('upload_file', filename=filename))
    return render_template('index.html')

"""[ouputs result of flower recognition]

Returns:
        [template]: [render_template('result.html', recognition=result, flower_file=path_uploaded_file, file_name=filename)]
"""
@app.route('/result', methods=['GET', 'POST'])
def give_result():
    if request.method == 'POST':
        uploaded_file = request.files['classify_input']
        filename = secure_filename(uploaded_file.filename)
        path_uploaded_file = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        img = PIL.Image.open(path_uploaded_file)
        result = recognition.get_flower_classification(img)
    return render_template('result.html', recognition=result, flower_file=path_uploaded_file, file_name=filename)

if __name__ == '__main__':
    app.run(debug=True, port=5000)