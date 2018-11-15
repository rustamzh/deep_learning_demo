from flask import Flask, request, session, g, redirect, url_for, abort, \
    render_template, flash, send_from_directory

from werkzeug.utils import secure_filename
import os
import tempfile
from custom_layers import CustomPooling, CustomResidual, load_custom_model
import keras
import yaafelib
import numpy as np
import librosa
from sklearn import preprocessing

app = Flask(__name__)
UPLOAD_FOLDER = tempfile.gettempdir()
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'mp3', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = "sdadsadasdasd"


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def hello_world():
    print(UPLOAD_FOLDER)
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            print(file.filename)
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return "".join(["./uploads/", filename])
    return render_template('index.html')


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)


@app.route('/offset/<filename>/<offset>')
def keras_class(filename, offset):
    full_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    model = keras.models.load_model('static/55_8feature.h5',
                                    custom_objects={'CustomResidual': CustomResidual, 'CustomPooling': CustomPooling})
    # model = load_custom_model()
    # model.build()
    # model.summary()
    z = extract_feature(full_path, offset)
    y = model.predict(z)
    # model.load_weights('static/55_8feature.h5')
    return y


def preprocessed(feats):
    feature_labels = ["ac","en","mfcc","sf","sr","ss","sp","zcr"]
    feature_m = []
    for feature in feature_labels :
        feature_m.append(feats[feature])
    return preprocessing.scale(np.asarray(np.matrix(feature_m).transpose()))


def extract_feature(filename, offset):
    fp = yaafelib.FeaturePlan(sample_rate=22050)
    fp.loadFeaturePlan('static/featureplan.txt')
    engine = yaafelib.Engine()
    engine.load(fp.getDataFlow())
    y, sr = librosa.load(filename, offset=offset, duration=30)
    feats = engine.processAudio(y)
    return preprocessed(feats)


if __name__ == '__main__':
    app.run()
