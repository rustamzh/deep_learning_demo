from flask import Flask, request, session, g, redirect, url_for, abort, \
    render_template, flash, send_from_directory

from werkzeug.utils import secure_filename
import os
import tempfile
from custom_layers import CustomPooling, CustomResidual, load_custom_model
import keras
import yaafelib
import numpy as np
import audioread
from sklearn import preprocessing
import soundfile as sf
from pydub import AudioSegment
from keras import backend as K
app = Flask(__name__)
UPLOAD_FOLDER = tempfile.gettempdir()
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'mp3', 'jpg', 'jpeg', 'gif', 'm4a', 'flac', 'au', 'wav'}
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
            print('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            print ('No selected file')
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
    if filename == "beethoven.mp3":
        full_path = "static/beethoven.mp3"
    model = keras.models.load_model('static/55_8feature.h5',
                                    custom_objects={'CustomResidual': CustomResidual, 'CustomPooling': CustomPooling})
    # model = load_custom_model()
    # model.build()
    # model.summary()
    z = extract_feature(full_path, offset)
    print np.shape(z[np.newaxis, ...])
    y = model.predict(z[np.newaxis, ...])
    K.clear_session()
    # model.load_weights('static/55_8feature.h5')
    return str(y.argmax()==1)


def preprocessed(feats):
    feature_labels = ["ac", "en", "mfcc", "sf", "sr", "ss", "sp", "zcr"]
    feature_m = []
    for feature in feature_labels:
        feature_m.append(feats[feature])

    return preprocessing.scale(np.asarray(np.matrix(np.column_stack(feature_m))))


def extract_feature(filename, offset):
    fp = yaafelib.FeaturePlan(sample_rate=22050, resample=True)
    fp.loadFeaturePlan('static/featureplan.txt')
    engine = yaafelib.Engine()
    engine.load(fp.getDataFlow())
    print(filename)
    print offset

    sound = AudioSegment.from_file(filename)

    halfway_point = int(offset)*1000
    end = halfway_point + 30000
    first_half = sound[halfway_point:end]
    filename = os.path.join(app.config['UPLOAD_FOLDER'], os.path.splitext(os.path.basename(filename))[0]+str(offset)+".cliped.wav")
    if not os.path.isfile(filename):
        first_half.export(filename, format="wav")
    afp = yaafelib.AudioFileProcessor()
    afp.processFile(engine, filename)
    feats = engine.readAllOutputs()
    return preprocessed(feats)

if __name__ == '__main__':
    app.run(threaded = False, host="0.0.0.0")
