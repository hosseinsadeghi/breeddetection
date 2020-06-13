from flask import render_template
from flask import Flask, flash, request, redirect
from werkzeug.utils import secure_filename
import random
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os
from model.dog_app import DogDetection

random.seed(8675309)


breed_detector = DogDetection()

UPLOAD_FOLDER = 'static/'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = '2b55241464af362a104880e46b36d2b6'


@app.route('/', methods=['GET', 'POST'])
@app.route('/home', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file', category='danger')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            flash('Upload successful', category='success')
            filename = secure_filename(file.filename)
            filename = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filename)
            message, breed = breed_detector.which_dog(filename)
            img_sim = breed_detector.get_image(breed)
            print(img_sim)
            return render_template('home.html', img=filename, img_sim=img_sim)
    return render_template('home.html')


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


if __name__ == '__main__':
    app.run(debug=False, threaded=False)
