import json

import cv2
from flask import Flask, request, render_template, flash
from flask_uploads import UploadSet, IMAGES, configure_uploads

UPLOAD_FOLDER = 'static/img'
from image_recognition import DetectChars, DetectPlates

app = Flask(__name__)
app.config["DEBUG"] = True
app.config['UPLOADED_FILES_DEST'] = 'static/img'
app.config['UPLOADS_DEFAULT_DEST'] = 'static/img'

photos = UploadSet('photos', IMAGES)
configure_uploads(app, photos)


@app.route('/', methods=['GET'])
def home():
    return "<h1>IMAGE RECOGNITION</h1>"


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    a = request.files
    if request.method == 'POST' and 'photo' in request.files:
        filename = photos.save(request.files['photo'])
        plate = read_plate(photos.path(filename))
        return json.dumps({'filename': filename, 'data': plate})
    return render_template('upload.html')


def read_plate(filepath):
    blnKNNTrainingSuccessful = DetectChars.loadKNNDataAndTrainKNN()  # attempt KNN training

    if blnKNNTrainingSuccessful == False:  # if KNN training was not successful
        print("\nerror: KNN traning was not successful\n")  # show error message
        return {'error': True, 'message': 'El entrenamiento de reconocimiento no tuvo éxito'}
    # end if

    imgOriginalScene = cv2.imread(filepath)  # open image

    if imgOriginalScene is None:  # if image was not read successfully
        print("\nerror: image not read from file \n\n")  # print error message to std out
        return {'error': True, 'message': 'No se pudo leer la imagen'}
    # end if

    listOfPossiblePlates = DetectPlates.detectPlatesInScene(imgOriginalScene)  # detect plates
    listOfPossiblePlates = DetectChars.detectCharsInPlates(listOfPossiblePlates)  # detect chars in plates

    if len(listOfPossiblePlates) == 0:  # if no plates were found
        print("\nno license plates were detected\n")  # inform user no plates were found
        return {'error': True, 'message': 'No se encontraron matrículas en la imagen'}
    else:  # else
        # if we get in here list of possible plates has at leat one plate

        # sort the list of possible plates in DESCENDING order (most number of chars to least number of chars)
        listOfPossiblePlates.sort(key=lambda possiblePlate: len(possiblePlate.strChars), reverse=True)

        # suppose the plate with the most recognized chars (the first plate in sorted by string length descending order) is the actual plate
        licPlate = listOfPossiblePlates[0]

        if len(licPlate.strChars) == 0:  # if no chars were found in the plate
            print("\nno characters were detected\n\n")  # show message
            return {'error': True, 'message': 'No hay caracteres en la imagen'}
        # end if

        print("\nlicense plate read from image = " + licPlate.strChars + "\n")  # write license plate text to std out
        print("----------------------------------------")
        return {'error': False, 'message': licPlate.strChars}


if __name__ == '__main__':
    app.run()
