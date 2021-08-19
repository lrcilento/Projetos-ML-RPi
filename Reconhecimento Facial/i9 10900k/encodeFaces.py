from imutils import paths
import face_recognition
import time
import argparse
import pickle
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", required=True, help="Input folder path")
ap.add_argument("-o", "--encodings", required=True, help="Output folder path")
ap.add_argument("-d", "--detection-method", type=str, default="hog", help="Detection model to use: either 'hog' or 'cnn'")
args = vars(ap.parse_args())

print("Quantificando imagens...")
imagePaths = list(paths.list_images(args["dataset"]))

knownEncodings = []
knownNames = []

start_time = time.time()

for (i, imagePath) in enumerate(imagePaths):
    print("Processando imagem {}/{}".format(i + 1, len(imagePaths)))
    name = imagePath.split(os.path.sep)[-2]
    image = cv2.imread(imagePath)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb, model=args["detection_method"])
    encodings = face_recognition.face_encodings(rgb, boxes)

    for encoding in encodings:
        knownEncodings.append(encoding)
        knownNames.append(name)

print("Serializando imagens...")
data = {"encodings": knownEncodings, "names": knownNames}
f = open(args["encodings"], "wb")
f.write(pickle.dumps(data))
f.close()

print ("Time Elapsed: ", str(time.time() - start_time))

# python3 encodeFaces.py --dataset '../Partial Datasets' --encodings encodings.pickle
