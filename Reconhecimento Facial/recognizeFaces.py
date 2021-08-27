import face_recognition
import argparse
import pickle
import cv2
import time

ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings", required=True, help="Path to 'encodings.pickle'")
ap.add_argument("-i", "--image", required=True, help="Path to input image")
ap.add_argument("-d", "--detection-method", type=str, default="hog", help="Detection model to use: either 'hog' or 'cnn'")
args = vars(ap.parse_args())

print("Carregando encodings...")
data = pickle.loads(open(args["encodings"], "rb").read())
image = cv2.imread(args["image"])
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
print("Reconhecendo faces...")
boxes = face_recognition.face_locations(rgb, model=args["detection_method"])
encodings = face_recognition.face_encodings(rgb, boxes)

names = []

start_time = time.time()

for encoding in encodings:
	matches = face_recognition.compare_faces(data["encodings"], encoding)
	name = "Desconhecido"

	if True in matches:
		matchedIdxs = [i for (i, b) in enumerate(matches) if b]
		counts = {}

		for i in matchedIdxs:
			name = data["names"][i]
			counts[name] = counts.get(name, 0) + 1

		name = max(counts, key=counts.get)
	
	names.append(name)


for ((top, right, bottom, left), name) in zip(boxes, names):
	cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
	y = top - 15 if top - 15 > 15 else top + 15
	cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
		0.75, (0, 255, 0), 2)

print ("Time Elapsed: ", str(time.time() - start_time))

cv2.imshow("Image", image)
cv2.waitKey(0)

# python3 recognizeFaces.py --encodings encodings.pickle --image 'Testes/Exemplo 1.jpg'
