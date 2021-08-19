import face_recognition
import argparse
import imutils
import pickle
import time
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings", required=True, help="Path to 'encodings.pickle'")
ap.add_argument("-i", "--input", required=True, help="Path to input video")
ap.add_argument("-o", "--output", type=str, help="Path to output video")
ap.add_argument("-y", "--display", type=int, default=1, help="Whether or not to display output frame to screen")
ap.add_argument("-d", "--detection-method", type=str, default="hog", help="Detection model to use: either 'hog' or 'cnn'")
args = vars(ap.parse_args())

print("Carregando encodings...")
data = pickle.loads(open(args["encodings"], "rb").read())

print("Processando video...")
stream = cv2.VideoCapture(args["input"])
writer = None

start_time = time.time()

while True:
	(grabbed, frame) = stream.read()

	if not grabbed:
		break

	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	rgb = imutils.resize(frame, width=750)
	r = frame.shape[1] / float(rgb.shape[1])

	boxes = face_recognition.face_locations(rgb, model=args["detection_method"])
	encodings = face_recognition.face_encodings(rgb, boxes)
	names = []

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
		top = int(top * r)
		right = int(right * r)
		bottom = int(bottom * r)
		left = int(left * r)

		cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
		y = top - 15 if top - 15 > 15 else top + 15
		cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 24,
			(frame.shape[1], frame.shape[0]), True)

	if writer is not None:
		writer.write(frame)

	if args["display"] > 0:
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF

		if key == ord("q"):
			break

print ("Time Elapsed: ", str(time.time() - start_time))

stream.release()

if writer is not None:
	writer.release()