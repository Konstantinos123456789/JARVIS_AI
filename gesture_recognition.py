import cv2
import numpy as np

class FaceDetector:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def detect_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        return faces

class GestureRecognizer:
    def __init__(self):
        self.gesture_model = cv2.ml.KNearest_create()

    def train(self, samples, responses):
        self.gesture_model.train(samples, cv2.ml.ROW_SAMPLE, responses)

    def recognize_gestures(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        features = self.extract_features(blur)
        if len(features) == 0:
            return None
        features_array = np.array([features[0]], dtype=np.float32)
        ret, results, neighbours, dist = self.gesture_model.findNearest(features_array, k=5)
        return results

    def extract_features(self, blur):
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        features = []
        for contour in contours:
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w)/h if h != 0 else 0
            features.append((area, aspect_ratio))
        return features

def gesture_recognition():
    """Main function to run gesture and face recognition"""
    cap = cv2.VideoCapture(0)

    face_detector = FaceDetector()
    gesture_recognizer = GestureRecognizer()

    # Sample data for training the model
    samples = np.array([
        [10, 2], [20, 4], [30, 6], [40, 8], [50, 10]
    ], dtype=np.float32)
    responses = np.array([
        1, 2, 3, 4, 5
    ], dtype=np.int32)
    gesture_recognizer.train(samples, responses)

    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = face_detector.detect_faces(frame)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        gestures = gesture_recognizer.recognize_gestures(frame)
        if gestures is not None:
            print(f"Gesture detected: {gestures}")

        cv2.imshow('Face Detection and Gesture Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    gesture_recognition()
