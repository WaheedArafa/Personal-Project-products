import face_recognition
import cv2
import numpy as np
import os

class SimpleFacerec:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []

    def load_encoding_images(self, images_path):
        if not os.path.exists(images_path):
            print(f"Error: The path '{images_path}' does not exist.")
            return

        for person_name in os.listdir(images_path):
            person_path = os.path.join(images_path, person_name)
            
            if not os.path.isdir(person_path):
                print(f"Skipping non-folder item: {person_name}")
                continue
            
            print(f"Loading images for person: {person_name}")
            
            for img_name in os.listdir(person_path):
                img_path = os.path.join(person_path, img_name)

                if not (img_name.endswith('.jpg') or img_name.endswith('.png') or img_name.endswith('.jpeg')):
                    print(f"Skipping non-image file: {img_name}")
                    continue

                img = cv2.imread(img_path)
                if img is None:
                    print(f"Warning: Unable to read image file: {img_name}")
                    continue

                try:
                    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    face_encodings = face_recognition.face_encodings(rgb_img)
                    if face_encodings:
                        img_encoding = face_encodings[0]

                        self.known_face_encodings.append(img_encoding)
                        self.known_face_names.append(person_name)
                    else:
                        print(f"No faces detected in {img_name}. Skipping.")
                except Exception as e:
                    print(f"Error processing {img_name}: {e}")

        print(f"Loaded {len(self.known_face_encodings)} face encodings.")

    def detect_known_faces(self, frame):
        try:
            if not isinstance(frame, np.ndarray):
                print("The frame is not a valid image.")
                return [], []

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            face_locations = face_recognition.face_locations(rgb_frame)

            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                name = "Unknown"

                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = self.known_face_names[best_match_index]

                face_names.append(name)

            return face_locations, face_names
        except Exception as e:
            print(f"Error during face detection: {e}")
            return [], []
