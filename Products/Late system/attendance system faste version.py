import cv2
import datetime
from tabulate import tabulate
from simple_facerec import SimpleFacerec
import csv
import time

pTime = 0


sfr = SimpleFacerec()
sfr.load_encoding_images("imgs") 

students = []

cap = cv2.VideoCapture("videos/3.mp4")


while True:
    success, img = cap.read()
    if not success:
       print("Error.. try again")
       break
  
    face_locations, face_names = sfr.detect_known_faces(img)
    
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(0,255,0),3)
    
    for (top, right, bottom, left), name in zip(face_locations, face_names):
      cv2.rectangle(img, (left, top), (right, bottom), (255, 0, 0), 2)  # Blue rectangle with thickness 2

      cv2.putText(img, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
      current_time = datetime.datetime.now().strftime("%H:%M:%S")
      if name == "Unknown":
           break
      if name not in [student[0] for student in students]:
           students.append((name, current_time))
        
    cv2.imshow("attendance system", img)

    key = cv2.waitKey(1)
    if key == 27:
       break

table_data = [(idx + 1, student[0], student[1]) for idx, student in enumerate(students)]


print(tabulate(table_data, headers=["No.", "Student Name", datetime.datetime.now().strftime("%Y-%m-%d")], tablefmt="grid"))

with open("attendance.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["No.", "Student Name", datetime.datetime.now().strftime("%Y-%m-%d")])
    writer.writerows(table_data)
