import cv2
import dlib
import numpy as np
import sqlite3
import tkinter as tk
from tkinter import simpledialog
from PIL import Image, ImageTk

# Initialize dlib's face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_rec = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

# Initialize SQLite database
conn = sqlite3.connect('face_recognition.db')
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS users
             (id INTEGER PRIMARY KEY AUTOINCREMENT,
              name TEXT NOT NULL,
              face_encoding BLOB NOT NULL)''')
conn.commit()

class FaceRecognitionApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)

        self.video_source = 0
        self.vid = cv2.VideoCapture(self.video_source)

        self.canvas = tk.Canvas(window, width=self.vid.get(cv2.CAP_PROP_FRAME_WIDTH), height=self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.canvas.pack()

        self.btn_add_user = tk.Button(window, text="Add User", width=10, command=self.add_user)
        self.btn_add_user.pack(pady=10)

        self.delay = 15
        self.update()

        self.window.mainloop()

    def add_user(self):
        name = simpledialog.askstring("Input", "What's your name?", parent=self.window)
        if name is not None:
            ret, frame = self.vid.read()
            if ret:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                faces = detector(rgb_frame)
                if len(faces) > 0:
                    shape = predictor(rgb_frame, faces[0])
                    face_encoding = np.array(face_rec.compute_face_descriptor(rgb_frame, shape))
                    
                    # Store in database
                    c.execute("INSERT INTO users (name, face_encoding) VALUES (?, ?)",
                              (name, face_encoding.astype(np.float64).tobytes()))
                    conn.commit()
                    print(f"User {name} added to database.")
                else:
                    print("No face detected. Please try again.")
            else:
                print("Failed to capture frame.")

    def update(self):
        ret, frame = self.vid.read()
        if ret:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = detector(rgb_frame)
            
            for face in faces:
                shape = predictor(rgb_frame, face)
                face_encoding = np.array(face_rec.compute_face_descriptor(rgb_frame, shape))
                
                # Initialize the best match variables
                best_match_name = None
                best_match_distance = 0.6  # Set the threshold here

                # Compare with database
                c.execute("SELECT name, face_encoding FROM users")
                for row in c.fetchall():
                    db_name, db_encoding = row
                    db_encoding = np.frombuffer(db_encoding, dtype=np.float64)
                    
                    # Ensure that both encodings are of the same shape
                    if face_encoding.shape == db_encoding.shape:
                        distance = np.linalg.norm(face_encoding - db_encoding)
                        if distance < best_match_distance:
                            best_match_distance = distance
                            best_match_name = db_name

                # Display the name of the best match
                if best_match_name:
                    cv2.putText(frame, best_match_name, (face.left(), face.top() - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
                cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)

            self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        
        self.window.after(self.delay, self.update)

# Create a window and pass it to the Application object
FaceRecognitionApp(tk.Tk(), "Face Recognition App")
