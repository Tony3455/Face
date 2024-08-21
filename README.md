import cv2
import dlib
import numpy as np
import sqlite3
import tkinter as tk
from tkinter import simpledialog, messagebox
from PIL import Image, ImageTk
import face_recognition

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

        self.btn_delete_user = tk.Button(window, text="Delete User", width=10, command=self.delete_user)
        self.btn_delete_user.pack(pady=10)

        self.delay = 15
        self.clean_database()  # Clean the database at the start
        self.update()

        self.window.mainloop()

    def clean_database(self):
        """Remove any users with invalid face encodings."""
        c.execute("SELECT id, name, face_encoding FROM users")
        for row in c.fetchall():
            user_id, name, db_encoding = row
            db_encoding = np.frombuffer(db_encoding, dtype=np.float64)
            
            if db_encoding.shape != (128,):
                print(f"Removing user {name} with invalid encoding.")
                c.execute("DELETE FROM users WHERE id=?", (user_id,))
                conn.commit()

    def add_user(self):
        name = simpledialog.askstring("Input", "What's your name?", parent=self.window)
        if name is not None:
            ret, frame = self.vid.read()
            if ret:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                faces = detector(rgb_frame)
                if len(faces) > 0:
                    shape = predictor(rgb_frame, faces[0])
                    face_encoding = np.array(face_rec.compute_face_descriptor(rgb_frame, shape), dtype=np.float64)

                    # Ensure correct encoding length
                    if face_encoding.shape == (128,):
                        # Store in database
                        c.execute("INSERT INTO users (name, face_encoding) VALUES (?, ?)",
                                  (name, face_encoding.tobytes()))
                        conn.commit()
                        print(f"User {name} added to database.")
                    else:
                        print("Face encoding has an unexpected size.")
                else:
                    print("No face detected. Please try again.")
                    self.display_image_not_found()
            else:
                print("Failed to capture frame.")
                self.display_image_not_found()

    def delete_user(self):
        name = simpledialog.askstring("Input", "Enter the name of the user to delete:", parent=self.window)
        if name:
            c.execute("SELECT id FROM users WHERE name=?", (name,))
            result = c.fetchone()
            if result:
                confirm = messagebox.askyesno("Confirm Delete", f"Are you sure you want to delete user {name}?")
                if confirm:
                    c.execute("DELETE FROM users WHERE name=?", (name,))
                    conn.commit()
                    print(f"User {name} deleted from the database.")
                else:
                    print("Delete operation cancelled.")
            else:
                messagebox.showerror("Error", f"No user found with the name {name}.")

    def display_image_not_found(self):
        self.photo = ImageTk.PhotoImage(Image.fromarray(np.zeros((int(self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)), 3), dtype=np.uint8)))
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

    def update(self):
        ret, frame = self.vid.read()
        if ret:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = detector(rgb_frame)

            for face in faces:
                shape = predictor(rgb_frame, face)
                face_encoding = np.array(face_rec.compute_face_descriptor(rgb_frame, shape), dtype=np.float64)

                # Compare with database
                c.execute("SELECT name, face_encoding FROM users")
                known_face_encodings = []
                known_face_names = []
                for row in c.fetchall():
                    db_name, db_encoding = row
                    db_encoding = np.frombuffer(db_encoding, dtype=np.float64)

                    if db_encoding.shape == (128,):
                        known_face_encodings.append(db_encoding)
                        known_face_names.append(db_name)

                # Calculate distances between the new face encoding and known face encodings
                if known_face_encodings:
                    distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                    min_distance = np.min(distances) if distances.size > 0 else None
                    best_match_index = np.argmin(distances) if min_distance is not None else None

                    if min_distance is not None and min_distance < 0.4:  # Adjust the threshold here
                        name = known_face_names[best_match_index]
                    else:
                        name = "User not found"
                else:
                    name = "User not found"

                cv2.putText(frame, name, (face.left(), face.top() - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)

            self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        else:
            self.display_image_not_found()

        self.window.after(self.delay, self.update)

# Create a window and pass it to the Application object
FaceRecognitionApp(tk.Tk(), "Face Recognition App")
