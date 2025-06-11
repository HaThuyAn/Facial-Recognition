import os.path
import datetime
import sys

import tkinter as tk
import cv2
from PIL import Image, ImageTk
import pickle
from utils import get_face_landmarks

with open('C:/Users/PC/Downloads/Applied ML/FacialProject/emotion-recognition/emotion_model', 'rb') as f:
    emotion_model = pickle.load(f)

emotion_labels = ['HAPPY', 'SAD', 'NEUTRAL']

import util
from test import test


class App:
    def __init__(self):
        self.main_window = tk.Tk()
        self.main_window.geometry("1100x520+250+100")

        self.login_button_main_window = util.get_button(self.main_window, 'Login', 'green', self.login)
        self.login_button_main_window.place(x=650, y=250)

        self.register_new_user_button_main_window = util.get_button(self.main_window, 'Register', 'gray',
                                                                    self.register_new_user, fg='black')
        self.register_new_user_button_main_window.place(x=650, y=400)

        self.webcam_label = util.get_img_label(self.main_window)
        self.webcam_label.place(x=10, y=0, width=600, height=500)

        ICON_SIZE = (50, 50)

        self.emotion_icons = {
            "happy": ImageTk.PhotoImage(Image.open("icons/happy.png").resize(ICON_SIZE, Image.Resampling.LANCZOS)),
            "neutral": ImageTk.PhotoImage(Image.open("icons/neutral.png").resize(ICON_SIZE, Image.Resampling.LANCZOS)),
            "sad": ImageTk.PhotoImage(Image.open("icons/sad.png").resize(ICON_SIZE, Image.Resampling.LANCZOS))
        }

        self.status_emotion_frame = tk.Frame(self.main_window)
        self.status_emotion_frame.place(x=650, y=100)

        self.detection_label = tk.Label(self.status_emotion_frame, text="Status: Waiting", font=("Arial", 14), fg="green")
        self.detection_label.grid(row=0, column=0, padx=10, pady=5, sticky="w")

        self.emotion_label = tk.Label(self.status_emotion_frame)
        self.emotion_label.grid(row=0, column=1, padx=10, pady=5)

        self.emotion_text_label = tk.Label(self.status_emotion_frame, font=("Arial", 14), fg="black")
        self.emotion_text_label.grid(row=1, column=1, padx=10, pady=5)

        self.add_webcam(self.webcam_label)

        self.db_dir = './db'
        if not os.path.exists(self.db_dir):
            os.mkdir(self.db_dir)

        self.log_path = './log.txt'

    def add_webcam(self, label):
        if 'cap' not in self.__dict__:
            self.cap = cv2.VideoCapture(0)
            # Reduce resolution for faster processing
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self._label = label
        self.frame_count = 0  # Track frame number
        self.process_webcam()

    def process_webcam(self):
        ret, frame = self.cap.read()

        if not ret:
            print("Error: Failed to capture image from webcam")
            return

        emotion_icon = None  # Initialize emotion icon

        # Run the anti-spoofing check
        label = test(
            image=frame,
            model_dir='C:/Users/PC/Downloads/Applied ML/project/Silent-Face-Anti-Spoofing/resources/anti_spoof_models',
            device_id=0
        )

        # Check if the face is real
        if label == 1:
            self.detection_label.config(text="Status: Real Face", fg="green")

            try:
                face_landmarks = get_face_landmarks(frame, draw=False, static_image_mode=False)

                # Ensure landmarks are valid and of correct shape
                if face_landmarks is not None and isinstance(face_landmarks, (list, tuple)) and len(face_landmarks) > 0:
                    try:
                        prediction = emotion_model.predict([face_landmarks])[0]
                        predicted_emotion = emotion_labels[int(prediction)]

                        self.emotion_text_label.config(text=f"Emotion: {predicted_emotion}")
                        emotion_icon = self.emotion_icons.get(predicted_emotion.lower())
                        if emotion_icon:
                            self.emotion_label.config(image=emotion_icon)
                    except Exception as e:
                        print("Prediction failed:", e)
                        self.emotion_text_label.config(text="")
                        self.emotion_label.config(image="")
                else:
                    # Handle missing face or landmarks
                    self.emotion_text_label.config(text="")
                    self.emotion_label.config(image="")
            except Exception as e:
                print("Emotion detection error:", e)
                self.emotion_text_label.config(text="")
                self.emotion_label.config(image="")

        else:
            # Display fake face status and clear emotion text and icon
            self.detection_label.config(text="Status: Spoof Detected", fg="red")
            self.emotion_text_label.config(text="")  # Clear emotion text
            self.emotion_label.config(image="")  # Clear emotion icon

        # Update the label with the processed frame
        self.most_recent_capture_arr = frame
        img_ = cv2.cvtColor(self.most_recent_capture_arr, cv2.COLOR_BGR2RGB)
        self.most_recent_capture_pil = Image.fromarray(img_)
        imgtk = ImageTk.PhotoImage(image=self.most_recent_capture_pil)
        self._label.imgtk = imgtk
        self._label.configure(image=imgtk)

        # Continue updating the frame
        self._label.after(20, self.process_webcam)

    def login(self):
        label = test(
                image=self.most_recent_capture_arr,
                model_dir='C:/Users/PC/Downloads/Applied ML/project/Silent-Face-Anti-Spoofing/resources/anti_spoof_models',
                device_id=0
                )

        if label == 1:
            unknown_img_path = "temp.jpg"
            cv2.imwrite(unknown_img_path, self.most_recent_capture_arr)
            name = util.recognize(unknown_img_path, self.db_dir)
            os.remove(unknown_img_path)

            if name == 'Not found':
                util.msg_box('User not found!', 'Unknown user. Please register or try again!')
            else:
                util.msg_box('Welcome back!', 'Welcome, {}.'.format(name[:-4]))
                with open(self.log_path, 'a') as f:
                    f.write('{},{},in\n'.format(name[:-4], datetime.datetime.now()))
                    f.close()

        else:
            util.msg_box('Spoof detected!', 'Please show your real face!')

    def register_new_user(self):
        label = test(
                image=self.most_recent_capture_arr,
                model_dir='C:/Users/PC/Downloads/Applied ML/project/Silent-Face-Anti-Spoofing/resources/anti_spoof_models',
                device_id=0
                )

        if label == 1:
            self.register_new_user_window = tk.Toplevel(self.main_window)
            self.register_new_user_window.geometry("1200x520+370+120")

            self.accept_button_register_new_user_window = util.get_button(self.register_new_user_window, 'Accept', 'green', self.accept_register_new_user)
            self.accept_button_register_new_user_window.place(x=750, y=300)

            self.try_again_button_register_new_user_window = util.get_button(self.register_new_user_window, 'Try again', 'red', self.try_again_register_new_user)
            self.try_again_button_register_new_user_window.place(x=750, y=400)

            self.capture_label = util.get_img_label(self.register_new_user_window)
            self.capture_label.place(x=10, y=0, width=700, height=500)

            self.add_img_to_label(self.capture_label)

            self.entry_text_register_new_user = util.get_entry_text(self.register_new_user_window)
            self.entry_text_register_new_user.place(x=750, y=150)

            self.text_label_register_new_user = util.get_text_label(self.register_new_user_window, 'Please, \ninput username:')
            self.text_label_register_new_user.place(x=750, y=70)
        else:
            util.msg_box('Spoof detected!', 'Please show your real face!')

    def try_again_register_new_user(self):
        self.register_new_user_window.destroy()

    def add_img_to_label(self, label):
        imgtk = ImageTk.PhotoImage(image=self.most_recent_capture_pil)
        label.imgtk = imgtk
        label.configure(image=imgtk)

        self.register_new_user_capture = self.most_recent_capture_arr.copy()

    def start(self):
        self.main_window.mainloop()

    def accept_register_new_user(self):
        name = self.entry_text_register_new_user.get(1.0, "end-1c")

        cv2.imwrite(os.path.join(self.db_dir, '{}.jpg'.format(name)), self.register_new_user_capture)

        util.msg_box('Successful!', 'User was registered successfully!')

        self.register_new_user_window.destroy()


if __name__ == "__main__":
    app = App()
    app.start()