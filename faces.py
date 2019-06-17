import tkinter
import cv2
import PIL.Image, PIL.ImageTk
import time
import dlib
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

class App:
    def __init__(self, window, window_title, video_source=0):
        self.window = window
        self.window.title(window_title)
        self.video_source = video_source
        self.vid = MyVideoCapture(self.video_source)
        self.canvas = tkinter.Canvas(window, width = 500, height =500,borderwidth=5)
        self.canvas.pack()
        self.btn_snapshot=tkinter.Button(window, text="Snap", width=13, command=self.snapshot)
        self.btn_snapshot.pack(anchor=tkinter.NW, expand=True)
        self.btn_points=tkinter.Button(window,text="PREDICT",width=13,command=self.points)
        self.btn_points.pack(anchor=tkinter.NW, expand=True)

        
        self.delay = 1
        self.update()
        self.window.mainloop()

        #capture
    def points(self):
        ret, frame = self.vid.get_frame()
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        img = cv2.imread("frame.jpg")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray,1.3,3)
        for (x,y,w,h) in faces:
            cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 4)
            area=(x,y,w,h)
            #print(area)
            crop_img = gray[y:y+h, x:x+w]
            cv2.imwrite("f.png",crop_img)
            #cv2.imshow('imge',img)
            k = cv2.waitKey(1)
            if k == 27:
                break
            model = load_model('model2x')
            file = 'f.png'
            true_image = image.load_img(file)
            img = image.load_img(file, grayscale=True, target_size=(48, 48))

            x = image.img_to_array(img)
            x = np.expand_dims(x, axis = 0)

            x /= 255

            custom = model.predict(x)
            print(custom[0])

            def emotion_analysis(emotions):
                objects = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
                y_pos = np.arange(len(objects))
                
                plt.bar(y_pos, emotions, align='center', alpha=0.5)
                plt.xticks(y_pos, objects)
                plt.ylabel('percentage')
                plt.title('emotion')
                
                plt.show()
            emotion_analysis(custom[0])

            x = np.array(x, 'float32')
            x = x.reshape([48, 48]);

            plt.gray()
            #plt.imshow(true_image)
            plt.show()          #img.release()

        cv2.destroyAllWindows()
    def snapshot(self):
        ret, frame = self.vid.get_frame()
        print(frame)
        if ret:
            cv2.imwrite("frame" + ".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            
            #frame update
    def update(self):
        ret, frame = self.vid.get_frame()
        if ret:
            self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
            self.canvas.create_image(0, 0, image = self.photo, anchor = tkinter.NW)
            self.window.after(self.delay, self.update)

            #video stream
class MyVideoCapture:
    def __init__(self, video_source=0):
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)
            self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
            self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
    def get_frame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                return (ret, None)
        else:
            return (ret, None)
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()
            #self.vid.destroyAllWindows()
            #break
App(tkinter.Tk(), "FED")