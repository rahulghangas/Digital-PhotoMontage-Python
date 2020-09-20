import threading
import tkinter as Tkinter
from tkinter import filedialog

import cv2
import face_recognition
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as Image
import PIL.ImageTk as ImageTk
from functools import partial
import os
import sys
# from keras.models import load_model

import photoMontage3

image = None
stop = False
low_power_mode = False

class App(Tkinter.Tk):
    def __init__(self, width=320, height=240):
        super().__init__()
        self.lock = threading.Lock()
        self.erase_mode = False
        self.cursors = ("", "plus")
        self.width = width
        self.height = height
        # self.model = load_model("./emotion_detector_models/model_v6_23.hdf5")
        # self.label_map = {0 : 'Angry', 5 : 'Sad', 4 : 'Neutral', 1 : 'Disgust', 6 : 'Surprise', 2 : 'Fear', 3 : 'Happy'}

        self.mask = None

        self.colors = ((255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0))
        self.color_names = ('RED', 'GREEN', 'BLUE', 'YELLOW')

        self.orig_images = [None] * 4
        self.images = []
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        for i in range(1, 5):
            string = "Press " + str(i) + " to capture image"

            empty = cv2.putText(
                np.zeros((height, width, 3), dtype=np.uint8),
                string,
                (height // 3, width // 3),
                self.font,
                0.5,
                (255, 0, 0),
                2,
                cv2.LINE_AA,
            )
            self.images += [empty]

        self.labels = []
        for i in range(len(self.images)):
            im = Image.fromarray(self.images[i])
            imgtk = ImageTk.PhotoImage(image=im)
            label = Tkinter.Label(self, image=imgtk, background = self.color_names[i], borderwidth = 1, relief = 'groove')
            label.photo = imgtk
            # label.bind("<Key>", self.input_event)
            label.bind("<B1-Motion>", self.draw_event)
            label.grid(row=(i // 2) * 2, column=i % 2, ipadx=4, ipady=4)
            label.focus_set()
            self.labels += [label]

        self.key_event_binding = self.bind("<Key>", self.key_press_event)
        self.buttons = []
        self.buttons.append(Tkinter.Button(self, text="GraphCut", command=self.callback))
        self.buttons[-1].grid(row=4, column=0)
        self.buttons.append(Tkinter.Button(
            self, text="Save Images", command=self.save_im_callback
        ))
        self.buttons[-1].grid(row=4, column=1)
        self.buttons.append(Tkinter.Button(
            self, text="Load Image 1", command=partial(self.load_image, 0)
        ))
        self.buttons[-1].grid(row=1, column=0)
        self.buttons.append(Tkinter.Button(
            self, text="Load Image 2", command=partial(self.load_image, 1)
        ))
        self.buttons[-1].grid(row=1, column=1)
        self.buttons.append(Tkinter.Button(
            self, text="Load Image 3", command=partial(self.load_image, 2)
        ))
        self.buttons[-1].grid(row=3, column=0)
        self.buttons.append(Tkinter.Button(
            self, text="Load Image 4", command=partial(self.load_image, 3)
        ))
        self.buttons[-1].grid(row=3, column=1)

        self.images_masked = [None] * len(self.images)

    def callback(self):
        global low_power_mode
        for button in self.buttons:
            button.config(state=Tkinter.DISABLED)
            button.update()
        low_power_mode = True
        z = photoMontage3.solve(
            np.array(self.images, dtype=np.int32), np.array(self.mask, dtype=np.int32)
        )
        plt.figure()
        plt.imshow(z)

        print(z.shape)
        print(self.mask.shape)
        print(self.images[0].shape)

        merged_im = np.zeros(self.images[0].shape, dtype=np.uint8)
        for i in range(len(self.images)):
            merged_im[z == i] = self.images[i][z == i]
        plt.figure()
        plt.imshow(merged_im)
        plt.show()

        low_power_mode = False
        
        for button in self.buttons:
            button.config(state=Tkinter.NORMAL)
            button.update()

    def cleanup(self):
        pass

    def flush(self):
        self.orig_images = [None] * 4
        self.images = []
        self.mask = None
        for i in range(1, 5):
            string = "Press " + str(i) + " to capture image"

            empty = cv2.putText(
                np.zeros((self.height, self.width, 3), dtype=np.uint8),
                string,
                (self.height // 3, self.width // 3),
                self.font,
                0.5,
                (255, 0, 0),
                2,
                cv2.LINE_AA,
            )
            self.images += [empty]
            im = Image.fromarray(empty)
            imgtk = ImageTk.PhotoImage(image=im)
            self.labels[i-1].configure(image=imgtk)
            self.labels[i-1].image = imgtk
        self.images_masked = [None] * len(self.images)

    def save_im_callback(self):
        if not os.path.exists('./saved_imgs'):
            os.makedirs('./saved_imgs')
        for i in range(len(self.images)):
            if self.images_masked[i] is None:
                continue
            cv2.imwrite(
                "./saved_imgs/image_" + str(i) + ".jpg", self.orig_images[i][..., ::-1]
            )

    def draw_event(self, event):
        if str(event.type) == "Motion":
            try:
                if str(event.widget) == ".!label":
                    image_index = 0
                elif str(event.widget) == ".!label2":
                    image_index = 1
                elif str(event.widget) == ".!label3":
                    image_index = 2
                elif str(event.widget) == ".!label4":
                    image_index = 3

                if self.images_masked[image_index] is None:
                    return

                im = self.images_masked[image_index]
                scale_factor_x = im.shape[1] / self.width
                scale_factor_y = im.shape[0] / self.height
                x_pixel = event.x * scale_factor_x
                y_pixel = event.y * scale_factor_y
                stroke_width = 3 * scale_factor_x
                stroke_height = 3 * scale_factor_y
                erase_width = 5 * scale_factor_x
                erase_height = 5 * scale_factor_y

                if not self.erase_mode:
                    im[
                        int(y_pixel - stroke_height) : int(y_pixel + stroke_height),
                        int(x_pixel - stroke_width)  : int(x_pixel + stroke_width)
                    ] = self.colors[image_index]
                    self.mask[
                        int(y_pixel - stroke_height) : int(y_pixel + stroke_height),
                        int(x_pixel - stroke_width)  : int(x_pixel + stroke_width)
                    ] = image_index
                else:
                    im[
                        int(y_pixel - erase_height) : int(y_pixel + erase_height),
                        int(x_pixel - erase_width)  : int(x_pixel + erase_width)
                    ] = self.images[image_index][
                        int(y_pixel - erase_height) : int(y_pixel + erase_height),
                        int(x_pixel - erase_width)  : int(x_pixel + erase_width)
                    ]
                    self.mask[int(y_pixel - erase_height) : int(y_pixel + erase_height),
                              int(x_pixel - erase_width)  : int(x_pixel + erase_width)] = -1

                im = Image.fromarray(cv2.resize(im, (self.width, self.height)))
                imgtk = ImageTk.PhotoImage(image=im)
                self.labels[image_index].configure(image=imgtk)
                self.labels[image_index].image = imgtk

            except:
                e = sys.exc_info()[0]
                print(e)

    def key_press_event(self, event):
        if str(event.type) == "KeyPress" and event.char == "e":
            self.erase_mode = not self.erase_mode
            for i in range(4):
                self.labels[i].config(cursor=self.cursors[int(self.erase_mode)])

        elif str(event.type) == "KeyPress" and event.char in ("1", "2", "3", "4"):
            global image
            image_index = int(event.char) - 1

            if self.mask is None:
                self.mask = np.ones((image.shape[0], image.shape[1])) * -1
            elif (image.shape[0], image.shape[1]) != self.mask.shape:
                print("Images have non-uniform shapes. Resetting images")
                self.flush()
            else:
                self.mask[self.mask == image_index] = -1

            self.orig_images[image_index] = image[..., ::-1]
            # img = cv2.resize(image, (self.width, self.height))
            img = image[...,::-1]
            self.images[image_index] = img
            self.images_masked[image_index] = self.encase(img)
            im = Image.fromarray(cv2.resize(self.images_masked[image_index], (self.width, self.height)))
            imgtk = ImageTk.PhotoImage(image=im)
            self.labels[image_index].configure(image=imgtk)
            self.labels[image_index].image = imgtk

    def load_image(self, image_index):
        filename = filedialog.askopenfilename()
        if not filename or not os.path.exists(filename):
            return
        img = cv2.imread(filename)

        if self.mask is None:
            self.mask = np.ones((img.shape[0], img.shape[1])) * -1
        elif (img.shape[0], img.shape[1]) != self.mask.shape:
            print("Images have non-uniform shapes. Resetting images")
            self.flush()
        else:
            self.mask[self.mask == image_index] = -1

        self.orig_images[image_index] = img[..., ::-1]
        # img = cv2.resize(img, (self.width, self.height))
        img = img[..., ::-1]
        self.images[image_index] = img
        self.images_masked[image_index] = self.encase(img)
        im = Image.fromarray(cv2.resize(self.images_masked[image_index], (self.width, self.height)))
        imgtk = ImageTk.PhotoImage(image=im)
        self.labels[image_index].configure(image=imgtk)
        self.labels[image_index].image = imgtk

    def encase(self, img):
        # scale_percent = 100
        # width = int(img.shape[1] * scale_percent / 100)
        # height = int(img.shape[0] * scale_percent / 100)
        # dim = (width, height)
        # img_copy = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        img_copy = np.copy(img)
        bBoxes1 = face_recognition.face_locations(img_copy, model='hog')
        for bb in bBoxes1:
            top, right, bottom, left = bb
            cv2.rectangle(
                img_copy,
                (left, top),
                (right, bottom),
                (0, 255, 0),
                2)

        return img_copy


class cvRead(threading.Thread):
    def __init__(self, threadID, name, counter):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.counter = counter

    def run(self):
        global cap
        cap = cv2.VideoCapture(0)
        global stop, image, low_power_mode
        while not stop:
            ret = cap.grab()
            if not low_power_mode:
                ret, image = cap.retrieve()


thread1 = cvRead(1, "Thread-1", 1)
thread1.start()
app = App()
app.mainloop()
stop = True
