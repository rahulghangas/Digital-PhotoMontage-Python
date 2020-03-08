import threading
import tkinter as Tkinter
from tkinter import filedialog

import cv2
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as Image
import PIL.ImageTk as ImageTk
from functools import partial
import os

import photoMontage3

image = None
stop = False
low_power_mode = False

class App(Tkinter.Tk):
    def __init__(self, width=320, height=240):
        super().__init__()
        self.lock = threading.Lock()
        self.haar_cascade_face = cv2.CascadeClassifier(
            "haar_cascade/haarcascade_frontalface_default.xml"
        )
        self.erase_mode = False
        self.cursors = ("", "plus")
        self.width = width
        self.height = height

        self.mask = np.ones((height, width)) * -1

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
        

    def flush(self, event):
        return "break"

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
                if not self.erase_mode:
                    im[
                        event.y - 2 : event.y + 2, event.x - 2 : event.x + 2
                    ] = self.colors[image_index]
                    self.mask[
                        event.y - 2 : event.y + 2, event.x - 2 : event.x + 2
                    ] = image_index
                else:
                    im[
                        event.y - 5 : event.y + 5, event.x - 5 : event.x + 5
                    ] = self.images[image_index][
                        event.y - 5 : event.y + 5, event.x - 5 : event.x + 5
                    ]
                    self.mask[event.y - 2 : event.y + 2, event.x - 2 : event.x + 2] = -1

                im = Image.fromarray(im)
                imgtk = ImageTk.PhotoImage(image=im)
                self.labels[image_index].configure(image=imgtk)
                self.labels[image_index].image = imgtk

            except:
                pass

    def key_press_event(self, event):
        if str(event.type) == "KeyPress" and event.char == "e":
            self.erase_mode = not self.erase_mode
            for i in range(4):
                self.labels[i].config(cursor=self.cursors[int(self.erase_mode)])

        elif str(event.type) == "KeyPress" and event.char in ("1", "2", "3", "4"):
            global image
            image_index = int(event.char) - 1
            self.orig_images[image_index] = image[..., ::-1]
            img = cv2.resize(image, (self.width, self.height))
            img = img[..., ::-1]
            self.images[image_index] = img
            self.images_masked[image_index] = self.encase(img)
            im = Image.fromarray(self.images_masked[image_index])
            imgtk = ImageTk.PhotoImage(image=im)
            self.labels[image_index].configure(image=imgtk)
            self.labels[image_index].image = imgtk

    def load_image(self, image_index):
        filename = filedialog.askopenfilename()
        if not filename or not os.path.exists(filename):
            return
        img = cv2.imread(filename)
        self.orig_images[image_index] = img
        img = cv2.resize(img, (self.width, self.height))
        img = img[..., ::-1]
        self.images[image_index] = img
        self.images_masked[image_index] = self.encase(img)
        im = Image.fromarray(self.images_masked[image_index])
        imgtk = ImageTk.PhotoImage(image=im)
        self.labels[image_index].configure(image=imgtk)
        self.labels[image_index].image = imgtk

    def encase(self, img):
        img_copy = np.copy(img)
        faces_rects = self.haar_cascade_face.detectMultiScale(
            img_copy, scaleFactor=1.05, minNeighbors=5
        )
        for (x, y, w, h) in faces_rects:
            cv2.rectangle(
                img_copy,
                (int(x + w / 5), int(y + h / 5)),
                (int(x + w - w / 5), int(y + h - h / 5)),
                (0, 255, 0),
                2,
            )

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
