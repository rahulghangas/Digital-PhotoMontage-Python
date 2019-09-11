import numpy as np
import cv2
import tkinter as Tkinter
import PIL.ImageTk as ImageTk
import PIL.Image as Image
import threading
import photoMontage3
import matplotlib.pyplot as plt

image = None
stop = False

class App(Tkinter.Tk): 
    def __init__(self, width=320, height=240): 
        super().__init__()
        self.lock = threading.Lock()
        self.im_counter = 0
        self.width = width
        self.height = height
        
        self.mask = np.ones((height, width)) * -1
        
        self.colors = ((255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0))
        
        self.images = []
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        for i in range(1,5):
            string = 'Press ' + str(i) + ' to capture image'

            empty = cv2.putText(np.zeros((height, width, 3), dtype= np.uint8), string , 
                        (height // 3, width //3), 
                        self.font, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
            self.images += [empty]
            
        self.labels = []
        for i in range(len(self.images)):
            im = Image.fromarray(self.images[i])
            imgtk = ImageTk.PhotoImage(image=im) 
            label = Tkinter.Label(self, image=imgtk, borderwidth=2, relief="solid")
            label.photo = imgtk
            label.bind("<Button-1>", self.print_event)
            label.bind("<B1-Motion>", self.print_event) 
            label.grid(row=i//2,column=i%2,padx=0, pady=0)
            self.labels += [label]
            
        self.button = Tkinter.Button(self, text ="GraphCut", command = self.callback).grid(row = 2, column = 0)
        self.button = Tkinter.Button(self, text ="Save Images", command = self.save_im_callback).grid(row = 2, column = 1)

        self.images_masked = self.images.copy()
        print("put images")
        
    def callback(self):
        self.lock.acquire()
        z = photoMontage3.solve(np.array(self.images, dtype=np.int32), np.array(self.mask, dtype=np.int32))
        self.lock.release()
        global stop
        stop = True
        plt.figure()
        plt.imshow(z)

        merged_im = np.zeros(self.images[0].shape, dtype=np.uint8)
        for i in range(len(self.images)):
            merged_im[z == i] = self.images[i][z == i]
        print(merged_im)
        plt.figure()
        plt.imshow(merged_im)
        
        plt.show()

    def save_im_callback(self):
        for i in range(len(self.images)):
            cv2.imwrite('./saved_imgs/image_' + str(i) +'.jpg', self.images[i][..., ::-1])

    def print_event(self, event):
        if self.im_counter > 3:
            if str(event.type) == 'Motion':
                try:
                    if str(event.widget) == '.!label':
                        im = self.images_masked[0]
                        im[event.y-2:event.y+2, event.x-2:event.x+2] = self.colors[0]
                        self.mask[event.y-2:event.y+2, event.x-2:event.x+2] = 0
                        im = Image.fromarray(im)
                        imgtk = ImageTk.PhotoImage(image=im)
                        self.labels[0].configure(image=imgtk)
                        self.labels[0].image = imgtk
                    elif str(event.widget) == '.!label2':
                        im = self.images_masked[1]
                        im[event.y-2:event.y+2, event.x-2:event.x+2] = self.colors[1]
                        self.mask[event.y-2:event.y+2, event.x-2:event.x+2] = 1
                        im = Image.fromarray(im)
                        imgtk = ImageTk.PhotoImage(image=im)
                        self.labels[1].configure(image=imgtk)
                        self.labels[1].image = imgtk
                    elif str(event.widget) == '.!label3':
                        im = self.images_masked[2]
                        im[event.y-2:event.y+2, event.x-2:event.x+2] = self.colors[2]
                        self.mask[event.y-2:event.y+2, event.x-2:event.x+2] = 2
                        im = Image.fromarray(im)
                        imgtk = ImageTk.PhotoImage(image=im)
                        self.labels[2].configure(image=imgtk)
                        self.labels[2].image = imgtk
                    elif str(event.widget) == '.!label4':
                        im = self.images_masked[3]
                        im[event.y-2:event.y+2, event.x-2:event.x+2] = self.colors[3]
                        self.mask[event.y-2:event.y+2, event.x-2:event.x+2] = 3
                        im = Image.fromarray(im)
                        imgtk = ImageTk.PhotoImage(image=im)
                        self.labels[3].configure(image=imgtk)
                        self.labels[3].image = imgtk
                except:
                    pass
        else:
            if str(event.type) == 'ButtonPress':
                global image
                img = cv2.resize(image, (self.width, self.height))
                img = img[..., ::-1]
                self.images[self.im_counter] = img
                self.images_masked[self.im_counter] = img.copy()
                im = Image.fromarray(img)
                imgtk = ImageTk.PhotoImage(image=im) 
                self.labels[self.im_counter].configure(image=imgtk)
                self.labels[self.im_counter].image = imgtk
                self.im_counter += 1
    

class cvRead (threading.Thread):
    def __init__(self, threadID, name, counter):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.counter = counter
    
    def run(self):
        cap = cv2.VideoCapture(0)
        global stop        
        while not stop:
            global image
            ret, image = cap.read()
            

thread1 = cvRead(1, "Thread-1", 1)
thread1.start()
app = App() 
app.mainloop()
stop = True
