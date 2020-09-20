import face_recognition as fr
cimport cython
from keras.preprocessing.image import img_to_array
import imutils
import cv2
from keras.models import load_model
import numpy as np

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def autoMontage(imgs):
    img1 = imgs[0]
    img2 = imgs[1]
    img3 = imgs[2]
    img4 = imgs[3]
    boxes_img1 = fr.face_locations(img1)
    boxes_img2 = fr.face_locations(img2)
    boxes_img3 = fr.face_locations(img3)
    boxes_img4 = fr.face_locations(img4)

    faces_map = dict()

    cdef int i, j, f, top, bottom, left, right, top2, bottom2, left2, right2

    print("Recognizing people in image 1")
    for i in range(len(boxes_img1)):
        faces_map[i] = [(0, boxes_img1[i])]

    print("Recognizing people and cross-referencing in image 2")

    defer_add_faces_map = []
    for i in range(len(boxes_img2)):
        box = boxes_img2[i]
        for f in faces_map:
            known_faces = faces_map[f]
            done = False
            for j in range(len(known_faces)):
                face = known_faces[j]
                top, right, bottom, left = face[1]
                face_img = imgs[face[0]][top:bottom,left:right, :]
                top2, right2, bottom2, left2 = box
                face_img_curr = img2[top2:bottom2,left2:right2, :]
                encoding1 = fr.face_encodings(face_img, model='large')
                encoding2 = fr.face_encodings(face_img_curr, model='large')
                if encoding1 and encoding2 and fr.compare_faces(encoding1, encoding2[0])[0]:
                    done = True
                    defer_add_faces_map.append((f, (1, box)))
                    break

            if done:
                break
        else:
            defer_add_faces_map.append((f, (1, box)))

    for i in range(len(defer_add_faces_map)):
        key, val = defer_add_faces_map[i]
        if key in faces_map:
            faces_map[key].append(val)
        else:
            faces_map[key] = [val]

    print("Recognizing people and cross-referencing in image 3")
    defer_add_faces_map = []
    for i in range(len(boxes_img3)):
        box = boxes_img3[i]
        for f in faces_map:
            known_faces = faces_map[f]
            done = False
            for j in range(len(known_faces)):
                face = known_faces[j]
                top, right, bottom, left = face[1]
                face_img = imgs[face[0]][top:bottom,left:right, :]
                top2, right2, bottom2, left2 = box
                face_img_curr = img3[top2:bottom2,left2:right2, :]
                encoding1 = fr.face_encodings(face_img, model='large')
                encoding2 = fr.face_encodings(face_img_curr, model='large')
                if encoding1 and encoding2 and fr.compare_faces(encoding1, encoding2[0])[0]:
                    done = True
                    defer_add_faces_map.append((f, (1, box)))
                    break

            if done:
                break
        else:
            defer_add_faces_map.append((f, (1, box)))

    for i in range(len(defer_add_faces_map)):
        key, val = defer_add_faces_map[i]
        if key in faces_map:
            faces_map[key].append(val)
        else:
            faces_map[key] = [val]

    print("Recognizing people and cross-referencing in image 4")
    defer_add_faces_map = []
    for i in range(len(boxes_img4)):
        box = boxes_img4[i]
        for f in faces_map:
            known_faces = faces_map[f]
            done = False
            for j in range(len(known_faces)):
                face = known_faces[j]
                top, right, bottom, left = face[1]
                face_img = imgs[face[0]][top:bottom,left:right, :]
                top2, right2, bottom2, left2 = box
                face_img_curr = img4[top2:bottom2,left2:right2, :]
                encoding1 = fr.face_encodings(face_img, model='large')
                encoding2 = fr.face_encodings(face_img_curr, model='large')
                if encoding1 and encoding2 and fr.compare_faces(encoding1, encoding2[0])[0]:
                    done = True
                    defer_add_faces_map.append((f, (1, box)))
                    break

            if done:
                break
        else:
            defer_add_faces_map.append((f, (1, box)))

    for i in range(len(defer_add_faces_map)):
        key, val = defer_add_faces_map[i]
        if key in faces_map:
            faces_map[key].append(val)
        else:
            faces_map[key] = [val]

    print("People recognized and cross referenced. Recognizing emotions")
    emotion_model_path = 'models/_mini_XCEPTION.102-0.66.hdf5'

    # hyper-parameters for bounding boxes shape
    # loading models
    emotion_classifier = load_model(emotion_model_path, compile=False)
    EMOTIONS = ["angry" ,"disgust","scared", "happy", "sad", "surprised",
     "neutral"]
    emotions_map = {"scared" : 1, "disgust" : 2, "sad" : 3, "angry" : 4, "neutral" : 5, "surprised" : 6, "happy" : 7}

    new_masks = []

    for person in faces_map.values():
        max_val = -1
        prob = 0
        info = None
        for i in range(len(person)):
            face = person[i]
            top, right, bottom, left = face[1]
            face_img = imgs[face[0]][top:bottom,left:right, :]
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            # Extract the ROI of the face from the grayscale image, resize it to a fixed 28x28 pixels, and then prepare
            # the ROI for classification via the CNN
            roi = gray
            roi = cv2.resize(roi, (64, 64))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)


            preds = emotion_classifier.predict(roi)[0]
            emotion_probability = np.max(preds)
            label = EMOTIONS[preds.argmax()]

            label_val = emotions_map[label]

            if label_val > max_val:
                max_val = label_val
                prob = emotion_probability
                info = face
            elif label_val == max_val and emotion_probability > prob:
                prob = emotion_probability
                info = face

        new_masks.append(info)

    print(new_masks)