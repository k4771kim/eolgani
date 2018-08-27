import os
import numpy as np
from math import sqrt
from sklearn import neighbors
from os import listdir
from os.path import isdir, join, isfile, splitext
import pickle
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
import face_recognition
from face_recognition import face_locations
from glob import glob
import cv2
from sklearn.svm import SVC

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def train(train_dir, model_save_path = "", n_neighbors = None, knn_algo = 'ball_tree', verbose=False, mode =""):
    X = []
    y = []
    for class_dir in listdir(train_dir):
        if not isdir(join(train_dir, class_dir)):
            continue
        for img_path in sorted(glob(join(train_dir, class_dir, '*'))):
            image = face_recognition.load_image_file(img_path)
            faces_bboxes = face_locations(image)
            if len(faces_bboxes) != 1:
                if verbose:
                    print("image {} not fit for training: {}".format(img_path, "didn't find a face" if len(faces_bboxes) < 1 else "found more than one face"))
                continue
            X.append(face_recognition.face_encodings(image, known_face_locations=faces_bboxes)[0])
            y.append(class_dir)

    if mode == "svm":
        my_model = SVC(kernel = 'linear', probability=True)
        my_model.fit(X,y)
    
    elif mode == "knn":
        if n_neighbors is None:
            n_neighbors = int(round(sqrt(len(X))))
            if verbose:
                print("Chose n_neighbors automatically as:", n_neighbors)

        my_model = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
        my_model.fit(X, y)

    if model_save_path != "":
        with open(model_save_path, 'wb') as f:
            pickle.dump(my_model, f)
    return my_model

def predict(X_img_path, model = None, model_save_path ="", DIST_THRESH = .4,mode=""):
    if not isfile(X_img_path) or splitext(X_img_path)[1][1:] not in ALLOWED_EXTENSIONS:
        raise Exception("invalid image path: {}".format(X_img_path))

    if model is None and model_save_path == "":
        raise Exception("must supply knn classifier either thourgh knn_clf or model_save_path")

    if model is None:
        with open(model_save_path, 'rb') as f:
            model = pickle.load(f)

    X_img = face_recognition.load_image_file(X_img_path) # rgb
    print (X_img.shape)
    
    X_faces_loc = face_locations(X_img) # (top, right, bottom, left)
    if len(X_faces_loc) == 0:
        return []
    print (X_faces_loc[0])
    start = cv2.getTickCount()
    faces_encodings = face_recognition.face_encodings(X_img, known_face_locations=X_faces_loc)

    #####################
    if mode == "svm":
        predictions = model.predict_proba(faces_encodings)
        print(predictions)
        best_class_indices = np.argmax(predictions, axis=1)
        print(best_class_indices)
        best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
        print(best_class_probabilities)

        time = (cv2.getTickCount() - start) / cv2.getTickFrequency() * 1000
        print('prediction time: %.2fms'%time)
        is_recognized = [best_class_indices == 1 for i in range(len(X_faces_loc))]
        print(is_recognized)
        return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(model.predict(faces_encodings), X_faces_loc, is_recognized)]
    #closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
    #is_recognized = [closest_distances[i] <= DIST_THRESH for i in range(len(X_faces_loc))]
    
    #####################
    elif mode == "knn":
        closest_distances = model.kneighbors(faces_encodings, n_neighbors=1)
        print(closest_distances)
        time = (cv2.getTickCount() - start) / cv2.getTickFrequency() * 1000
        print('prediction time: %.2fms'%time)
        is_recognized = [closest_distances[0][i][0] <= DIST_THRESH for i in range(len(X_faces_loc))]
    
    # predict classes and cull classifications that are not with high confidence
        return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(model.predict(faces_encodings), X_faces_loc, is_recognized)]
    


def draw_preds(img_path, preds):
    source_img = Image.open(img_path).convert("RGBA")
    draw = ImageDraw.Draw(source_img)
    for pred in preds:
        loc = pred[1]
        name = pred[0]
        # (top, right, bottom, left) => (left,top,right,bottom)
        draw.rectangle(((loc[3], loc[0]), (loc[1],loc[2])), outline="red")
        # draw.text((loc[3], loc[0] - 30), name, font=ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', 30))
        draw.text((loc[3], loc[0] - 21), name, font=ImageFont.truetype('./BMDOHYEON_TTF.TTF', 20))
    source_img.show()

if __name__ == "__main__":
    knn_clf = train("./data/train", model_save_path='./models/fr_knn.pkl',mode="knn")
    for img_path in listdir("./data/test"):
        
        # preds = predict(join("./data/test", img_path), knn_clf=knn_clf)
        preds = predict(join("./data/test", img_path), model_save_path='./models/fr_knn.pkl', DIST_THRESH=0.4,mode="knn")
        
        print(os.path.basename(img_path), preds)
        # draw_preds(join("./data/test", img_path), preds)