import cv2 as cv
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os
import pandas as pd
from PIL import Image
import pytesseract
import re
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer

def featurize(data_path, out_path, model_path='./dl_models/blaze_face_short_range.tflite'):

    # get list of companies
    companies = []
    for name in os.listdir(data_path):
        if name[0] != ".":
            companies.append(name)
    
    # iterate through csv files and save relevant information (text, image paths)
    c_text_dfs = []
    c_image_names_dfs = []
    c_label_dfs = []
    for c in companies:
        c_data_file_path = os.path.join(data_path, c, "data.csv")
        c_df = pd.read_csv(c_data_file_path)
        c_text_df = c_df.loc[:, ["body", "cta_text", "title", "link_description"]]
        c_text_df = c_text_df.fillna("")
        c_text_df = c_text_df.agg(" ".join, axis=1)
        c_text_df = c_text_df.str.replace("\n", " ")
        c_text_dfs.append(c_text_df)

        c_image_names_dfs.append("{}/{}/tiled/".format(data_path, c) + c_df.loc[:, "image_name"])

        c_label_dfs.append(pd.Series(c_text_df.shape[0] * [c]))
    label_df = pd.concat(c_label_dfs, axis=0)

    # consolidate all relevant text for each ad
    image_names_df = pd.concat(c_image_names_dfs, axis=0)
    # ocr to read text from images
    ocr_text_df = image_names_df.apply(ocr)
    text_df = pd.concat(c_text_dfs, axis=0) + " " + ocr_text_df
    # encode text data using bag-of-words
    text_list = text_df.to_list()
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(text_list).toarray()
    bow_features = {k:X[:, v] for k, v in vectorizer.vocabulary_.items()}
    # proportion of uppercase characters
    prop_uppercase_df = text_df.apply(prop_uppercase)
    # check for user testimony
    user_testimony_df = text_df.apply(user_testimony)
    text_features = {
        "prop_uppercase":prop_uppercase_df,
        #"user_testimony":user_testimony_df
    }

    # quantify visual complexity using canny edge detection
    edge_detection_means_df = image_names_df.apply(edge_detection_mean)
    # quantify color using clustering and within-cluster variance
    cluster_vars_df = image_names_df.apply(clustering_variance)
    # check for human face in ad
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceDetectorOptions(base_options=base_options)
    detector = vision.FaceDetector.create_from_options(options)
    face_detected_df = image_names_df.apply(lambda x: face_detection(x, detector))
    image_features = {
        "edge_detection_mean":edge_detection_means_df,
        "clusters_var":cluster_vars_df,
        "face_detected":face_detected_df
    }

    features_df = pd.DataFrame({"label":label_df, **bow_features, **text_features, **image_features})
    features_df.to_csv(out_path)
    



def prop_uppercase(s):
    return 0 if len(s) == 0 else len(re.findall(r'[A-Z]', s)) / len(s)

def user_testimony(s):
    return 1 if len(re.findall(r'".* .*"', s)) > 0 else 0

def edge_detection_mean(img_path):
    img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
    edges = cv.Canny(img, 100, 200)
    return edges.mean()

def clustering_variance(img_path, n_clusters=4):
    col_img = cv.imread(img_path)
    pixels = col_img.reshape(-1, 3)
    model = KMeans(n_clusters=n_clusters, n_init='auto')
    assignments = model.fit_predict(pixels)
    cluster_vars = np.zeros((n_clusters, ))
    for k in range(n_clusters):
        cluster = pixels[assignments == k, :]
        center = model.cluster_centers_[k, :]
        dists = np.sum((cluster - center[np.newaxis, :])**2, axis=1)
        cluster_vars[k] = dists.mean()

    return cluster_vars.sum()

def face_detection(img_path, detector, threshold=0.8):
    image = mp.Image.create_from_file(img_path)
    detection_result = detector.detect(image)

    return 1 if len(detection_result.detections) > 0 and detection_result.detections[0].categories[0].score > threshold else 0

def ocr(img_path):
    return pytesseract.image_to_string(Image.open(img_path)).replace("\n", " ")
        

if __name__ == "__main__":
    data_path = "./data_take_home"
    featurize(data_path, "features.csv")