import cv2 as cv
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer

def featurize(data_path, out_path, model_path='./dl_models/blaze_face_short_range.tflite'):

    companies = []
    for name in os.listdir(data_path):
        if name[0] != ".":
            companies.append(name)
    

    c_text_dfs = []
    c_image_names_dfs = []
    for c in companies:
        c_data_file_path = os.path.join(data_path, c, "data.csv")
        c_df = pd.read_csv(c_data_file_path)
        c_text_df = c_df.loc[:, ["body", "cta_text", "title"]]
        c_text_df = c_text_df.fillna("")
        c_text_df = c_text_df.agg(" ".join, axis=1)
        c_text_df = c_text_df.str.replace("\n", " ")
        c_text_dfs.append(c_text_df)

        c_image_names_dfs.append("{}/{}/tiled/".format(data_path, c) + c_df.loc[:, "image_name"])

    text_df = pd.concat(c_text_dfs, axis=0)
    text_list = text_df.to_list()
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(text_list)
    print(X.shape)

    image_names_df = pd.concat(c_image_names_dfs, axis=0)
    edge_detection_means_df = image_names_df.apply(edge_detection_mean)
    cluster_vars_df = image_names_df.apply(clustering_variance)

    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceDetectorOptions(base_options=base_options)
    detector = vision.FaceDetector.create_from_options(options)
    face_detected_df = image_names_df.apply(lambda x: face_detection(x, detector))

    image_features_df = pd.DataFrame({"edge_detection_mean":edge_detection_means_df, "clusters_var":cluster_vars_df, "face_detected":face_detected_df})
    print(image_features_df.head())


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

def face_detection(img_path, detector, threshold=0.3):
    image = mp.Image.create_from_file(img_path)
    detection_result = detector.detect(image)

    return 1 if len(detection_result.detections) > 0 and detection_result.detections[0].categories[0].score > threshold else 0
        

if __name__ == "__main__":
    data_path = "./data_take_home"
    featurize(data_path, "")
    pass