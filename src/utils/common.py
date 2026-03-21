import os
import joblib


def save_object(file_path, obj):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    joblib.dump(obj, file_path)


def load_object(file_path):
    return joblib.load(file_path)