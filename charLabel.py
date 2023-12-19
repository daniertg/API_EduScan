from tensorflow.keras.models import load_model

import joblib

digits = '0123456789'
letters = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWZYZ'
symbols = '!"#$%&' + "'" +'()*+,-./:;<=>?@[\]^_`{|}~'
char_list = digits + letters + symbols
char_list = [ch for ch in char_list]

joblib.dump(char_list, "EduScan.pkl")