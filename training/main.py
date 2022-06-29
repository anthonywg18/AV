import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import sklearn
import imgaug
print(tf.__version__)
print(cv2.__version__)
print(np.__version__)
print(pd.__version__)
print(sklearn.__version__)
print(imgaug.__version__)

print('Setting Up')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from sklearn.model_selection import train_test_split
from utlis import *

path = 'DataCollected'
data = importDataInfo(path)
print(data.head())
data = balanceData(data,display=True)
