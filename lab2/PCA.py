import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from RPCA import *
import shutup

shutup.please()

'''
We have to find out if PCA helps us with training,
so we have to write a function to evaluate.
'''
def evaluator(img_list, lbl_list):
    divide_position = int(0.8 * len(img_list))
    logisticRegr = LogisticRegression(solver='lbfgs')
    logisticRegr.fit(img_list[:divide_position], lbl_list[:divide_position])
    score = logisticRegr.score(img_list[divide_position:], lbl_list[divide_position:])
    return score

# Read the csv file
sample = pd.read_csv("./mnist/mnist_train.csv")
sample.fillna(0, inplace=True)
print("Before applying PCA and RPCA: ")
print("We have " + str(sample.shape[0]) + " samples originally.")

'''
As we know, the first column is what the number is.
Other columns are the grey scale of the picture.
We should extract them.
'''
facts = sample[sample.columns[0]]
pixels = sample.iloc[:, 1:]
prevScore = evaluator(pixels, facts)
print("The score is " + str(prevScore) + ".\n")

# Convert pixels to numpy arrays.
pixels = np.array(pixels)

# Apply PCA
pca = PCA(0.99)
pca.fit(pixels)
pixels_aftr = pca.transform(pixels)
aftrScore = evaluator(pixels_aftr, facts)
print("After applying PCA: ")
print("We have " + str(pca.n_components_) +" samples left.")
print("The score is " + str(aftrScore) + ".\n")

# Apply RPCA
print("Start applying RPCA: ")
rpca = R_pca(pixels)
L, S = rpca.fit(max_iter=1000)
rpcaScore = evaluator(L, facts)
print("After applying RPCA: ")
print("The score is " + str(rpcaScore) + ".")
