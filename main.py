import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.utils import shuffle
from nueralNetwork import NeuralNetwork
from layer import Layer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold, cross_val_score

training_data = pd.read_csv('exoTrain.csv')
testing_data = pd.read_csv('exoTest.csv')

print("Size of Training: ", training_data.shape)
print("Size of Testing:  ", testing_data.shape)

data = pd.concat([training_data, testing_data])
data.reset_index(inplace=True, drop=True)
print("Size of data: ", data.shape)

data_features = data.loc[:, "FLUX.1":"FLUX.3197"]
data_labels = data.loc[:, "LABEL"]

scaler = StandardScaler()
data_features_standardize = scaler.fit_transform(data_features)
print("data_standardize: ", np.round(np.mean(data_features_standardize)), np.std(data_features_standardize), "\n")

pca = PCA(n_components=12)
data_features_standardize_reduced = pca.fit_transform(data_features_standardize)
print("data_features_standardize reduced to ", pca.n_components_, " features\n")

data_features_reduced_df = pd.DataFrame(data=data_features_standardize_reduced,
                                        columns=['principal component ' + str(i) for i in
                                                 range(1, pca.n_components_ + 1)])

data_features_madeup = pd.read_csv('class2madeup.csv')
data_label_madeup = pd.read_csv('class2madeup-labels.csv')

print("data_features_madeup: ", data_features_madeup.shape, "\n")
print("data_label_madeup: ", data_label_madeup.shape, "\n")

new_data_data_madeup = np.vstack((data_features_reduced_df, data_features_madeup.values))

new_data_data_madeup_df = pd.DataFrame(data=new_data_data_madeup,
                                       columns=['principal component ' + str(i) for i in
                                                range(1, pca.n_components_ + 1)])
data_label_madeupNP = np.reshape(data_label_madeup, (-1, 1))
data_labelsNP = np.reshape(data_labels.values, (-1, 1))
new_training_label_madeup = np.vstack((data_labelsNP, data_label_madeupNP))

new_training_label_madeup_df = pd.DataFrame(data=new_training_label_madeup,
                                            columns=['Label'])

print("new_data_data_madeup_df shape(Added new data): ", new_data_data_madeup_df.shape, "\n")
print("new_training_label_madeup_df shape(Added new data): ", new_training_label_madeup_df.shape, "\n")

new_data = pd.concat([new_training_label_madeup_df, new_data_data_madeup_df], axis=1)
print("new_data shape(Added new data): ", new_data.shape, "\n")

new_data_rand = shuffle(new_data)
new_data_rand.reset_index(inplace=True, drop=True)

inputsize = new_data_data_madeup_df.shape[1]
outputsize = 1

print("Input Size: ", inputsize)
print("Output Size: ", outputsize)

planetNet = NeuralNetwork(lr=0.0001, epoch=150)
planetNet.add_layer(Layer(inputsize, inputsize * 2, activation="sigmoid"))
planetNet.add_layer(Layer(inputsize * 2, inputsize * 2, activation="sigmoid"))
planetNet.add_layer(Layer(inputsize * 2, inputsize, activation="sigmoid"))
planetNet.add_layer(Layer(inputsize, outputsize, activation="sigmoid"))

# X = new_data_data_madeup_df.values
# y = new_training_label_madeup_df.values
# scores = cross_val_score(planetNet, X, Y, cv=5)

# k_fold = KFold(n_splits=5)
# print("\n\n")
# fold_scores = []
# cross_val_scores = []
# for i in range(10):
#     j = 1
#     for train_indices, test_indices in k_fold.split(X):
#         print("\n\nCross Fold: ", j)
#         j += 1
#         print('Train: %s | test: %s' % (train_indices, test_indices))
#         planetNet.fit(X[train_indices], y[train_indices])
#         fold_scores.append(planetNet.score(X[test_indices], y[test_indices]))
#
#     cross_val_scores.append(np.mean(fold_scores))

# print(cross_val_scores)
