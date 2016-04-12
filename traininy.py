import numpy as np
import csv
import random
import matplotlib.pyplot as plt
import pylab as pl
from mpl_toolkits.mplot3d import Axes3D

from itertools import cycle
from sklearn import svm, datasets
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import tree

action_list = ['walking', 'sitting', 'running', 'stairs', 'biking']

def plot_features(feature_list):
	X, y = create_sets(action_list)
	print X
	print y

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	x_walking = []
	y_walking = []
	z_walking = []

	x_sitting = []
	y_sitting = []
	z_sitting = []

	x_running = []
	y_running = []
	z_running = []

	x_stairs = []
	y_stairs = []
	z_stairs = []

	x_biking = []
	y_biking = []
	z_biking = []

	for i in range (0, 145):
		if i < 30:
			x_walking.append(X[i][0])
			y_walking.append(X[i][1])
			z_walking.append(X[i][2])
		elif i < 58:
			x_sitting.append(X[i][0])
			y_sitting.append(X[i][1])
			z_sitting.append(X[i][2])
		elif i < 87:
			x_running.append(X[i][0])
			y_running.append(X[i][1])
			z_running.append(X[i][2])
		elif i < 116:
			x_stairs.append(X[i][0])
			y_stairs.append(X[i][1])
			z_stairs.append(X[i][2])
		else:
			x_biking.append(X[i][0])
			y_biking.append(X[i][1])
			z_biking.append(X[i][2])

	ax.scatter(x_walking, y_walking, z_walking, c='r', marker='o')
	ax.scatter(x_sitting, y_sitting, z_sitting, c='b', marker='o')
	ax.scatter(x_running, y_running, z_running, c='g', marker='o')
	ax.scatter(x_stairs, y_stairs, z_stairs, c='y', marker='o')
	ax.scatter(x_biking, y_biking, z_biking, c='m', marker='o')


	ax.set_xlabel(feature_list[0])
	ax.set_ylabel(feature_list[1])
	ax.set_zlabel(feature_list[2])

	plt.show()
    
def create_sets(type_list):  
	X = []
	y = []
	for index, action in enumerate(action_list): 
		for num in range(0,29):
			features = get_features('data/'+ action + '.csv', num)
			X.append(features)
			y.append(index)
	return X, np.array(y)


def get_features(input_csv, num):
	file = open (input_csv, 'r')
	reader = csv.DictReader(file)
	features = []
	for row, line in enumerate(reader):
		if row == num:
			for key in feature_list:
				features.append(float(line[key]))
	return features

def get_score(X, y):
    clf_svm = svm.SVC(gamma=0.001, C=100)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)
    svm_classifier = clf_svm.fit(X_train, y_train)
    y_pred_svm = svm_classifier.predict(X_test)
    a = np.array(y_pred_svm)
    b = np.array(y_test)
    error = np.mean( a != b )
    print 1 - error

    clf_tree = tree.DecisionTreeClassifier()
    tree_classifier = clf_tree.fit(X_train, y_train)
    y_pred_tree = tree_classifier.predict(X_test)
    a = np.array(y_pred_tree)
    error = np.mean( a != b )
    print 1 - error

    clf_log = LogisticRegression()
    log_classifier = clf_log.fit(X_train, y_train)
    y_pred_log = log_classifier.predict(X_test)
    a = np.array(y_pred_log)
    error = np.mean( a != b )
    print  1 - error

feature_list = ["'gravity_x_fft'", "'gravity_y_fft'", "'gravity_z_fft'"]
plot_features(feature_list)
print "\n"

feature_list = ["'user_acc_x_fft'", "'user_acc_y_fft'","'user_acc_z_fft'"]
plot_features(feature_list)
print "\n"

feature_list = ["'user_acc_x_fft_stDev'", "'user_acc_y_fft_stDev'","'user_acc_z_fft_stDev'"]
plot_features(feature_list)
print "\n"

feature_list = ["'attitude_pitch_fft'", "'attitude_pitch_fft_stDev'", "'attitude_pitch_mean'"]
plot_features(feature_list)
print "\n"

feature_list = ["'magnetic_field_x_fft'", "'magnetic_field_y_fft'", "'magnetic_field_z_fft'"]
plot_features(feature_list)
print "\n"

feature_list = ["'rotation_rate_x_fft'", "'rotation_rate_y_fft'", "'rotation_rate_z_fft'"]
plot_features(feature_list)
print "\n"

feature_list = ["'rotation_rate_x_fft_stDev'", "'rotation_rate_y_fft_stDev'", "'rotation_rate_z_fft_stDev'"]
plot_features(feature_list)
print "\n"

feature_list = ["'gravity_x_fft'", "'gravity_y_fft'", "'gravity_z_fft'"]
X_grav, y_grav = create_sets(action_list)
print X_grav
print y_grav
print "Gravity FFT Max"
print "----------------------"
get_score(X_grav, y_grav)
print "\n"

feature_list = ["'user_acc_x_fft'", "'user_acc_y_fft'","'user_acc_z_fft'"]
X, y = create_sets(action_list)
print X
print y
print "User Acceleration FFT Max"
print "----------------------"
get_score(X, y)
print "\n"

feature_list = ["'user_acc_x_fft_stDev'", "'user_acc_y_fft_stDev'","'user_acc_z_fft_stDev'"]
X, y = create_sets(action_list)
print "User Acceleration FFT St Dev"
print "----------------------"
get_score(X, y)
print "\n"

feature_list = ["'attitude_pitch_fft'", "'attitude_pitch_fft_stDev'", "'attitude_pitch_mean'"]
X, y = create_sets(action_list)
print "Attitude Pitch FFT, Attitude Pitch FFT St Dev, Attitude Pitch Mean"
print "----------------------"
get_score(X, y)
print "\n"

feature_list = ["'magnetic_field_x_fft'", "'magnetic_field_y_fft'", "'magnetic_field_z_fft'"]
X, y = create_sets(action_list)
print "Magnetic Field FFT"
print "----------------------"
get_score(X, y)
print "\n"

feature_list = ["'rotation_rate_x_fft'", "'rotation_rate_y_fft'", "'rotation_rate_z_fft'"]
X, y = create_sets(action_list)
print "Rotation Rate FFT"
print "----------------------"
get_score(X, y)
print "\n"

feature_list = ["'rotation_rate_x_fft_stDev'", "'rotation_rate_y_fft_stDev'", "'rotation_rate_z_fft_stDev'"]
X, y = create_sets(action_list)
print "Rotation Rate FFT StDev"
print "----------------------"
get_score(X, y)
print "\n"

