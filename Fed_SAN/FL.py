from itertools import count
from re import sub
import numpy as np
import cv2
import os
import random
from imutils import paths
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
from operator import add


class Model:
	def make_model(self,shape, classes):
		model = np.zeros((shape,))
		return model

def load_dataset(paths):

	data = []
	labels = []

	# loop over the input images
	for (i, imgpath) in enumerate(paths):

		# load the image and extract the class labels
		im_gray = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)
		image = np.array(im_gray).flatten()/255
		image = np.append(image,np.array([1]))
		label = imgpath.split(os.path.sep)[-2]

		#print(label)
		# scale the image to [0, 1] and add to list
		data.append(image)
		labels.append(label)

	# return a tuple of the data and labels
	return data, labels



def make_clients(images_data, labels, num_clients,labels_actual):

	# list of client names
	client_names = ['{}_{}'.format('client', i+1) for i in range(num_clients)]
	#print(type(labels))
	# labels_actual = np.array(labels_actual)
	# ii = np.where(labels_actual == '1')[0]
	# print(ii)
	print(len(labels_actual))
	data = list(zip(images_data, labels))
	random.shuffle(data)

	# sub-dividing the train set among n clients
	size = len(data) // num_clients
	sub_data_list = [data[i:i + size] for i in range(0, size*num_clients, size)]

	# returning the sub train dataset for each client
	return {client_names[i] : sub_data_list[i] for i in range(len(client_names))}



def data_into_batches(sub_data, batch_size):

	data, label = zip(*sub_data)

	print(len(data))
	exit()
	dataset = tf.data.Dataset.from_tensor_slices((list(data), list(label)))
	return dataset.shuffle(len(label)).batch(batch_size)


def scaling_wts(n_samples,wts,nk_samples):

	scale_factor = nk_samples/n_samples
	scaled_wts = []
	for i in range(len(wts)):
		scaled_wts.append(scale_factor*wts[i])

	return scaled_wts

def test_model(X_test,Y_test,model):

	cross_entropy = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
	logits = model.predict(X_test)
	print(logits)
	print(Y_test)
	loss = cross_entropy(Y_test, logits)
	acc = accuracy_score(tf.argmax(logits, axis=1), tf.argmax(Y_test, axis=1))

	return acc*100, loss

def sum_scaled_wts(scaled_weight_list):

	avg_grad = [0]*len(scaled_weight_list[0])

	for wts in scaled_weight_list:
		print(wts)
		avg_grad = list(map(add, avg_grad, list(wts)))
		
	return avg_grad

def make_clients_noniid(images_data, labels, num_clients,labels_actual):
	d = {}
	for i in range(10):
		if(i not in d):
			l=[]
			d[i]=l

	for i in range(len(labels_actual)):
		if(int(labels_actual[i]) in d):
			#print(type(images_data[i]))
			d[int(labels_actual[i])].append(images_data[i])
			if(i==0):
				print(images_data[i])
				print(labels_actual[i])
				print(d)

	final_images_data = []
	final_labels_actual = []
	for i in d:
		lx = [i]*len(d[i])
		final_labels_actual = final_labels_actual + lx
		final_images_data = final_images_data + d[i]

	print(final_labels_actual)

	label_expand = LabelBinarizer()
	final_labels = label_expand.fit_transform(final_labels_actual)

	print(final_labels)
	print(len(final_images_data))

	client_names = ['{}_{}'.format('client', i+1) for i in range(num_clients)]
	size = len(final_labels_actual)//num_clients
	print("size "+str(size))
	labels_tuple = []
	final_labels_actual = np.array(final_labels_actual)
	for i in range(10):

		list_i = np.where(final_labels_actual == i)[0]
		print(list_i)          
		tuple_i = (list_i[0],list_i[len(list_i)-1])
		labels_tuple.append(tuple_i)

	subdata_list = []

	cc = 0
	while(len(subdata_list)<len(client_names)):

		if(cc>len(labels_tuple)-2):
			cc=0
		else:
			images_data_i = final_images_data[labels_tuple[cc][0]:labels_tuple[cc+1][1]+1]
			labels_i = final_labels[labels_tuple[cc][0]:labels_tuple[cc+1][1]+1]
			data_i = list(zip(images_data_i, labels_i))
			random.shuffle(data_i)
			subdata_list.append(data_i[:size])
		cc+=1

	return {client_names[i] : subdata_list[i] for i in range(len(client_names))}

def loss_prime(y,y_hat):

	return y_hat - y

def loss_dprime(y,y_hat):

	return np.ones_like(y_hat)

def regularizer_prime(x):

	return x

def regularizer_dprime(x):

	return np.ones_like(x)

def train(wts, data, epochs, lr,reg):

	n,d = len(data), len(data[0][0])
	print(n,d)
	alphas = np.zeros((n,d))

	wts_copy = wts.copy()
	print(wts_copy.shape)
	for epoch in range(epochs):
		for i in range(n):

			data_i = np.array(data[i][0])
			# print(data_i.shape)
			label_i = int(data[i][1])

			if(i==n):

				alphas = alphas - np.mean(alphas, axis=0, keepdims=True)  # update all alphas

			else:
				dot_i = data_i @ wts_copy

				dprime = loss_dprime(label_i, dot_i)

				diff = alphas[i,:] - (loss_prime(label_i, dot_i)*data_i) - reg*regularizer_prime(wts_copy)

				# print(diff.shape)
				inv = 1. / (1. + reg * regularizer_dprime(wts_copy))

				# print(inv.shape)
				scaled_data = inv * data_i
				# print(scaled_data.shape)

				# 10th line 1st term

				cte = dprime * (scaled_data @ diff) / (1 + dprime * (data_i @ scaled_data))

				# print(cte.shape)
				# inv*diff = 10th line 2nd term and update is gamma*dk
				update = lr * (inv * diff - cte * scaled_data)

				# print(update.shape)
				alphas[i, :] -= update  # update i-th alpha
				wts_copy += update  # update wts
			
			#print(i)
	
	#print(wts_copy)
	return wts_copy

print("How many clients??")
tot_clients = int(input())

print("Total rounds??")
tot_rounds =  int(input())

batch_size_list = []
epochs_list = []
print("Mini batch size for clients ")
batchx = int(input())
print("Epochs for clients ")
epochx = int(input())
for i in range(tot_clients):
	#print("Mini batch size for client "+str(i+1))
	batch_size_list.append(batchx)
	#print("epochs for client "+str(i+1))
	epochs_list.append(epochx)
	#print()

# img_path = './dataset/trainingSet/trainingSet'

# #get the path list using the path object
# image_paths = list(paths.list_images(img_path))

#data, labels = load_dataset(image_paths)

#print(len(data[0]), len(labels[0]))
#print(labels_actual)

# fourclass.txt
data = list(np.load("Data.npy"))
labels = list(np.load("label.npy"))

#split data into training and test set
X_train, X_test, Y_train, Y_test = train_test_split(data, labels,test_size=0.1, random_state=42)

labels_actual = Y_train

tot_samples = len(X_train)
clients = make_clients(X_train,Y_train,tot_clients,labels_actual)

print(clients['client_1'][0])

server_MLP = Model()
server_model = server_MLP.make_model(3,2)
# quantize_model = tfmot.quantization.keras.quantize_model
# server_model = quantize_model(server_model)
client_models = []

for i in range(len(list(clients.keys()))):

	client_MLP_x = Model()
	client_model_x = client_MLP_x.make_model(3,2)
	client_models.append(client_model_x)

K_clients = 3
test_acc_list = []
test_loss_list = []
for i in range(tot_rounds):

	scaled_wts_clients_list = []
	server_wts = server_model
	client_names= list(clients.keys())
	countx = 0

	for client in client_names:

		if(countx>K_clients):
			break
		client_model = client_models[countx]
		# setting client_model weights as the server_model wts
		print(len(server_wts))
		# client_model = server_wts

		client_data = np.array(clients[client])
		client_model = 		(client_model, clients[client], epochs_list[countx], 0.01, 1)
		# client_model.compile(optimizer = 'adam',loss='categorical_crossentropy',metrics=['accuracy'])

		# #fit local model with client's data
		# client_model.fit(batched_data_clients[client], epochs = epochs_list[countx], verbose=0)
		#print(len(client_model.get_weights()))
		#scaling the wts and storing them in a list
		scaled_wts = scaling_wts(tot_samples,client_model, len(clients[client]))
		print(len(scaled_wts))
		print("S "+str(len(scaled_wts)))
		scaled_wts_clients_list.append(scaled_wts)

		countx+=1
		K.clear_session()

	avg_wts_server = sum_scaled_wts(scaled_wts_clients_list)
	print("HI" +str(len(avg_wts_server)))
	server_model = np.array(avg_wts_server)

	countx = 0
	for client in client_names:
		client_model[countx] = server_model
		countx += 1

	print(server_model)
	countx = 0
	for client in client_names:
		client_models[countx] = server_model
		countx += 1
# 	server_acc, server_loss = test_model(X_test, Y_test, server_model)
# 	test_acc_list.append(server_acc)
# 	test_loss_list.append(server_loss)
# 	print("For round "+str(i+1))
# 	print("Test Accuracy: "+str(server_acc)+"||"+"Test Loss: "+str(server_loss))
# 	print()

# rounds = []
# for x in range(tot_rounds):
# 	rounds.append(x+1)
# plt.plot(rounds,test_acc_list)
# plt.show()
