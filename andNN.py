import numpy as np


def sigmoid(x):
	return 1/(1+np.exp(-x))

vf = np.vectorize(sigmoid)

def predict(input_vector):
	assert(input_vector.shape==(4,1))
	
	# taking weighted sum
	wsum0_vector = np.dot(w01, input_vector)
	
	# activating
	h1_layer = vf(wsum0_vector)

	# taking weighted sum
	wsum1_vector = np.dot(w10, h1_layer)
	
	# activating
	op_layer = vf(wsum1_vector)

	return np.argmax(op_layer), op_layer


file = open("sample.txt")
dataset = [eval(x.rstrip("\n")) for x in file.readlines()]
file.close()
print("---------------------DATASET-----------------\n",dataset)

# Dividing the the dataset
training_data = dataset[8:]
testing_data = dataset[:8]
print("-----------------------traing------------------\n",
	training_data, len(training_data))
print("------------------test---------------------\n",
	testing_data, len(testing_data))

ip_layer = np.array([[None],
					 [None], 
					 [None], 
					 [None],
					])

h1_layer = np.array([[None], 
					 [None], 
					 [None], 
					 [None]])

op_layer = np.array([[None], 
					 [None]])

# Initializing Weight matrices
np.random.seed(0)
w01 = np.random.randn(4,4)
w10 = np.random.randn(2,4)


'''print(np.dot(w01,np.array([[1],
						   [2],
						   [3],
						   [4]
							])))'''

print(predict(np.array([[0],[0],[0],[0]])))