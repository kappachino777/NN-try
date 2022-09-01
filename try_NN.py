from base64 import standard_b64decode
from json import load
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import numpy.random as r
from sklearn.metrics import accuracy_score

#preparing data from scikit
digits = load_digits()
X_scale = StandardScaler()
X = X_scale.fit_transform(digits.data)
y = digits.target
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.4)

#convert output data to vector
def convert_y_to_vect(y):
    y_vect = np.zeros((len(y),10))
    for i in range (len(y)):
        y_vect[i,y[i]] = i
    return y_vect

y_v_train = convert_y_to_vect(y_train)
y_v_test = convert_y_to_vect(y_test)


nn_structure = [64,30,10]

#sigmoid activation function
def f(x):
    return 1/(1+np.exp(-x))
#sigmoid  derivative
def f_deriv(x):
    return f(x)*(1-f(x))

#weight and bias initialisation
def setup_and_init_weights(nn_structure):
    W={}
    b={}
    for i in range (1,len(nn_structure)):
        W[i] = r.random_sample((nn_structure[i],nn_structure[i-1]))
        b[i] = r.random_sample((nn_structure[i],))
    return W,b

#mean acumulation value initialisation
#same size as weight and bias matrix 
#set to zero
def init_tri_values(nn_structure):
    tri_W = {}
    tri_b = {}
    for l in range (1,len(nn_structure)):
        tri_W[l] = np.zeros((nn_structure[l],nn_structure[l-1]))
        tri_b[l] = np.zeros((nn_structure[l],))
    return tri_W,tri_b

#feed foward function
def feed_foward(x,W,b):
    h = {1:x}
    z = {}
    for i in range (1,len(W) + 1):
        if i == 1:
            node_in = x
        else:
            node_in = h[i]
        z[i+1] = W[i].dot(node_in) + b[i]
        h[i+1] = f(z[i+1])
    return h,z

#calculation for surface layer
def calculate_out_layer_delta(y,h_out,z_out):
    return -(y-h_out) * f_deriv(z_out)

#calculation for hidden layer
def calculate_hidden_delta(delta_plus_1,w_1,z_1):
    return np.dot(np.transpose(w_1),delta_plus_1)*f_deriv(z_1)

#to start training call this function
def train_nn(nn_structure,X,y,iter_num = 3000, alpha = 0.1):
    W,b = setup_and_init_weights(nn_structure)
    cnt = 0
    m = len(y)
    avg_cost_func = []
    print('Starting grad descent for {} iteration'.format(iter_num))
    while cnt < iter_num:
        if cnt%1000 == 0:
            print('iteration {} of {}'.format(cnt,iter_num))
        tri_W,tri_b = init_tri_values(nn_structure)
        avg_cost = 0
        for i in range (len(y)):
            delta = {}
            h,z = feed_foward(X[i,:],W,b)
            for l in range (len(nn_structure),0,-1):
                if l == len(nn_structure):
                    delta[l] = calculate_out_layer_delta(y[i,:],h[l],z[l])
                    avg_cost += np.linalg.norm((y[i,:]-h[l]))
                else :
                    if l>1:
                        delta[l] = calculate_hidden_delta(delta[l+1],W[l],z[l])
                    tri_W[l] += np.dot(delta[l+1][:,np.newaxis],np.transpose(h[l][:,np.newaxis]))
                    tri_b[l] += delta[l+1]
        for l in range (len(nn_structure)-1,0,-1):
            W[l] += -alpha * (1.0/m * tri_W[l])
            b[l] += -alpha * (1.0/m * tri_b[l])
        avg_cost = 1.0/m * avg_cost
        avg_cost_func.append(avg_cost)
        cnt+=1
    return W,b, avg_cost_func

#start training
W,b,avg_cost_func = train_nn(nn_structure,X_train,y_v_train)

#plt.plot(avg_cost_func)
#plt.ylabel('Average J') 
#plt.xlabel('iteration')
#plt.show()           

#to test training result
def predict_y (W,b,X,n_layers):
    m = X.shape[0]
    y = np.zeros((m,))
    for i in range(m):
        h,z = feed_foward(X[i,:],W,b)
        y[i] = np.argmax(h[n_layers])
    return y 

y_pred = predict_y(W,b,X_test,3)

print(y_pred)
print (accuracy_score(y_test,y_pred)*100)
 
        