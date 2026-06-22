import numpy as np
import matplotlib.pyplot as plt
import sys
np.set_printoptions(threshold=sys.maxsize)
print("\033[H\033[J")
plt.close('all')

## Spectral characterstics##
max_features = 25
n_points = 1000
nu = np.linspace(0,1,n_points)


def random_chi3():
    """ generates a random spectrum, without NRB.
    output:
        params =  matrix of parameters. each row corresponds to the [amplitude, resonance, linewidth] of each generated feature (n_lor,3)
    """
    n_lor = np.random.randint(1,max_features)
    a = np.random.uniform(0.01,1.0,n_lor)
    w = np.random.uniform(0,1,n_lor)
    g = np.random.uniform(0.001,0.02, n_lor)
    params = np.c_[a,w,g]
    return params

def build_chi3(params):
    """ buiilds the normalized chi3 complex vector
    inputs: params: (n_lor, 3)
    outputs chi3: complex, (n_points, )"""

    chi3 = np.sum(params[:,0]/(-nu[:,np.newaxis]+params[:,1]-1j*params[:,2]),axis = 1)

    return chi3/np.max(np.abs(chi3))

def sigmoid(x,c,b):
    return 1/(1+np.exp(-(x-c)*b))



# ##### Sigmoid SPECNET NRB ####
# def generate_nrb():  
#     bs = np.random.normal(10,5,2)
#     c1 = np.random.normal(0.2,0.3)
#     c2 = np.random.normal(0.7,.3)
#     cs = np.r_[c1,c2]
#     sig1 = sigmoid(nu, cs[0], bs[0])
#     sig2 = sigmoid(nu, cs[1], -bs[1])
#     nrb  = sig1*sig2
#     return nrb

# ##### One Sigmoid NRB ####
# j=[-2,-1,1,2]
# k=[-5,-4,-3,-2,-1,1,2,3,4,5]

# def generate_nrb():
#     c = np.random.randint(0, 4,size=1)
#     c1=j[c[0]]
#     bs = np.random.randint(0, 10,size=1)
#     bs1=k[bs[0]]
#     nrb = sigmoid(nu, c1, bs1)
#     return nrb


### Polynomial NRB ####
def generate_nrb():
    """
    Produces a normalized shape for the Polynomial NRB
    outputs
        NRB: (n_points,)
    """
    [r2, r4, r5]= np.random.uniform(-10, 10,size=3)
    [r1,r3]=np.random.uniform(-1, 1,size=2)
    nrb=np.polyval([r1,r2,r3,r4,r5], nu)
    nrb=nrb-min(nrb)
    nrb=nrb/max(nrb)
    return nrb


def get_spectrum():
    """ Produces a cars spectrum.
    It outputs the normalized cars and the corresponding imaginary part.
    Outputs cars: (n_points,)
        chi3.imag: (n_points,) """
    chi3 = build_chi3(random_chi3())*np.random.uniform(0.3,1)
    nrb = generate_nrb()
    noise = np.random.randn(n_points)*np.random.uniform(0.0005,0.003)
    cars = ((np.abs(chi3+nrb)**2)/2+noise)
    return cars, chi3.imag

def generate_batch(size = 1):
    X = np.empty((size, n_points,1))
    y = np.empty((size,n_points))

    for i in range(size):
        X[i,:,0], y[i,:] = get_spectrum()
    return X, y

xnew, ynew = generate_batch(10000)
np.save("cars_10000.npy", xnew)
np.save("raman_10000.npy", ynew)
print("Cars data saved: cars_10000.npy")
print("Raman data saved: raman_10000.npy")


import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, BatchNormalization, Activation, Dropout
from keras import regularizers

tf.keras.backend.clear_session()
model = Sequential()
model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None,input_shape = (n_points, 1)))
model.add(Activation('relu'))
model.add(Conv1D(128, activation = 'relu', kernel_size = (32)))
model.add(Conv1D(64, activation = 'relu', kernel_size = (16)))
model.add(Conv1D(16, activation = 'relu', kernel_size = (8)))
model.add(Conv1D(16, activation = 'relu', kernel_size = (8)))
model.add(Conv1D(16, activation = 'relu', kernel_size = (8)))
model.add(Dense(32, activation = 'relu', kernel_regularizer=regularizers.l1_l2(l1 = 0, l2=0.1)))
model.add(Dense(16, activation = 'relu', kernel_regularizer=regularizers.l1_l2(l1 = 0, l2=0.1)))
model.add(Flatten())
model.add(Dropout(.25))
model.add(Dense(n_points,activation = 'relu'))
model.compile(loss='mse', optimizer='Adam', metrics=['mean_absolute_error','mse','accuracy'])
model.summary()

history = model.fit(xnew, ynew,epochs=10, verbose = 1, validation_split=0.25, batch_size=20, shuffle='true')

'Plotting Accuracy and loss'

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.figure()
plt.plot(epochs, acc, 'bo', label='Training')
plt.plot(epochs, val_acc, 'r', label='Validation')
plt.ylabel('Accuracy',fontweight='bold',size=20,family='Times New Roman')
plt.xlabel('Epoch',fontweight='bold',size=20,family='Times New Roman')
plt.grid(color = 'black', linestyle = '--', linewidth = 0.5)
plt.title('Training and validation accuracy',fontweight='bold',size=17,family='Times New Roman')
plt.legend(loc=0)
legend_properties = {'weight':'bold','size': 17,'family': 'Times New Roman'}
plt.legend(prop=legend_properties)
plt.figure()
plt.plot(epochs, loss, 'bo-', label='Training')
plt.plot(epochs, val_loss, 'ro-', label='Validation')
plt.ylabel('Loss',fontweight='bold',size=20,family='Times New Roman')
plt.xlabel('Epoch',fontweight='bold',size=20,family='Times New Roman')
plt.title('Training and validation loss',fontweight='bold',size=17,family='Times New Roman')
plt.grid(color = 'black', linestyle = '--', linewidth = 0.5)
plt.legend(loc=0)
legend_properties = {'weight':'bold','size': 17,'family': 'Times New Roman'}
plt.legend(prop=legend_properties)
plt.show()

# model.save('my_trained_model_with_default_paramters_of_specnet_spyder53_freq.h5')