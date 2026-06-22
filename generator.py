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
np.save("synthetic_data/cars_10000.npy", xnew)
np.save("synthetic_data/raman_10000.npy", ynew)