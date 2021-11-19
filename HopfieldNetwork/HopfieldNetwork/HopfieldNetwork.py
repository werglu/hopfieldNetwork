import numpy as np
import pandas as pd
import random

def read_data(filename):
    df = pd.read_csv(f"data/{filename}")
    return df

def get_weights_hebb(x):
    w = np.zeros([len(x),len(x)])
    for i in range(len(x)):
        for j in range(i,len(x)):
            if i == j:
                w[i,j] = 0
            else:
                w[i,j] = x[i]*x[j]
                w[j,i] = w[i,j]
    return w

def show(x):
    n = len(x)
    for i in range(0, n):
        if i%col == 0:
            print()
        if x[i] == 1:
            print('*', end = ''),
        else:
            print(' ', end = ''),
    print()
            

def get_test_data(x, points_count):
    test_data = np.array(x)
    n = len(x)
    noise_position = list(range(n))
    random_generator.shuffle(noise_position)
    for k in noise_position[:points_count]:  # invert points_count points in the pattern
        test_data[k] = -test_data[k]
    test_data = test_data.reshape((n, 1)) 
    return test_data
  
def process(w, y_vec, theta=0.5, time=100):
    for s in range(time):
        m = len(y_vec)
        #i = random.randint(0,m-1)
        for i in range(m): 
            u = np.dot(w[i][:], y_vec) - theta #COMMENT: w sumie nie wiem czy to jest sync czy async ale uaktualniam wszytskie po prostu

            if u >= 0:
                y_vec[i] = 1
            elif u < 0:
                y_vec[i] = -1
            
    return y_vec

df_test = read_data("animals-14x9.csv")
col = 9
data = np.array(df_test);
neurons_count = len(data[1]);
random_generator = np.random.default_rng(seed=123)


for i in range(1, len(data)):
    weights = weights + get_weights_hebb(data[i])
    
weights = weights / len(data) # divide when hebb rule

for i in range(1, len(data)):
    print(i)
    show(data[i])
    data_test = get_test_data(x[i], 15)
    print('Test data:')
    show(data_test)
    print('After process:')
    show(process(weights, data_test))
    print()