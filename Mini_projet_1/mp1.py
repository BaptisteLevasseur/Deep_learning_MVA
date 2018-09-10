import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches

import json

from keras.models import Sequential, model_from_json
from keras.utils import np_utils
from keras.layers import Dense, Activation, Conv2D, Dropout, MaxPooling2D, Flatten, UpSampling2D
from keras.optimizers import SGD
from sklearn.preprocessing import StandardScaler
from keras.callbacks import TensorBoard




def generate_a_drawing(figsize, U, V, noise=0.0):
    fig = plt.figure(figsize=(figsize,figsize))
    ax = plt.subplot(111)
    plt.axis('Off')
    ax.set_xlim(0,figsize)
    ax.set_ylim(0,figsize)
    ax.fill(U, V, "k")
    fig.canvas.draw()
    imdata = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)[::3].astype(np.float32)
    imdata = imdata + noise * np.random.random(imdata.size)
    plt.close(fig)
    return imdata

def generate_a_rectangle(noise=0.0, free_location=False,return_denoise=False):
    figsize = 1.0    
    U = np.zeros(4)
    V = np.zeros(4)
    if free_location:
        corners = np.random.random(4)
        top = max(corners[0], corners[1])
        bottom = min(corners[0], corners[1])
        left = min(corners[2], corners[3])
        right = max(corners[2], corners[3])
    else:
        side = (0.3 + 0.7 * np.random.random()) * figsize
        top = figsize/2 + side/2
        bottom = figsize/2 - side/2
        left = bottom
        right = top
    U[0] = U[1] = top
    U[2] = U[3] = bottom
    V[0] = V[3] = left
    V[1] = V[2] = right
    if return_denoise:
        return [generate_a_drawing(figsize, U, V, noise),generate_a_drawing(figsize, U, V, 0)]
    else:
        return generate_a_drawing(figsize, U, V, noise)


def generate_a_disk(noise=0.0, free_location=False):
    figsize = 1.0
    if free_location:
        center = np.random.random(2)
    else:
        center = (figsize/2, figsize/2)
    radius = (0.3 + 0.7 * np.random.random()) * figsize/2
    N = 50
    U = np.zeros(N)
    V = np.zeros(N)
    i = 0
    for t in np.linspace(0, 2*np.pi, N):
        U[i] = center[0] + np.cos(t) * radius
        V[i] = center[1] + np.sin(t) * radius
        i = i + 1
    return generate_a_drawing(figsize, U, V, noise)

def generate_a_triangle(noise=0.0, free_location=False):
    figsize = 1.0
    if free_location:
        U = np.random.random(3)
        V = np.random.random(3)
    else:
        size = (0.3 + 0.7 * np.random.random())*figsize/2
        middle = figsize/2
        U = (middle, middle+size, middle-size)
        V = (middle+size, middle-size, middle-size)
    imdata = generate_a_drawing(figsize, U, V, noise)
    return [imdata, [U[0], V[0], U[1], V[1], U[2], V[2]]]


im = generate_a_rectangle(10, True)
plt.imshow(im.reshape(72,72), cmap='gray')
plt.show()

im = generate_a_disk(10)
plt.imshow(im.reshape(72,72), cmap='gray')
plt.show()

[im, v] = generate_a_triangle(20, False)
plt.imshow(im.reshape(72,72), cmap='gray')
plt.show()

def generate_dataset_classification(nb_samples, noise=0.0, free_location=False):
    # Getting im_size:
    im_size = generate_a_rectangle().shape[0]
    X = np.zeros([nb_samples,im_size])
    Y = np.zeros(nb_samples)
    print('Creating data:')
    for i in range(nb_samples):
        if i % 10 == 0:
            print(i)
        category = np.random.randint(3)
        if category == 0:
            X[i] = generate_a_rectangle(noise, free_location)
        elif category == 1: 
            X[i] = generate_a_disk(noise, free_location)
        else:
            [X[i], V] = generate_a_triangle(noise, free_location)
        Y[i] = category
    X = (X + noise) / (255 + 2 * noise)
    return [X, Y]

def generate_dataset_rectangle(nb_samples,noise=0.0, free_location=True):
    # Get the size
    im_size = generate_a_rectangle().shape[0]
    X = np.zeros([nb_samples,im_size])
    Y = np.zeros([nb_samples,im_size])
    
    print('Creating data:')
    for i in range(nb_samples):
        if i % 10 == 0:
            print(i)
        X[i],Y[i] = generate_a_rectangle(noise, free_location, return_denoise=True)
    X = (X + noise) / (255 + 2 * noise)
    return [X, Y]

def generate_test_set_classification():
    np.random.seed(42)
    [X_test, Y_test] = generate_dataset_classification(300, 20, True)
    Y_test = np_utils.to_categorical(Y_test, 3) 
    return [X_test, Y_test]

def generate_dataset_regression(nb_samples, noise=0.0):
    # Getting im_size:
    im_size = generate_a_triangle()[0].shape[0]
    X = np.zeros([nb_samples,im_size])
    Y = np.zeros([nb_samples, 6])
    print('Creating data:')
    for i in range(nb_samples):
        if i % 10 == 0:
            print(i)
        [X[i], Y[i]] = generate_a_triangle(noise, True)
    X = (X + noise) / (255 + 2 * noise)
    return [X, Y]


def visualize_prediction(x, y):
    fig, ax = plt.subplots(figsize=(5, 5))
    I = x.reshape((72,72))
    ax.imshow(I, extent=[-0.15,1.15,-0.15,1.15],cmap='gray')
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])

    xy = y.reshape(3,2)
    tri = patches.Polygon(xy, closed=True, fill = False, edgecolor = 'r', linewidth = 5, alpha = 0.5)
    ax.add_patch(tri)

    plt.show()

def generate_test_set_regression(n_test,noise):
    np.random.seed(42)
    [X_test, Y_test] = generate_dataset_regression(n_test, noise)
    return [X_test, Y_test]



def generate_dataset(load=False, n_train=300,n_test=100,noise=20,free_location=False):
      # Generating and processing data. Can save them
    if free_location == True:
        X = 'Xl'
        y = 'yl'
    else:
        X = 'X'
        y = 'y'
            
    if load:
        X_train = np.load(X+'_train.npy')
        y_train = np.load(y+'_train.npy')   
        X_test = np.load(X+'_test.npy')
        y_test = np.load(y+'_test.npy')   
    else:
        [X_train, Y_train] = generate_dataset_classification(n_train,noise,free_location)
        [X_test, Y_test] = generate_dataset_classification(n_test,noise,free_location)
        y_train = np_utils.to_categorical(Y_train, 3)
        y_test = np_utils.to_categorical(Y_test, 3)
        np.save(X+'_train',X_train)
        np.save(y+'_train',y_train)
        np.save(X+'_test',X_test)
        np.save(y+'_test',y_test)
    return X_train,y_train, X_test,y_test

def generate_data_regression(load=False, n_train=300, n_test=100,noise=20):
    if load:
        X_train = np.load('Xr_train.npy')
        y_train = np.load('yr_train.npy')   
        X_test = np.load('Xr_test.npy')
        y_test = np.load('yr_test.npy')
    else:
        X_train,y_train = generate_dataset_regression(n_train, noise)
        X_test, y_test = generate_test_set_regression(n_test,noise)
        np.save('Xr_train',X_train)
        np.save('yr_train',y_train)
        np.save('Xr_test',X_test)
        np.save('yr_test',y_test)
    scaler = StandardScaler()
    scaler.fit(y_train)
    scaler.transform(y_train)
    scaler.transform(y_test)
    return X_train,y_train, X_test,y_test, scaler


def generate_data_denoise(load=False, n_train=300, n_test=100,noise=20):
    if load:
        X_train = np.load('Xn_train.npy')
        y_train = np.load('yn_train.npy')   
        X_test = np.load('Xn_test.npy')
        y_test = np.load('yn_test.npy')
    else:
        X_train,y_train = generate_dataset_rectangle(n_train, noise)
        X_test,y_test = generate_dataset_rectangle(n_test, noise)
        np.save('Xn_train',X_train)
        np.save('yn_train',y_train)
        np.save('Xn_test',X_test)
        np.save('yn_test',y_test)
    return X_train,y_train, X_test,y_test

def test_model(model,X_test,y_test):
    score = model.evaluate(X_test, y_test, batch_size=128)
    if type(score)==list:
        print("Loss : " + str(score[0]*100) + " %, Accuracy : " + str(score[1]*100) + "  %")
    else:
        print("Loss : " + str(score*100) + " %")


def fc_model(X_train,y_train):
    # Tune the hyperparametres here
    print("Definition of the model")
    # Definition of the type of model
    model = Sequential()
    model.add(Dense(3,input_shape=(5184,),activation='sigmoid'))
    
    # Definition of the optimization method
#    sgd = SGD(lr=0.01, decay = 1e-6, momentum = 0.9, nesterov = True)
    
    model.compile(loss='mean_squared_error',optimizer='adam',metrics=['accuracy'])
    # Running optimization
    print("Training of the model")
    model.fit(X_train, y_train, epochs=100, batch_size=32) 
    
    return model

def cnn_model(X_train,y_train):
    
    # Tune the hyperparametres here
    print("Definition of the model")
    # Definition of the type of model
    model = Sequential()
    
    model.add(Conv2D(16,(5,5),activation = 'relu', input_shape=(72,72,1)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(8,(5,5),activation = 'relu'))
    model.add(Flatten()) #size 18496
    model.add(Dropout(0.25))
    model.add(Dense(3,activation='sigmoid'))
    
    
    # Definition of the optimization method
#    sgd = SGD(lr=0.01, decay = 1e-6, momentum = 0.9, nesterov = True)
    
    model.compile(loss='mean_squared_error',optimizer='adam',metrics=['accuracy'])
    # Running optimization
    print("Training of the model")
    model.fit(X_train, y_train, epochs=100, batch_size=32) 
    
    return model


    
    
    

def regression_model(X_train,y_train):
    
    # Tune the hyperparametres here
    print("Definition of the model")
    # Definition of the type of model
    model = Sequential()
    
    model.add(Conv2D(16,(5,5),activation = 'relu', input_shape=(72,72,1)))
    model.add(Flatten()) #size 18496
    model.add(Dense(128, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(6))
    
    model.compile(loss='mean_squared_error',optimizer='adam',metrics=['accuracy'])
    # Running optimization
    print("Training of the model")
    model.fit(X_train, y_train, epochs=100, batch_size=32) 
    
    return model
    
def hourglass_model(X_train,y_train,X_test,y_test):
        # Tune the hyperparametres here
    print("Definition of the model")
    # Definition of the type of model
    model = Sequential()
    
    #encode
    model.add(Conv2D(32,(3,3),activation = 'relu', padding='same', input_shape=(72,72,1)))
    model.add(MaxPooling2D((2,2),padding='same'))
    model.add(Conv2D(32,(3,3),activation = 'relu', padding='same'))
    model.add(MaxPooling2D((2,2),padding='same'))
    #decode
    model.add(Conv2D(32,(3,3),activation = 'relu', padding='same', input_shape=(72,72,1)))
    model.add(UpSampling2D((2,2)))
    model.add(Conv2D(32,(3,3),activation = 'relu', padding='same'))
    model.add(UpSampling2D((2,2)))
    model.add(Conv2D(1,(3,3),activation = 'sigmoid', padding='same'))
    print(model.output_shape)
    
    model.compile(loss='binary_crossentropy',optimizer='adadelta')
    # Running optimization
    print("Training of the model")
    model.fit(X_train, y_train, epochs=20, batch_size=1000, validation_data=(X_test,y_test)) 
    return model
        

def fully_connected(X_train,y_train, X_test, y_test, load_model=False,free_location=False):
    if free_location:
        name_weights='model_l_fc.h5'
        name_model='model_l_fc.json'
    else:
        name_weights='model_l_fc.h5'
        name_model='model_l_fc.json'
    if load_model:
        with open(name_model, "r") as jfile:
            model = model_from_json(json.load(jfile))
        model.load_weights(name_weights)
        model.compile("adam", "mse")
        
    else:
        model = fc_model(X_train,y_train)
        model.save_weights(name_weights, overwrite=True)
        with open(name_model, "w") as outfile:
            json.dump(model.to_json(), outfile)
            
    
    print("Layer 1")
    weights = model.get_layer(index=0).get_weights()
    columns = weights[0]
    
    form_shape = ['rectangle','disque','triangle'] 
    for i in range(3):
        shape_i = columns[:,i].reshape((72,72))
        plt.imshow(shape_i)
        plt.title(form_shape[i])
        plt.show()

    test_model(model,X_test,y_test)
    
def regression(X_train,y_train, X_test, y_test,scaler,load_model=False):
    # Adapt data format
    X_train = X_train.reshape(X_train.shape[0],72,72,1)
    X_test = X_test.reshape(X_test.shape[0],72,72,1)
    
    name_weights='model_r.h5'
    name_model='model_r.json'
    if load_model:
        with open(name_model, "r") as jfile:
            model = model_from_json(json.load(jfile))
        model.load_weights(name_weights)
        model.compile("adam", "mse",metrics=['accuracy'])
        
    else:
        model = regression_model(X_train,y_train)
        model.save_weights(name_weights, overwrite=True)
        with open(name_model, "w") as outfile:
            json.dump(model.to_json(), outfile)
            
    test_model(model,X_test,y_test)
    return model
    


def CNN(X_train,y_train, X_test, y_test,load_model=False,free_location=False):
    # Adapt data format
    X_train = X_train.reshape(X_train.shape[0],72,72,1)
    X_test = X_test.reshape(X_test.shape[0],72,72,1)
    if free_location:
        name_weights='model_l_cnn.h5'
        name_model='model_l_cnn.json'
    else:
        name_weights='model_cnn.h5'
        name_model='model_cnn.json'
    if load_model:
        with open(name_model, "r") as jfile:
            model = model_from_json(json.load(jfile))
        model.load_weights(name_weights)
        model.compile("sgd", "mse")
        
    else:
        model = cnn_model(X_train,y_train)
        model.save_weights(name_weights, overwrite=True)
        with open(name_model, "w") as outfile:
            json.dump(model.to_json(), outfile)
            

    test_model(model,X_test,y_test)
    
def hourglass(X_train,y_train, X_test, y_test,load_model=False):
    X_train = X_train.reshape(X_train.shape[0],72,72,1)
    y_train = y_train.reshape(y_train.shape[0],72,72,1)
    X_test = X_test.reshape(X_test.shape[0],72,72,1) 
    y_test = y_test.reshape(y_test.shape[0],72,72,1) 
    name_weights='model_hr.h5'
    name_model='model_hr.json'
    if load_model:
        with open(name_model, "r") as jfile:
            model = model_from_json(json.load(jfile))
        model.load_weights(name_weights)
        model.compile("sgd", "mse")
        
    else:
        model = hourglass_model(X_train,y_train,X_test,y_test)
        model.save_weights(name_weights, overwrite=True)
        with open(name_model, "w") as outfile:
            json.dump(model.to_json(), outfile)
    return model

if __name__ == "__main__":
    free_location = True
    mode_list = ['classification','regression','denoise']
    mode = mode_list[2]
#    mode = 'test'
    if mode=='classification':
        if not free_location:
            # dataset
            X_train,y_train,X_test,y_test = generate_dataset(load=True)
            # models
            fully_connected(X_train,y_train,X_test,y_test,load_model=True)
            CNN(X_train,y_train,X_test,y_test,load_model=True)
        else:
            # dataset
            X_train,y_train,X_test,y_test = generate_dataset(load=True,free_location=True,n_train=1000,n_test=200)
            # models
            fully_connected(X_train,y_train,X_test,y_test,load_model=True,free_location=True)
            CNN(X_train,y_train,X_test,y_test,load_model=False,free_location=True)
    elif mode=='regression':
        # dataset
        X_train,y_train,X_test,y_test,scaler = generate_data_regression(load=True,n_train=1000)
        model = regression(X_train,y_train,X_test,y_test,scaler,load_model=True)
        
        i=20
        x_pred = X_train[i]
        y_pred = y_train[i]
        visualize_prediction(x_pred,y_pred)
        y_pred= model.predict(x_pred.reshape(1,72,72,1))
        visualize_prediction(x_pred,y_pred)
    elif mode=='denoise':
        # dataset
        X_train,y_train,X_test,y_test = generate_data_denoise(load=True, n_train=300, n_test=100,noise=20)
        # models
        model = hourglass(X_train,y_train,X_test,y_test,load_model=True)
        test = model.predict(X_train[10].reshape(1,72,72,1)).reshape(72,72)
        plt.imshow(test)
        plt.show()
        plt.imshow(y_train[10].reshape(72,72))
        
    else:
        print("Hello world")
        
        