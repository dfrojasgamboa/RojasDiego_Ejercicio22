import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets
import sklearn.preprocessing
import sklearn.neural_network
import sklearn.model_selection


numeros = sklearn.datasets.load_digits()
imagenes = numeros['images']  # Hay 1797 digitos representados en imagenes 8x8
n_imagenes = len(imagenes)
X = imagenes.reshape((n_imagenes, -1)) # para volver a tener los datos como imagen basta hacer data.reshape((n_imagenes, 8, 8))
Y = numeros['target']
print(np.shape(X), np.shape(Y))


X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.5)


scaler = sklearn.preprocessing.StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


max_iteration = 2500
loss_array = []
f1_array = []
f1_test_array = []
for i in range(1,21):
    mlp = sklearn.neural_network.MLPClassifier(activation='logistic', 
                                               hidden_layer_sizes=(i), 
                                               max_iter=max_iteration)
    mlp.fit(X_train, Y_train)
    loss_array.append( mlp.loss_ )
    f1_array.append( sklearn.metrics.f1_score(Y_train, mlp.predict(X_train), average='macro') )
    f1_test_array.append( sklearn.metrics.f1_score(Y_test, mlp.predict(X_test), average='macro') )
    
    
neuron_array = np.arange( 1, 21, 1)

plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.plot(neuron_array, loss_array )
plt.ylabel("Loss")
plt.xlabel( 'Neurons' )

plt.subplot(1,2,2)
plt.plot(neuron_array, f1_array , label= 'Train' )
plt.plot(neuron_array, f1_test_array , label= 'Test' )
plt.ylabel("F1")
plt.xlabel( 'Neurons' )
plt.legend( )

plt.subplots_adjust(hspace=.5)
plt.savefig( 'loss_f1.png' )
plt.show()


neuron_best = 6
mlp = sklearn.neural_network.MLPClassifier(activation='logistic', 
                                           hidden_layer_sizes=(neuron_best), 
                                           max_iter=max_iteration)
mlp.fit(X_train, Y_train)
print('Loss', mlp.loss_)
print('F1', sklearn.metrics.f1_score(Y_test, mlp.predict(X_test), average='macro'))
sklearn.metrics.plot_confusion_matrix(mlp, X_test, Y_test)


print(np.shape(mlp.coefs_))
for i in range(len(mlp.coefs_)):
    print(np.shape(mlp.coefs_[i]))
    

scale = np.max(mlp.coefs_[0])

plt.figure(figsize=(15,10))
for i in range(0,neuron_best):
    plt.subplot(2,3,i+1)
    plt.imshow(mlp.coefs_[0][:,i].reshape(8,8),cmap=plt.cm.RdBu, vmin=-scale, vmax=scale)
    plt.title( 'Neuron ' + str(i+1) )
    plt.subplots_adjust(hspace=.5)

plt.savefig('neuronas.png')