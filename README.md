# dl_train
marvelz@debian:~/Desktop/dl_train$ export LD_LIBRARY_PATH=/usr/local/cuda-11.2/lib64
marvelz@debian:~/Desktop/dl_train$ python3 asl_27k.py 
...
2021-01-20 01:15:12.401913: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1720] Found device 0 with properties: 
pciBusID: 0000:02:00.0 name: Quadro P2000 computeCapability: 6.1
coreClock: 1.4805GHz coreCount: 8 deviceMemorySize: 4.94GiB deviceMemoryBandwidth: 130.53GiB/s
...
(27455, 784)
(27455,)
(27455, 28, 28, 1)
(27455,)
...
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 28, 28, 75)        750       
_________________________________________________________________
batch_normalization (BatchNo (None, 28, 28, 75)        300       
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 14, 14, 75)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 14, 14, 50)        33800     
_________________________________________________________________
dropout (Dropout)            (None, 14, 14, 50)        0         
_________________________________________________________________
batch_normalization_1 (Batch (None, 14, 14, 50)        200       
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 7, 7, 50)          0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 7, 7, 25)          11275     
_________________________________________________________________
batch_normalization_2 (Batch (None, 7, 7, 25)          100       
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 4, 4, 25)          0         
_________________________________________________________________
flatten (Flatten)            (None, 400)               0         
_________________________________________________________________
dense (Dense)                (None, 512)               205312    
_________________________________________________________________
dropout_1 (Dropout)          (None, 512)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 26)                13338     
=================================================================
Total params: 265,075
Trainable params: 264,775
Non-trainable params: 300
_________________________________________________________________
...
858/858 [==============================] - 14s 13ms/step - loss: 0.8344 - accuracy: 0.7572 - val_loss: 0.1534 - val_accuracy: 0.9467
Epoch 2/10
858/858 [==============================] - 7s 9ms/step - loss: 0.0244 - accuracy: 0.9920 - val_loss: 0.5122 - val_accuracy: 0.8857
Epoch 3/10
858/858 [==============================] - 7s 9ms/step - loss: 0.0122 - accuracy: 0.9961 - val_loss: 0.4250 - val_accuracy: 0.9272
Epoch 4/10
858/858 [==============================] - 7s 9ms/step - loss: 0.0088 - accuracy: 0.9975 - val_loss: 0.3347 - val_accuracy: 0.9235
Epoch 5/10
858/858 [==============================] - 8s 9ms/step - loss: 0.0064 - accuracy: 0.9984 - val_loss: 0.2303 - val_accuracy: 0.9571
Epoch 6/10
858/858 [==============================] - 8s 9ms/step - loss: 0.0061 - accuracy: 0.9984 - val_loss: 0.3032 - val_accuracy: 0.9519
Epoch 7/10
858/858 [==============================] - 7s 9ms/step - loss: 0.0032 - accuracy: 0.9990 - val_loss: 0.3543 - val_accuracy: 0.9438
Epoch 8/10
858/858 [==============================] - 8s 9ms/step - loss: 0.0040 - accuracy: 0.9991 - val_loss: 0.2930 - val_accuracy: 0.9558
Epoch 9/10
858/858 [==============================] - 7s 9ms/step - loss: 0.0034 - accuracy: 0.9991 - val_loss: 0.3925 - val_accuracy: 0.9522
Epoch 10/10
858/858 [==============================] - 8s 9ms/step - loss: 0.0026 - accuracy: 0.9992 - val_loss: 0.2852 - val_accuracy: 0.9607
2021-01-20 01:16:36.929668: W tensorflow/python/util/util.cc:348] Sets are not currently considered sequences, but this may change in the future, so consider avoiding using them.


marvelz@debian:~/Desktop/dl_train$ python3 asl_27k_predict.py 
a
b