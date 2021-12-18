"""   JLL, SLT, 2021.12.16
rename AB2B to AB and then run Project B
run train_modelAB2A.py first and then train_modelAB2B.py
from /home/jinn/YPN/OPNet/train_modelA4.py, train_modelB3.py
train modelAB = UNet + RNN + PoseNet
combine Projects A and B

1. Same as supercombo I/O
2. Tasks: multiclass semantic segmentation + temporal state (features) + path planning (PP)
   The temporal state (a 512 array) represents (are features of) all path planning details:
   path prediction, lane detection, lead car xyva, desire etc., i.e., outs[0]...[10] in output.txt.
3. Ground Truth from pathdata.h5, radardata.h5:
   Pose: 56 = Ego path 51 (x, y) + 5 radar lead car's dRel (relative distance), yRel, vRel (velocity), aRel (acceleration), prob
4. Loss: mean squared error (mse)
5. Input:
   inputs[ 0 ].shape = (1, 12, 128, 256)   # 12 = 2 frames x 6 channels (YUV_I420: Y=4, U=1, V=1)
   inputs[ 1 ].shape = (1, 8)
   inputs[ 2 ].shape = (1, 2)
   inputs[ 3 ].shape = (1, 512)
6. Output:
   outs[ 0 ].shape = (1, 385)
   outs[ 1 ].shape = (1, 386)
   outs[ 2 ].shape = (1, 386)
   outs[ 3 ].shape = (1, 58)
   outs[ 4 ].shape = (1, 200)
   outs[ 5 ].shape = (1, 200)
   outs[ 6 ].shape = (1, 200)
   outs[ 7 ].shape = (1, 8)
   outs[ 8 ].shape = (1, 4)
   outs[ 9 ].shape = (1, 32)
   outs[ 10 ].shape = (1, 12)
   outs[ 11 ].shape = (1, 512)

Run: on 3 terminals
   (YPN) jinn@Liu:~/YPN/OPNet$ python serverAB.py --port 5557
   (YPN) jinn@Liu:~/YPN/OPNet$ python serverAB.py --port 5558 --validation
   (YPN) jinn@Liu:~/YPN/OPNet$ python train_modelAB.py --port 5557 --port_val 5558
Input:
  /home/jinn/dataB/UHD--2018-08-02--08-34-47--32/yuv.h5, pathdata.h5, radardata.h5
  /home/jinn/dataB/UHD--2018-08-02--08-34-47--33/yuv.h5, pathdata.h5, radardata.h5
Output:
  /OPNet/saved_model/modelAB_loss.npy

Training History:
  BATCH_SIZE = 16  EPOCHS = 2
  71/71 [==============================] - 475s 7s/step -
  loss: 1.7758 - val_loss: 0.2739 - val_concatenate_3_loss: 0.2651 - val_conv2d_transpose_12_loss: 0.0088 - val_concatenate_3_custom_loss: 0.2651 - val_conv2d_transpose_12_custom_loss: 0.0088
  Training Time: 00:16:03.38
"""
import os
import h5py
import time
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from modelAB import get_model
from serverAB import client_generator, BATCH_SIZE

EPOCHS = 2

os.environ['CUDA_VISIBLE_DEVICES'] = "-1"

def gen(hwm, host, port, model):
    for tup in client_generator(hwm=hwm, host=host, port=port):
        Ximgs, Xin1, Xin2, Xin3, Ytrue0, Ytrue1, Ytrue2, Ytrue3, Ytrue4, Ytrue5, Ytrue6, Ytrue7, Ytrue8, Ytrue9, Ytrue10, Ytrue11, Ymasks = tup

        Xins  = [Ximgs, Xin1, Xin2, Xin3]   #  (imgs, traffic_convection, desire, rnn_state)
        Ytrue0_11 = np.hstack((Ytrue0, Ytrue1, Ytrue2, Ytrue3, Ytrue4, Ytrue5, Ytrue6, Ytrue7, Ytrue8, Ytrue9, Ytrue10, Ytrue11))
        Ytrue = [Ytrue0_11, Ymasks]
          #---  Ytrue[0].shape = (16, 2383)
          #---  Ytrue[1].shape = (16, 256, 512, 12)

        yield Xins, Ytrue

def custom_loss(y_true, y_pred):
    loss = tf.keras.losses.mse(y_true[0], y_pred[0])

    return loss

if __name__=="__main__":
    start = time.time()
    parser = argparse.ArgumentParser(description='Training modelAB')
    parser.add_argument('--host', type=str, default="localhost", help='Data server ip address.')
    parser.add_argument('--port', type=int, default=5557, help='Port of server.')
    parser.add_argument('--port_val', type=int, default=5558, help='Port of server for validation dataset.')
    args = parser.parse_args()

    # Build model
    img_shape = (12, 128, 256)
    desire_shape = (8)
    traffic_convection_shape = (2)
    rnn_state_shape = (512)
    num_classes = 6
    model = get_model(img_shape, desire_shape, traffic_convection_shape, rnn_state_shape, num_classes)
    #model.summary()

    filepath = "./saved_model/modelAB-BestWeights.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1,
                                 save_best_only=True, mode='min')
    callbacks_list = [checkpoint]

    model.load_weights('./saved_model/modelAB_trainedA.h5', by_name=True)
    adam = tf.keras.optimizers.Adam(lr=0.0001)
    print('---BBBBBBBBBB Project B BBBBBBBBBB---')
    model.compile(optimizer=adam, loss=custom_loss, metrics=custom_loss)

    history = model.fit(
        gen(20, args.host, port=args.port, model=model),
        steps_per_epoch=1150//BATCH_SIZE, epochs=EPOCHS,
        validation_data=gen(20, args.host, port=args.port_val, model=model),
        validation_steps=1150//BATCH_SIZE, verbose=1, callbacks=callbacks_list)
          # steps_per_epoch = total images//BATCH_SIZE

    model.save('./saved_model/modelAB_trainedAB.h5')

    end = time.time()
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Training Time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))

    np.save('./saved_model/modelAB_loss', np.array(history.history['loss']))
    lossnpy = np.load('./saved_model/modelAB_loss.npy')
    plt.plot(lossnpy)  #--- lossnpy.shape = (10,)
    plt.draw() #plt.show()
    plt.pause(0.5)
    print('---BBBBBBBBBB Project B BBBBBBBBBB---')
    input("Press ENTER to exit ...")
    plt.close()
