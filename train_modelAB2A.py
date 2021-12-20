"""   JLL, SLT, 2021.12.20
AB2A: modelAB = UNet + RNN + PoseNet combines models A and B but runs only model A data
from /home/jinn/YPN/OPNet/train_modelA4.py, train_modelB3.py

1. Same as supercombo I/O
2. A Task: Multiclass semantic segmentation (num_classes = 6)
3. Loss: tf.keras.losses.CategoricalCrossentropy
4. Ground Truth:
   /home/jinn/dataAll/comma10k/Xmasks/*.png
   binary mask: one-hot encoded tensor = (None, 256, 512, 2x6)
5. Input:
   inputs[ 0 ].shape = (1, 12, 128, 256)   # 12 = 2 frames x 6 channels (YUV_I420: Y=4, U=1, V=1)
   inputs[ 1 ].shape = (1, 8)
   inputs[ 2 ].shape = (1, 2)
   inputs[ 3 ].shape = (1, 512)
6. Output:
   outs[ 12 ].shape = (1, 512) = Ymasks

Run: on 3 terminals
   (YPN) jinn@Liu:~/YPN/OPNet$ python serverAB2A.py --port 5557
   (YPN) jinn@Liu:~/YPN/OPNet$ python serverAB2A.py --port 5558 --validation
   (YPN) jinn@Liu:~/YPN/OPNet$ python train_modelAB2A.py --port 5557 --port_val 5558
Input:
   X_img:   /home/jinn/dataAll/comma10k/Ximgs_yuv/*.h5  (X for debugging with 10 imgs)
   Y_GTmsk: /home/jinn/dataAll/comma10k/Xmasks/*.png
   X_batch.shape = (none, 2x6, 128, 256) (num_channels = 6, 2 yuv images)
   Y_batch.shape = (none, 256, 512, 2x6) (num_classes  = 6)
Output:
  /YPN/ABNet/saved_model/modelAB_lossA.npy
  /YPN/ABNet/saved_model/modelAB_trainedA.h5


Training History:
  BATCH_SIZE = 16  EPOCHS = 2
  12/12 [==============================] - 130s 11s/step -
  loss: 1.4788 - accuracy: 0.4896 - val_loss: 1.6805 - val_accuracy: 0.2242
  Training Time: 00:04:53.37
"""
import os
import h5py
import time
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from modelAB2A import get_model
from serverAB2A import client_generator, BATCH_SIZE

EPOCHS = 2

os.environ['CUDA_VISIBLE_DEVICES'] = "-1"

def gen(hwm, host, port, model):
    for tup in client_generator(hwm=hwm, host=host, port=port):
        Ximgs, Xin1, Xin2, Xin3, Ytrue0, Ytrue1, Ytrue2, Ytrue3, Ytrue4, Ytrue5, Ytrue6, Ytrue7, Ytrue8, Ytrue9, Ytrue10, Ytrue11, Ymasks = tup

        Xins  = [Ximgs, Xin1, Xin2, Xin3]   #  (imgs, traffic_convection, desire, rnn_state)
        Ytrue0_11 = np.hstack((Ytrue0, Ytrue1, Ytrue2, Ytrue3, Ytrue4, Ytrue5, Ytrue6, Ytrue7, Ytrue8, Ytrue9, Ytrue10, Ytrue11))
          #Ytrue = [Ytrue0_11, Ymasks]
          #---  Ytrue[0].shape = (16, 2383)
          #---  Ytrue[1].shape = (16, 256, 512, 12)
        Ytrue = Ymasks
          # the following two lines are needed for Project A
        Ypred = model.predict(Xins)
          #---  Ypred.shape = (16, 256, 512, 12)
        loss = custom_loss(Ytrue, Ypred)

        yield Xins, Ytrue

def custom_loss(y_true, y_pred):
      #----- Project A Part
    cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    loss1 = cce(y_true[:, :, :,  0:6], y_pred[:, :, :,  0:6])
    loss2 = cce(y_true[:, :, :, 6:12], y_pred[:, :, :, 6:12])
    loss = (loss1 + loss2)/2

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

    #model.load_weights('./saved_model/modelAB-BestWeights.hdf5', by_name=True)
    adam = tf.keras.optimizers.Adam(lr=0.0001)
    print('---AAAAAAAAAA Project A AAAAAAAAAA---')
    model.compile(optimizer=adam, loss=custom_loss, metrics=["accuracy"])

    history = model.fit(
        gen(20, args.host, port=args.port, model=model),
        steps_per_epoch=200//BATCH_SIZE, epochs=EPOCHS,
        validation_data=gen(20, args.host, port=args.port_val, model=model),
        validation_steps=200//BATCH_SIZE, verbose=1, callbacks=callbacks_list)
          # steps_per_epoch = total images//BATCH_SIZE

    model.save('./saved_model/modelAB_trainedA.h5')

    end = time.time()
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Training Time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))

    np.save('./saved_model/modelAB_lossA', np.array(history.history['loss']))
    lossnpy = np.load('./saved_model/modelAB_lossA.npy')
    plt.plot(lossnpy)  #--- lossnpy.shape = (10,)
    plt.draw() #plt.show()
    plt.pause(0.5)
    print('---AAAAAAAAAA Project A AAAAAAAAAA---')
    input("Press ENTER to exit ...")
    plt.close()
