"""   JLL, SLT, 2021.12.8
from /home/jinn/YPN/OPNet/train_modelA4.py, train_modelB3.py
train modelAB = UNet + RNN + PoseNet

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
  BATCH_SIZE = 8  EPOCHS = 2
  2/143 [..............................] - ETA: 1:45 - loss: 125.8662 - path_loss: 0.2352 - left_lane_loss: 6.2859e-05 - right_lane_loss: 5.8311e-05 - lead_loss: 125.3410 - long_x_loss: 1.0176e-04 - long_a_loss: 5.9675e-05 - long_v_loss: 3.8941e-05 - desire_state_loss: 0.0156 - meta_loss: 0.2393 - desire_pred_loss: 0.0157 - pose_loss: 0.0019 - add_9_loss: 8.0104e-04 - conv2d_transpose_12_loss: 0.0163 - path_accuracy: 0.0000e+00 - left_lane_accuracy: 0.0000e+00 - right_lane_accuracy: 0.0000e+00 - lead_accuracy: 0.0000e+00 - long_x_accuracy: 0.0000e+00 - long_a_accuracy: 0.0000e+00 - long_v_accuracy: 0.0000e+00 - desire_state_accuracy: 1.0000 - meta_accuracy: 0.0000e+00 - desire_pred_accuracy: 0.0625 - pose_accuracy: 0.0000e+00 - add_9_accuracy: 0.0000e+00 - conv2d_transpose_12_accuracy: 0.0459  #---  running def gen()
  143/143 [==============================] - 457s 3s/step - loss: 66.0288 - path_loss: 0.0644 - left_lane_loss: 2.2636e-05 - right_lane_loss: 1.8324e-05 - lead_loss: 64.6974 - long_x_loss: 3.0939e-05 - long_a_loss: 4.0386e-05 - long_v_loss: 2.7945e-05 - desire_state_loss: 0.0158 - meta_loss: 0.0038 - desire_pred_loss: 0.0157 - pose_loss: 6.5371e-05 - add_9_loss: 1.2111 - conv2d_transpose_12_loss: 0.0204 - path_accuracy: 0.2351 - left_lane_accuracy: 0.0122 - right_lane_accuracy: 0.0000e+00 - lead_accuracy: 0.3960 - long_x_accuracy: 0.0000e+00 - long_a_accuracy: 8.7413e-04 - long_v_accuracy: 0.0000e+00 - desire_state_accuracy: 0.1722 - meta_accuracy: 0.0411 - desire_pred_accuracy: 0.2474 - pose_accuracy: 0.0726 - add_9_accuracy: 0.0000e+00 - conv2d_transpose_12_accuracy: 0.0778 - val_loss: 8.8604 - val_path_loss: 0.0654 - val_left_lane_loss: 4.2193e-07 - val_right_lane_loss: 2.7443e-07 - val_lead_loss: 8.6227 - val_long_x_loss: 1.0884e-06 - val_long_a_loss: 2.9815e-07 - val_long_v_loss: 2.6015e-07 - val_desire_state_loss: 0.0156 - val_meta_loss: 0.1225 - val_desire_pred_loss: 0.0156 - val_pose_loss: 7.5895e-06 - val_add_9_loss: 0.0064 - val_conv2d_transpose_12_loss: 0.0120 - val_path_accuracy: 0.2220 - val_left_lane_accuracy: 0.0726 - val_right_lane_accuracy: 0.0000e+00 - val_lead_accuracy: 0.0035 - val_long_x_accuracy: 0.0000e+00 - val_long_a_accuracy: 0.0000e+00 - val_long_v_accuracy: 0.0000e+00 - val_desire_state_accuracy: 0.9904 - val_meta_accuracy: 0.0166 - val_desire_pred_accuracy: 0.0000e+00 - val_pose_accuracy: 0.0000e+00 - val_add_9_accuracy: 0.0000e+00 - val_conv2d_transpose_12_accuracy: 0.0821
  Training Time: 00:15:30.27
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

        Xins  = (Ximgs, Xin1, Xin2, Xin3)   #  (imgs, traffic_convection, desire, rnn_state)
        Ytrue = (Ytrue0, Ytrue1, Ytrue2, Ytrue3, Ytrue4, Ytrue5, Ytrue6, Ytrue7, Ytrue8, Ytrue9, Ytrue10, Ytrue11, Ymasks)
        #Ypred = model.predict(x=Ximgs)
        #loss = custom_loss(Ytrue, Ypred)

        yield Xins, Ytrue

  # multiple losses: https://stackoverflow.com/questions/53707199/keras-combining-two-losses-with-adjustable-weights-where-the-outputs-do-not-have
  # https://stackoverflow.com/questions/58230074/how-to-optimize-multiple-loss-functions-separately-in-keras
def custom_loss(y_true, y_pred):
      #----- Project A Part
    '''
    cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    y_mskT = y_true[12]
    y_mskP = y_pred[12]
    loss1 = cce(y_mskT[:, :, :,  0:6], y_mskP[:, :, :,  0:6])
    loss2 = cce(y_mskT[:, :, :, 6:12], y_mskP[:, :, :, 6:12])
    loss = (loss1 + loss2)/2
    '''
      #----- Project B Part
    loss0 = tf.keras.losses.mse(y_true[0], y_pred[0])
    loss1 = tf.keras.losses.mse(y_true[1], y_pred[1])
    loss2 = tf.keras.losses.mse(y_true[2], y_pred[2])
    loss3 = tf.keras.losses.mse(y_true[3], y_pred[3])
    loss4 = tf.keras.losses.mse(y_true[4], y_pred[4])
    loss5 = tf.keras.losses.mse(y_true[5], y_pred[5])
    loss6 = tf.keras.losses.mse(y_true[6], y_pred[6])
    loss7 = tf.keras.losses.mse(y_true[7], y_pred[7])
    loss8 = tf.keras.losses.mse(y_true[8], y_pred[8])
    loss9 = tf.keras.losses.mse(y_true[9], y_pred[9])
    loss10 = tf.keras.losses.mse(y_true[10], y_pred[10])
    loss11 = tf.keras.losses.mse(y_true[11], y_pred[11])
    loss = 0.5*loss0 + 0.5*loss1

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
    model.compile(optimizer=adam, loss=custom_loss, metrics=["accuracy"])

    history = model.fit(
        gen(20, args.host, port=args.port, model=model),
        steps_per_epoch=1150//BATCH_SIZE, epochs=EPOCHS,
        validation_data=gen(20, args.host, port=args.port_val, model=model),
        validation_steps=1150//BATCH_SIZE, verbose=1, callbacks=callbacks_list)
      # steps_per_epoch=1150//BATCH_SIZE = 71  use this for longer training

    model.save('./saved_model/modelAB.h5')

    end = time.time()
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Training Time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))

    np.save('./saved_model/modelAB_loss', np.array(history.history['loss']))
    lossnpy = np.load('./saved_model/modelAB_loss.npy')
    plt.plot(lossnpy)  #--- lossnpy.shape = (10,)
    plt.draw() #plt.show()
    plt.pause(0.5)
    input("Press ENTER to exit ...")
    plt.close()
