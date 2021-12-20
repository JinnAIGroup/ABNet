'''   JLL, SLT, YJW, 2021.9.9, 11.4, 12.19
AB2B: modelAB = UNet + RNN + PoseNet combines models A and B but runs only model B data
from /home/jinn/YPN/OPNet/train_modelA4.py, train_modelB3.py
supercombo: https://drive.google.com/file/d/1L8sWgYKtH77K6Kr3FQMETtAWeQNyyb8R/view
output.txt: https://github.com/JinnAIGroup/OPNet/blob/main/output.txt

1. Same as supercombo I/O
2. AB Tasks: multiclass semantic segmentation + temporal state (features) + path planning (PP)
   The temporal state (a 512 array) represents (are features of) all path planning details:
   path prediction, lane detection, lead car xyva, desire etc., i.e., outs[0]...[10] in output.txt.
3. Input:
   inputs[ 0 ].shape = (1, 12, 128, 256)   # 12 = 2 frames x 6 channels (YUV_I420: Y=4, U=1, V=1)
   inputs[ 1 ].shape = (1, 8)
   inputs[ 2 ].shape = (1, 2)
   inputs[ 3 ].shape = (1, 512)
4. Output:
   outs[ 0 ].shape = (1, 385) = Ytrue0
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
   outs[ 12 ].shape = (1, 512) = Ymasks

Projects:
Goal: modelAB successfully drives my car like supercombo does
1. How to train modelAB to a satisfactory validation loss in semantic segmentation (SS)?
2. How to train modelAB using both A and B data?
3. How to make modelAB predict 3D SS?
   Deep learning for monocular depth estimation: A review
4. How to combine PP outputs (from PoseNet) with SS outputs (UNet)?
5. How to extend A and B data to make better trained modelAB?
6. How to use modelAB to do lateral and longitudinal controls?

Run:
   (YPN) jinn@Liu:~/YPN/ABNet$ python modelAB2B.py
'''
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

#def UNet(x0, desire, traffic_convection, rnn_state, num_classes):
def UNet(x0, num_classes):
      ### [First half of the network: spatial contraction] ###
    x_crop = []
      # Entry block
    x = layers.Conv2D(32, 3, strides=2, padding="same")(x0)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x_crop.append(x)
    previous_block_activation = x  # Set aside residual

      # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [64, 128, 256]:

        x = layers.Conv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        x = layers.Conv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D(2)(x)

        residual = layers.Conv2D(filters, 3, padding="same")(previous_block_activation)
        x_crop.append(x)
        residual = layers.MaxPooling2D(2)(residual)

        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

      ### [Second half of the network: spatial expansion] ###
    for filters in [256, 128, 64, 32]:
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x_crop_i = x_crop.pop()
        x = tf.concat(axis=-1, values = [x_crop_i,x])
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        if (filters == 32):
            #out0_out11 = PoseNet(x, desire, traffic_convection, rnn_state, num_classes)
            x_to_PN = x

        shape1 = x.shape.as_list()
        x = tf.compat.v1.image.resize_bilinear(x, (shape1[1]*2, shape1[2]*2))

        residual = layers.Conv2DTranspose(filters, 3, padding="same")(previous_block_activation)
        shape = previous_block_activation.shape.as_list()

        residual = tf.compat.v1.image.resize_bilinear(residual, (shape[1]*2, shape[2]*2))
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

      # Per-pixel classification layer
    out12 = layers.Conv2DTranspose(2*num_classes, 3, strides=2, activation="softmax", padding="same")(x)
    #return out0_out11, out12
    return x_to_PN, out12

#def PoseNet(x, desire, traffic_convection, rnn_state, num_classes):
def PoseNet(x, num_classes):
      # Add a per-pixel classification layer (UNet final layer)
    x = layers.Conv2D(2*num_classes, 3, activation="softmax", padding="same")(x)

      # Add layers for PN
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(32, 1, strides=2, padding="same")(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(64, 1, strides=2, padding="same")(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(128, 1, strides=2, padding="same")(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(64, 1, strides=2, padding="same", name="58")(x)
    x = layers.Activation("relu", name="59")(x)
    x = layers.Conv2D(128, 1, strides=2, padding="same", name="60")(x)
    x= layers.Activation("relu", name="61")(x)

    x_to_RNNfk2fk3 = layers.Flatten(name="vision_features")(x)
    #vf = layers.Flatten(name="vision_features")(x)
    #out11 = RNN(vf, desire, traffic_convection, rnn_state)
    #out0_out7  = fork1(out11)
    #out8_out9  = fork2(vf)
    #out10      = fork3(vf)
    #out0_out11 = layers.Concatenate(axis=-1)([out0_out7, out8_out9, out10, out11])
    #return out0_out11
    return x_to_RNNfk2fk3

def RNN(x, desire, traffic_convection, rnn_state):
    desire1 = layers.Dense(use_bias=False, units=8)(desire)
    traffic_convection1 = layers.Dense(use_bias=False, units = 2)(traffic_convection)
    x_concate = layers.Concatenate(axis=-1)([desire1, traffic_convection1, x])
    x_dense = layers.Dense(use_bias=False, units=1024)(x_concate)
    x_1 = layers.Activation("relu")(x_dense)

    rnn_rz = layers.Dense(use_bias=False, units=512)(rnn_state)
    rnn_rr = layers.Dense(use_bias=False, units=512)(rnn_state)
    snpe_pleaser = layers.Dense(use_bias=False, units=512)(rnn_state)
    rnn_rh = layers.Dense(use_bias=False, units = 512)(snpe_pleaser)

    rnn_z = layers.Dense(use_bias=False, units=512)(x_1)
    rnn_h = layers.Dense(use_bias=False, units=512)(x_1)
    rnn_r = layers.Dense(use_bias=False, units=512)(x_1)

    add = layers.add([rnn_rz , rnn_z])
    activation_1 = layers.Activation("sigmoid")(add)
    add_1 = layers.add([rnn_rr , rnn_r])

    activation = layers.Activation("sigmoid")(add_1)
    multiply = rnn_rh*activation
    add_2 = layers.add([rnn_h , multiply])

    activation_2 = layers.Activation("tanh")(add_2)
    one_minus = layers.Dense(use_bias=False, units=512)(activation_1)
    multiply_2 = one_minus*activation_2
    multiply_1 = snpe_pleaser*activation_1
    out11 = layers.add([multiply_1 , multiply_2])
    return out11

def fork1(x):
    xp = layers.Dense(256, activation='relu', name="1_path")(x)
    xp = layers.Dense(256, activation='relu', name="2_path")(xp)
    xp = layers.Dense(256, activation='relu', name="3_path")(xp)
    x0 = layers.Dense(128, activation='relu', name="final_path")(xp)

    xll = layers.Dense(256, activation='relu', name="1_left_lane")(x)
    xll = layers.Dense(256, activation='relu', name="2_left_lane")(xll)
    xll = layers.Dense(256, activation='relu', name="3_left_lane")(xll)
    x1 = layers.Dense(128, activation='relu', name="final_left_lane")(xll)

    xrl = layers.Dense(256, activation='relu', name="1_right_lane")(x)
    xrl = layers.Dense(256, activation='relu', name="2_right_lane")(xrl)
    xrl = layers.Dense(256, activation='relu', name="3_right_lane")(xrl)
    x2 = layers.Dense(128, activation='relu', name="final_right_lane")(xrl)

    xl = layers.Dense(256, activation='relu', name="1_lead")(x)
    xl = layers.Dense(256, activation='relu', name="2_lead")(xl)
    xl = layers.Dense(256, activation='relu', name="3_lead")(xl)
    x3 = layers.Dense(128, activation='relu', name="final_lead")(xl)

    xlx = layers.Dense(256, activation='relu', name="1_long_x")(x)
    xlx = layers.Dense(256, activation='relu', name="2_long_x")(xlx)
    xlx = layers.Dense(256, activation='relu', name="3_long_x")(xlx)
    x4 = layers.Dense(128, activation='relu', name="final_long_x")(xlx)

    xla = layers.Dense(256, activation='relu', name="1_long_a")(x)
    xla = layers.Dense(256, activation='relu', name="2_long_a")(xla)
    xla = layers.Dense(256, activation='relu', name="3_long_a")(xla)
    x5 = layers.Dense(128, activation='relu', name="final_long_a")(xla)

    xlv = layers.Dense(256, activation='relu', name="1_long_v")(x)
    xlv = layers.Dense(256, activation='relu', name="2_long_v")(xlv)
    xlv = layers.Dense(256, activation='relu', name="3_long_v")(xlv)
    x6 = layers.Dense(128, activation='relu', name="final_long_v")(xlv)

    xds = layers.Dense(128, activation='relu', name="1_desire_state")(x)
    x7 = layers.Dense(8, name="final_desire_state")(xds)

    out0 = layers.Dense(385, name="path")(x0)
    out1 = layers.Dense(386, name="left_lane")(x1)
    out2 = layers.Dense(386, name="right_lane")(x2)
    out3 = layers.Dense(58, name="lead")(x3)
    out4 = layers.Dense(200, name="long_x")(x4)
    out5 = layers.Dense(200, name="long_a")(x5)
    out6 = layers.Dense(200, name="long_v")(x6)
    out7 = layers.Softmax(axis=-1, name="desire_state")(x7)
    out0_out7 = layers.Concatenate(axis=-1)([out0, out1, out2, out3, out4, out5, out6, out7])
    return out0_out7

def fork2(x):
    x1 = layers.Dense(256, activation='relu', name="meta0")(x)
    out8 = layers.Dense(4, activation='sigmoid', name="meta")(x1)
    dp1 = layers.Dense(32, name="desire_final_dense")(x1)
    dp2 = layers.Reshape((4, 8), name="desire_reshape")(dp1)
    dp3 = layers.Softmax(axis=-1, name="desire_pred0")(dp2)
    out9 = layers.Flatten(name="desire_pred")(dp3)
    out8_out9 = layers.Concatenate(axis=-1)([out8, out9])
    return out8_out9

def fork3(x):
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(32, activation='relu')(x)
    out10 = layers.Dense(12, name="pose")(x)
    return out10

def get_model(img_shape, desire_shape, traffic_convection_shape, rnn_state_shape, num_classes):
    imgs = keras.Input(shape=img_shape, name="pic")
    print('#---modelAB imgs.shape =', imgs.shape)
    in0 = layers.Permute((2, 3, 1))(imgs)
    print('#---modelAB in0.shape =', in0.shape)
    in1 = keras.Input(shape=desire_shape, name="desire")
    in2 = keras.Input(shape=traffic_convection_shape, name="traffic_convection")
    in3 = keras.Input(shape=rnn_state_shape, name="rnn_state")

    #out0_out11, out12 = UNet(in0, in1, in2, in3, num_classes)
    x_to_PN, out12 = UNet(in0, num_classes)
    x_to_RNNfk2fk3 = PoseNet(x_to_PN, num_classes)
    out11     = RNN(x_to_RNNfk2fk3, in1, in2, in3)
    out0_out7 = fork1(out11)
    out8_out9 = fork2(x_to_RNNfk2fk3)
    out10     = fork3(x_to_RNNfk2fk3)
    out0_out11 = layers.Concatenate(axis=-1)([out0_out7, out8_out9, out10, out11])

      # Define the model
    model = keras.Model(inputs=[imgs, in1, in2, in3], outputs=[out0_out11, out12], name='modelAB')
    #model = keras.Model(inputs=[imgs, in1, in2, in3], outputs=[out12], name='modelAB')
    return model

if __name__=="__main__":
      # Build model
    img_shape = (12, 128, 256)
    desire_shape = (8)
    traffic_convection_shape = (2)
    rnn_state_shape = (512)
    num_classes = 6

    model = get_model(img_shape, desire_shape, traffic_convection_shape, rnn_state_shape, num_classes)
    model.summary()

    #tf.keras.utils.plot_model(model, to_file='./saved_model/modelAB.png')
    model.save('./saved_model/modelAB.h5')
