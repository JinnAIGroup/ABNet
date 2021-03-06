'''
YPL, JLL, YJW, 2021.9.17, 10.12, 12.24
  #model.load_weights('./saved_model/supercombo079.keras')
  #----- Error: You use 'json mode' to covert this model

(YPN) jinn@Liu:~/YPN/ABNet$ python h5topbB4.py modelB4.pb
----- Frozen pb model inputs needed for converting pb to dlc:
[<tf.Tensor 'Input:0' shape=(None, 12, 128, 256) dtype=float32>, <tf.Tensor 'Input_1:0' shape=(None, 8) dtype=float32>, <tf.Tensor 'Input_2:0' shape=(None, 2) dtype=float32>, <tf.Tensor 'Input_3:0' shape=(None, 512) dtype=float32>]
----- Frozen pb model outputs needed for converting pb to dlc:
[<tf.Tensor 'Identity:0' shape=(None, 385) dtype=float32>, <tf.Tensor 'Identity_1:0' shape=(None, 386) dtype=float32>, <tf.Tensor 'Identity_2:0' shape=(None, 386) dtype=float32>, <tf.Tensor 'Identity_3:0' shape=(None, 58) dtype=float32>, <tf.Tensor 'Identity_4:0' shape=(None, 200) dtype=float32>, <tf.Tensor 'Identity_5:0' shape=(None, 200) dtype=float32>, <tf.Tensor 'Identity_6:0' shape=(None, 200) dtype=float32>, <tf.Tensor 'Identity_7:0' shape=(None, 8) dtype=float32>, <tf.Tensor 'Identity_8:0' shape=(None, 4) dtype=float32>, <tf.Tensor 'Identity_9:0' shape=(None, 32) dtype=float32>, <tf.Tensor 'Identity_10:0' shape=(None, 12) dtype=float32>, <tf.Tensor 'Identity_11:0' shape=(None, 512) dtype=float32>]
----- OK: pb is saved in ./saved_model

--- from pb to dlc
(snpe) jinn@Liu:~/snpe/dlc$ export ANDROID_NDK_ROOT=android-ndk-r22b
(snpe) jinn@Liu:~/snpe/dlc$ source snpe-1.48.0.2554/bin/envsetup.sh -t snpe-1.48.0.2554
(snpe) jinn@Liu:~/snpe/dlc$ snpe-tensorflow-to-dlc --input_network modelB4.pb \
--input_dim Input "1,12,128,256" --input_dim Input_1 "1,8" --input_dim Input_2 "1,2" --input_dim Input_3 "1,512" \
--out_node "Identity" --out_node "Identity_1" --out_node "Identity_2" --out_node "Identity_3" --out_node "Identity_4" \
--out_node "Identity_5" --out_node "Identity_6" --out_node "Identity_7" --out_node "Identity_8" --out_node "Identity_9" \
--out_node "Identity_10" --out_node "Identity_11" --output_path modelB4.dlc

ConverterError: ERROR_TF_LAYER_NO_INPUT_FOUND: FullyConnected layer modelB4/1_left_lane/MatMul requires at least one input layer

(snpe) jinn@Liu:~/snpe/dlc$ snpe-tensorflow-to-dlc --input_network modelB4.pb \
--input_dim Input "1,12,128,256" --input_dim Input_1 "1,8" --input_dim Input_2 "1,2" --input_dim Input_3 "1,512" \
--out_node "Identity" --output_path modelB4.dlc

(snpe) jinn@Liu:~/snpe/dlc$ snpe-dlc-viewer -i modelB4.dlc
'''
import os
import argparse
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from modelB4 import get_model

def h5_to_pb(pb_save_name):
  try:
    img_shape = (12, 128, 256)
    desire_shape = (8)
    traffic_convection_shape = (2)
    rnn_state_shape = (512)
    num_classes = 6
    model = get_model(img_shape, desire_shape, traffic_convection_shape, rnn_state_shape, num_classes)
    model.load_weights('./saved_model/modelB4.h5')
    model.summary()
  except ValueError:
    print("----- Error: You use 'json mode' to covert this model")
    return 1

  full_model = tf.function(lambda Input: model(Input))
  full_model = full_model.get_concrete_function([tf.TensorSpec(model.inputs[i].shape, model.inputs[i].dtype)
                                                 for i in range(len(model.inputs))])

  # Get frozen Concrete Function
  frozen_func = convert_variables_to_constants_v2(full_model)
  print('----- Frozen pb model graph: ')
  print(frozen_func.graph)
  layers = [op.name for op in frozen_func.graph.get_operations()]
  print("-" * 50)
  print("----- Frozen pb model inputs needed for converting pb to dlc: ")
  print(frozen_func.inputs)
  print("----- Frozen pb model outputs needed for converting pb to dlc: ")
  print(frozen_func.outputs)

  tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                    logdir="./saved_model",
                    name=pb_save_name,
                    as_text=False)
  print("----- OK: pb is saved in ./saved_model ")

if __name__=="__main__":
   parser = argparse.ArgumentParser(description='h5 to pb')
   parser.add_argument('pb', type=str, default="transition", help='save pb name')
   args = parser.parse_args()
   h5_to_pb(args.pb)
