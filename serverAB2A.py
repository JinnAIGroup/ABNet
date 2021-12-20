"""   JLL, SLT, 2021.12.7, 12.20
AB2A: modelAB = UNet + RNN + PoseNet combines models A and B but runs only model A data
from /home/jinn/YPN/OPNet/serverA4.py, serverB3.py

Run: on 2 terminals
  (YPN) jinn@Liu:~/YPN/OPNet$ python serverAB2A.py --port 5557
  (YPN) jinn@Liu:~/YPN/OPNet$ python serverAB2A.py --port 5558 --validation
Input:
  bRGB (874, 1164, 3) = (H, W, C) <=> bYUV (1311, 1164) <=>  CbYUV (6, 291,  582) = (C, H, W) [key: 1311 =  874x3/2]
  sRGB (256,  512, 3) = (H, W, C) <=> sYUV  (384,  512) <=>  CsYUV (6, 128,  256) = (C, H, W) [key:  384 =  256x3/2]
  /home/jinn/dataB/UHD--2018-08-02--08-34-47--32/yuv.h5
  /home/jinn/dataB/UHD--2018-08-02--08-34-47--32/pathdata.h5
  /home/jinn/dataB/UHD--2018-08-02--08-34-47--32/radardata.h5
Output:
  X_batch.shape = (none, 2x6, 128, 256)  (num_channels = 6, 2 yuv images)
  Y_batch.shape = (none, 2x56)  (56 = 51 pathdata + 5 radardata)
"""
import os
import zmq
import six
import numpy
import random
import logging
import argparse
from datagenAB2A import datagen
from numpy.lib.format import header_data_from_array_1_0

BATCH_SIZE = 16
  # Project A Part
DATA_DIR_Imgs = '/home/jinn/dataAll/comma10k/Ximgs_yuv/'  # Ximgs with 10 images only for debugging
DATA_DIR_Msks = '/home/jinn/dataAll/comma10k/Xmasks/'
IMAGE_H = 128
IMAGE_W = 256
MASK_H = 256
MASK_W = 512
class_values = [41,  76,  90, 124, 161, 0] # 0 added for padding
all_img_dirs = os.listdir(DATA_DIR_Imgs)
all_msk_dirs = os.listdir(DATA_DIR_Msks)
all_images = [DATA_DIR_Imgs+i for i in all_img_dirs]
all_masks = [DATA_DIR_Msks+i for i in all_msk_dirs]
all_images = sorted(all_images)
all_masks = sorted(all_masks)
train_lenA = int(0.8*len(all_images))
valid_lenA = int(0.2*len(all_images))
train_images = all_images[: train_lenA]
valid_images = all_images[train_lenA: train_lenA + valid_lenA]
train_masks = all_masks[: train_lenA]
valid_masks = all_masks[train_lenA: train_lenA + valid_lenA]

if six.PY3:
  buffer_ = memoryview
else:
  buffer_ = buffer  # noqa

logger = logging.getLogger(__name__)

def send_arrays(socket, arrays, stop=False):
  if arrays:
      # The buffer protocol only works on contiguous arrays
    arrays = [numpy.ascontiguousarray(array) for array in arrays]
  if stop:
    headers = {'stop': True}
    socket.send_json(headers)
  else:
    headers = [header_data_from_array_1_0(array) for array in arrays]
    socket.send_json(headers, zmq.SNDMORE)
    for array in arrays[:-1]:
      socket.send(array, zmq.SNDMORE)
    socket.send(arrays[-1])

def recv_arrays(socket):
  headers = socket.recv_json()
  if 'stop' in headers:
    raise StopIteration

  arrays = []
  for header in headers:
    data = socket.recv()
    buf = buffer_(data)
    array = numpy.frombuffer(buf, dtype=numpy.dtype(header['descr']))
    array.shape = header['shape']
    if header['fortran_order']:
      array.shape = header['shape'][::-1]
      array = array.transpose()
    arrays.append(array)

  return arrays

def client_generator(port=5557, host="localhost", hwm=20):
  context = zmq.Context()
  socket = context.socket(zmq.PULL)
  socket.set_hwm(hwm)
  socket.connect("tcp://{}:{}".format(host, port))
  logger.info('client started')
  while True:
    data = recv_arrays(socket)
    yield tuple(data)

def start_server(data_stream, port=5557, hwm=20):
  logging.basicConfig(level='INFO')
  context = zmq.Context()
  socket = context.socket(zmq.PUSH)
  socket.set_hwm(hwm)
  socket.bind('tcp://*:{}'.format(port))

    # it = itertools.tee(data_stream)
  it = data_stream
  logger.info('server started')
  while True:
    try:
      data = next(it)
      stop = False
      logger.debug("sending {} arrays".format(len(data)))
    except StopIteration:
      it = data_stream
      data = None
      stop = True
      logger.debug("sending StopIteration")

    send_arrays(socket, data, stop=stop)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data Server')
    parser.add_argument('--port', dest='port', type=int, default=5557, help='Port of the ZMQ server')
    parser.add_argument('--buffer', dest='buffer', type=int, default=20, help='High-water mark. Increasing this increses buffer and memory usage.')
    parser.add_argument('--validation', dest='validation', action='store_true', default=False, help='Serve validation dataset instead.')
    args, more = parser.parse_known_args()

      # Project B Part
    all_dirs = os.listdir('/home/jinn/dataB')
    all_yuvs = ['/home/jinn/dataB/'+i+'/yuv.h5' for i in all_dirs]
      #print('#---  all_yuvs =', all_yuvs)
    random.seed(0) # makes the random numbers predictable
    random.shuffle(all_yuvs)

    train_len  = int(0.5*len(all_yuvs))
    valid_len  = int(0.5*len(all_yuvs))
    train_files = all_yuvs[: train_len]
    valid_files = all_yuvs[train_len: train_len + valid_len]
    print('#---serverAB  len(all_yuvs) =', len(all_yuvs))
    print('#---serverAB  len(train_files) =', len(train_files))
    print('#---serverAB  len(valid_files) =', len(valid_files))

    if args.validation:
        camera_files = valid_files
        images = valid_images
        masks  = valid_masks
    else:
        camera_files = train_files
        images = train_images
        masks  = train_masks

    print('#---serverAB  camera_files =', camera_files)
    data_s = datagen(BATCH_SIZE, camera_files, images, masks, IMAGE_H, IMAGE_W, MASK_H, MASK_W, class_values)
    start_server(data_s, port=args.port, hwm=args.buffer)
