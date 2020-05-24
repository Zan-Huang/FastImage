

from model.srgan import generator
from PIL import Image
import numpy as np
import common
import tensorflow as tf
import sys

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus and len(sys.argv)> 1 and sys.argv[1].startswith("-a"):
    print("allowing growth")
    growth = True
else:
    print("nogrowth")
    growth = False

try:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, growth)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
except RuntimeError as e:
    print(e)

def load_image(path):
    return np.array(Image.open(path))

model = generator()
model.load_weights('../weights/srgan/gan_generator.h5')

lr = load_image('demo/0869x4-crop.png')
sr = common.resolve_single(model, lr)

print(sr)
