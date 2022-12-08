import glob
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt

train_img_data = []

def convert_to_data(filepath):
    img_filename = filepath
    im = Image.open(img_filename)
    im = im.getdata()
    for f in im:
        train_img_data.append(f)



train_images = []
test_images = []

class_names = ['01_palm', '02_I', '03_fist', '04_fist_moved', '05_thumb',
               '06_index', '07_ok', '08_palm_moved', '09_c', '10_down']
#Initialize training dataset
for filename in glob.glob('KaggleData/00/*/*.png'): #assuming gif
    im=Image.open(filename)
    train_images.append(im)
    convert_to_data(filename)

#Initialize testing dataset
for filename in glob.glob('KaggleData/00/*/*.png'): #assuming gif
    im=Image.open(filename)
    test_images.append(im)
    convert_to_data(filename)

'''train_images = train_images / 255.0

test_images = test_images / 255.0'''

'''plt.figure(figsize=(10,10))
for i in range(10):
    plt.subplot(5,2,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)`
    plt.imshow(train_images[i * 200], cmap=plt.cm.binary)
    plt.xlabel(class_names[i])
plt.show()'''

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_img_data, class_names, epochs=10)

