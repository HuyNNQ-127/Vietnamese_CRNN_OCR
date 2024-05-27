from tensorflow.keras.layers import Dense, LSTM, BatchNormalization, Input, Conv2D, MaxPool2D
from tensorflow.keras.layers import Lambda, Bidirectional, Add, Activation
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
import os
import time
import cv2
import numpy as np

char_list = r' !"%&()+,-.\/0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz°²ÀÁÂÃÈÉÊÌÍÒÓÔÕÖÙÚÝàáâãèéêìíòóôõöùúüýĂăĐđĨĩŌŨũƠơƯưẠạẢảẤấẦầẨẩẪẫẬậẮắẰằẳẵẶặẸẹẺẻẼẽẾếỀềỂểỄễỆệỈỉỊịỌọỎỏỐốỒồỔổỖỗỘộỚớỜờỞởỠỡỢợỤụỦủỨứỪừỬửỮữỰựỲỳỴỶỷỸỹ–—“”'
max_label_len = 240
resize_max_width = 2167

import os
import cv2
import numpy as np

def pad_image(image_path, target_width, target_height, save_path):
    # Read the image
    image = cv2.imread(image_path)
    
    # Get the current dimensions
    height, width, _ = image.shape
    
    # Skip if dimensions exceed the maximum
    if height > target_height or width > target_width:
        print(f"Skipping {os.path.basename(image_path)} as dimensions exceed maximum")
        return
    
    # Calculate the horizontal padding amount
    horizontal_padding = max(0, target_width - width)
    
    # Create a white canvas of the target dimensions
    padded_image = np.ones((target_height, target_width, 3), dtype=np.uint8) * 255
    
    # Calculate the position to paste the original image
    x_offset = 0
    
    # Paste the original image onto the padded canvas
    padded_image[:height, :width] = image
    
    cv2.imwrite(save_path, padded_image)

# Target dimensions
target_width = 2167
target_height = 118

inputs = Input(shape=(118,2167,1))

# Block 1
x = Conv2D(64, (3,3), padding='same')(inputs)
x = MaxPool2D(pool_size=3, strides=3)(x)
x = Activation('relu')(x)
x_1 = x

# Block 2
x = Conv2D(128, (3,3), padding='same')(x)
x = MaxPool2D(pool_size=3, strides=3)(x)
x = Activation('relu')(x)
x_2 = x

# Block 3
x = Conv2D(256, (3,3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x_3 = x

# Block4
x = Conv2D(256, (3,3), padding='same')(x)
x = BatchNormalization()(x)
x = Add()([x,x_3])
x = Activation('relu')(x)
x_4 = x

# Block5
x = Conv2D(512, (3,3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x_5 = x

# Block6
x = Conv2D(512, (3,3), padding='same')(x)
x = BatchNormalization()(x)
x = Add()([x,x_5])
x = Activation('relu')(x)

# Block7
x = Conv2D(1024, (3,3), padding='same')(x)
x = BatchNormalization()(x)
x = MaxPool2D(pool_size=(3, 1))(x)
x = Activation('relu')(x)

# pooling layer with kernel size (2,2) to make the height/2 #(1,9,512)
x = MaxPool2D(pool_size=(3, 1))(x)

# # to remove the first dimension of one: (1, 31, 512) to (31, 512)
squeezed = Lambda(lambda x: K.squeeze(x, 1))(x)

# # # bidirectional LSTM layers with units=128
blstm_1 = Bidirectional(LSTM(512, return_sequences=True, dropout = 0.2))(squeezed)
blstm_2 = Bidirectional(LSTM(512, return_sequences=True, dropout = 0.2))(blstm_1)

# # this is our softmax character proprobility with timesteps
outputs = Dense(len(char_list)+1, activation = 'softmax')(blstm_2)

# model to be used at test time

act_model = Model(inputs, outputs)

labels = Input(name='the_labels', shape=[max_label_len], dtype='float32')


input_length = Input(name='input_length', shape=[1], dtype='int64')
label_length = Input(name='label_length', shape=[1], dtype='int64')

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([outputs, labels, input_length, label_length])

model = Model(inputs=[inputs, labels, input_length, label_length], outputs=loss_out)

#model.summary()

act_model.load_weights('E:\Python\CV\OCR_Vietnamese_Project\VN_handwritten_images\model_40(221).weights.h5')
'''
test_image_path = r"E:\Python\CV\OCR_Vietnamese_Project\reduced_img\test\17216.jpg"
#test_image_path = r"E:\Python\CV\OCR_Vietnamese_Project\VN_handwritten_images\padded_image.jpg"
#test_image_path = r"E:\Python\CV\OCR_Vietnamese_Project\receipt\data_line\data\mcocr_public_145013ckejs.jpg_1.png"

test_image = cv2.imread(test_image_path)

test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)

height, width = test_image.shape

test_image = np.pad(test_image, ((0,0),(0, 2167 - width)), 'median')

test_image = cv2.GaussianBlur(test_image, (5,5), 0)

test_image = cv2.adaptiveThreshold(test_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 4)

test_image = np.expand_dims(test_image, axis = 2)
test_image = np.expand_dims(test_image, axis = 0)

test_image = test_image / 255

start_time = time.time()
prediction = act_model.predict(test_image)
out = K.get_value(K.ctc_decode(prediction, input_length=np.ones(prediction.shape[0])*prediction.shape[1], greedy=True)[0][0])
one_predictions = []

i = 0
for x in out:
    print("predicted text = " , end = '')
    pred = ""
    for p in x:
        if int(p) != -1:
            pred += char_list[int(p)]
    print(pred)
    one_predictions.append(pred)
    i+=1
end_time = time.time()
execution_time = end_time - start_time

print("Time Execution: ", execution_time, " seconds")
'''

# Function to predict text from images in a folder
def predict_text_from_folder(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
            image_path = os.path.join(folder_path, filename)
            test_image = cv2.imread(image_path)
            test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
            height, width = test_image.shape
            test_image = np.pad(test_image, ((0,0),(0, 2167 - width)), 'median')
            test_image = cv2.GaussianBlur(test_image, (5,5), 0)
            test_image = cv2.adaptiveThreshold(test_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 4)
            test_image = np.expand_dims(test_image, axis=2)
            test_image = np.expand_dims(test_image, axis=0)
            test_image = test_image / 255
            prediction = act_model.predict(test_image)
            out = K.get_value(K.ctc_decode(prediction, input_length=np.ones(prediction.shape[0])*prediction.shape[1], greedy=True)[0][0])
            predictions = []
            for x in out:
                pred = ""
                for p in x:
                    if int(p) != -1:
                        pred += char_list[int(p)]
                predictions.append(pred)
            print(f"Image: {filename}, Predicted Text: {predictions}")
        else:
            continue

# Folder containing images for prediction
folder_path = r"E:\Python\CV\OCR_Vietnamese_Project\VN_handwritten_images\east_ocr\img"

# Call the function to predict text from images in the folder
predict_text_from_folder(folder_path)