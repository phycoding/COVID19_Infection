from PIL import Image, ImageOps
import streamlit as st
import keras
import tensorflow as tf
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt

#Loss function

def dice_coeff(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return score

def dice_loss(y_true, y_pred):
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss


def bce_dice_loss(y_true, y_pred):
    loss = 0.5* tf.keras.losses.binary_crossentropy(y_true, y_pred) + 0.5*dice_loss(y_true, y_pred)
    return loss

def tversky_loss(y_true, y_pred):
    alpha = 0.5
    beta  = 0.5
    
    ones = K.ones(K.shape(y_true))
    p0 = y_pred      # proba that voxels are class i
    p1 = ones-y_pred # proba that voxels are not class i
    g0 = y_true
    g1 = ones-y_true
    
    num = K.sum(p0*g0, (0,1,2))
    den = num + alpha*K.sum(p0*g1,(0,1,2)) + beta*K.sum(p1*g0,(0,1,2))
    
    T = K.sum(num/den) # when summing over classes, T has dynamic range [0 Ncl]
    
    Ncl = K.cast(K.shape(y_true)[-1], 'float32')
    return Ncl-T

def weighted_bce_loss(y_true, y_pred, weight):
    epsilon = 1e-7
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
    logit_y_pred = K.log(y_pred / (1. - y_pred))
    loss = weight * (logit_y_pred * (1. - y_true) + 
                     K.log(1. + K.exp(-K.abs(logit_y_pred))) + K.maximum(-logit_y_pred, 0.))
    return K.sum(loss) / K.sum(weight)

def weighted_dice_loss(y_true, y_pred, weight):
    smooth = 1.
    w, m1, m2 = weight, y_true, y_pred
    intersection = (m1 * m2)
    score = (2. * K.sum(w * intersection) + smooth) / (K.sum(w * m1) + K.sum(w * m2) + smooth)
    loss = 1. - K.sum(score)
    return loss

def weighted_bce_dice_loss(y_true, y_pred):
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    # if we want to get same size of output, kernel size must be odd
    averaged_mask = K.pool2d(
            y_true, pool_size=(50, 50), strides=(1, 1), padding='same', pool_mode='avg')
    weight = K.ones_like(averaged_mask)
    w0 = K.sum(weight)
    weight = 5. * K.exp(-5. * K.abs(averaged_mask - 0.5))
    w1 = K.sum(weight)
    weight *= (w0 / w1)
    loss = 0.5*weighted_bce_loss(y_true, y_pred, weight) + 0.5*dice_loss(y_true, y_pred)
    return loss


#model backbone
new_dim = 224
inputs = tf.keras.Input((224, 224, 1))
# s = Lambda(lambda x: x / 255) (inputs)

# def mish(inputs):
#     return inputs * tf.math.tanh(tf.math.softplus(inputs))
from keras.layers import Conv2D,Conv2D, BatchNormalization,MaxPooling2D, Dropout, concatenate, Conv2DTranspose
c1 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal") (inputs)
c1 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal") (c1)
c1 = BatchNormalization()(c1)
p1 = MaxPooling2D((2, 2)) (c1)
p1 = Dropout(0.25)(p1)

c2 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal") (p1)
c2 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal") (c2)
c2 = BatchNormalization()(c2)
p2 = MaxPooling2D((2, 2)) (c2)
p2 = Dropout(0.25)(p2)

c3 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal") (p2)
c3 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal") (c3)
c3 = BatchNormalization()(c3)
p3 = MaxPooling2D((2, 2)) (c3)
p3 = Dropout(0.25)(p3)

c4 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal") (p3)
c4 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal") (c4)
c4 = BatchNormalization()(c4)
p4 = MaxPooling2D(pool_size=(2, 2)) (c4)
p4 = Dropout(0.25)(p4)

c5 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal") (p4)
c5 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal") (c5)

u6 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c5)
u6 = concatenate([u6, c4])
u6 = BatchNormalization()(u6)
c6 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal") (u6)
c6 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal") (c6)


u7 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c6)
u7 = concatenate([u7, c3])
u7 = BatchNormalization()(u7)
c7 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal") (u7)
c7 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal") (c7)


u8 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c7)
u8 = concatenate([u8, c2])
u8 = BatchNormalization()(u8)
c8 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal") (u8)
c8 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal") (c8)


u9 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c8)
u9 = concatenate([u9, c1], axis=3)
u9 = BatchNormalization()(u9)
c9 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal") (u9)
c9 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal") (c9)

outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)

model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 0.0005), loss=bce_dice_loss, metrics=[dice_coeff])
model.summary()
model.load_weights("unet_covid_fold1.hdf5")

st.title("Covid19 Lungs Infection Segmentator")
placeholder = st.empty()
st.sidebar.title("Please select the mode")

#Load image function
def load_image(image_file):
    img = Image.open(image_file)
    return img

#Preprocessing centercrop
def center_crop(img, new_width=None, new_height=None):        

    img_width, img_height = img.size
    center_cropped_img = img.crop(((img_width - new_width) // 2,
                         (img_height - new_height) // 2,
                         (img_width + new_width) // 2,
                         (img_height + new_height) // 2))

    return center_cropped_img

placeholder.image(load_image('lungs.jpg'), use_column_width=True)
        

menu = ["Image Upload","Camera"]
choice = st.sidebar.selectbox("Mode",menu)

if choice == "Image Upload":
        st.subheader("Image")
        image_file = st.file_uploader("Upload an image",type=["jpg","png"])
        if image_file is not None:
            file_details = {"filename":image_file.name, "filetype":image_file.type,"filesize":image_file.size}
            st.write(file_details)
            img = load_image(image_file)
            img_cropped = center_crop(img, new_width=224, new_height=224)
            img_cropped_gray = ImageOps.grayscale(img_cropped)
            img_cropped_gray = np.array(img_cropped_gray).reshape(1,224,224,1)
            infection = model.predict(img_cropped_gray)
            fig = plt.figure(figsize=(10,10))
            st.image(img_cropped)
            plt.imshow(img_cropped_gray.reshape(224,224),cmap = "bone")
            plt.imshow(infection.reshape(224,224), alpha=0.5, cmap = "nipy_spectral")
            plt.axis('off')
            st.pyplot(fig)
            #st.image(img, caption="Original Image", use_column_width=True)
            #st.image(infection, caption="Infection", use_column_width=True)
            placeholder.empty()
            
elif choice == "Camera":
    st.write("Camera")
    picture = st.camera_input("Take a picture")
    if picture:
        placeholder = st.empty()
        img = Image.open(picture)
        img_cropped = center_crop(img, new_width=224, new_height=224)
        img_cropped_gray = ImageOps.grayscale(img_cropped)
        img_cropped_gray = np.array(img_cropped_gray).reshape(1,224,224,1)
        infection = model.predict(img_cropped_gray)
        fig = plt.figure(figsize=(10,10))
        st.image(img_cropped)
        plt.imshow(img_cropped_gray.reshape(224,224),cmap = "bone")
        plt.imshow(infection.reshape(224,224), alpha=0.5, cmap = "nipy_spectral")
        plt.axis('off')
        st.pyplot(fig)
        
