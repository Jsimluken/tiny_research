import os
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Input, BatchNormalization, Flatten, Dense,Conv2DTranspose,Reshape,Concatenate,Activation
from tensorflow.keras.models import Model
from enum import Enum
import tqdm
import datetime
import numpy as np
import time


def create_encoder(initial_filter = 32):
    input_ = Input((8192,1),name="audio")
    x = input_
    encs = []
    num = 7
    for i in range(num):
        x = Conv1D(initial_filter*(i+1),15,strides=2,padding="same",activation="elu")(x)
        x = BatchNormalization()(x)
        encs.append(x)
    return Model(input_,encs,name="encoder")

def create_decoder(initial_filter = 32):
  num = 7
  inputs = [Input((64*2**i,initial_filter*(num-i)),name=f"enc{i}") for i in range(7)]
  pitch = Input((128,),name="pitch")
  instruments = Input((11,),name="instrument")

  base = 64
  enc = Reshape((1,64,initial_filter*7),name="renc0")(inputs[0])
  infos = Concatenate(axis=-1)([pitch,instruments])
  infos = Dense(64,activation="elu")(infos)
  infos = Reshape((1,64,1))(infos)
  enc = Concatenate(axis=-1)([enc,infos])
  #dec = Conv2DTranspose(128,3,strides=(1,2))(enc)

  for i in range(1,num):
    dec = Conv2DTranspose(initial_filter*(num-i),7,strides=(1,2),padding="same",activation="relu")(enc)
    dec = BatchNormalization()(dec)
    enc = Reshape((1,64*2**i,initial_filter*(num-i)))(inputs[i])
    enc = Concatenate(axis=-1)([dec,enc])
  out = Conv2DTranspose(1,7,strides=(1,2),padding="same",activation="relu")(enc)
  out = BatchNormalization()(out)
  out = Reshape((8192,1),name="out_reshape")(out)
  #out = Activation("softmax",name="out")(out)
  out = Conv1D(1,7,padding="same",activation="tanh")(out)

  
  return Model(inputs=[inputs,pitch,instruments],outputs=out,name="decoder")


def create_classifier(initial_filter = 32):
    input_ = Input((64,initial_filter*7),name="enc_input")
    instruments_input = Input((11,))
    instruments_i = Dense(64,activation="elu")(instruments_input)
    instruments_i = Reshape((64,1))(instruments_i)
    
    #pitch = Conv1D(128,5,strides=2,activation="elu")(input_)


    instruments = Conv1D(64,5,strides=2,activation="elu")(input_)
    instruments = Conv1D(32,5,strides=2,activation="elu")(instruments)
    instruments = Flatten()(instruments)
    instruments = Dense(11, activation="softmax")(instruments)
    
    x = Concatenate(axis=-1)([input_,instruments_i])
    pitch = Conv1D(64,5,strides=2,activation="elu")(x)
    pitch = BatchNormalization()(pitch)
    pitch = Conv1D(32,5,strides=2,activation="elu")(pitch)
    pitch = BatchNormalization()(pitch)
    pitch = Flatten()(pitch)
    pitch = Dense(128, activation="softmax")(pitch)
    
    fake = Conv1D(64,5,strides=2,activation="elu")(x)
    fake = Conv1D(32,5,strides=2,activation="elu")(fake)
    fake = Flatten()(fake)
    fake = Dense(1,"sigmoid")(fake)

    #x = Flatten()(input_)
    #x = Concatenate(axis=-1)([x,instruments_input])
    #pitch = Dense(128, activation="softmax")(x)
    #
    #fake = Dense(1,activation="sigmoid")(x)
    return Model([input_,instruments_input],[pitch,instruments,fake])