# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 19:20:12 2017

@author: kevin
"""

import numpy as np

# define the gates

def tanh(z,derivative=False):
    if derivative==True:
        return 1-z**2
    return np.tanh(z)

def sigmoid(z,derivative=False):
    if derivative==True:
        return z*(1-z)
    return 1/(1+np.exp(-z))

def softmax(o):
    p=np.exp(o)
    return p/np.exp(p)

#hyperparameters

nh=100# number of neurons in the hidden layer
seq_len=25# sequence length
gamma=0.1# learning rate
beta1=0.002# adagrad factor
beta2=0.9# rmsprop factor

#define the dataset

data = open('input.txt', 'r').read() # should be simple plain text file
chars = list(set(data))
textSize, vocabSize = len(data), len(chars)
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }

position=0# position of the lecture
time=0# running iteration
BackPropagationOnly=True

#define the weights
U=0.1*np.random.random((nh,vocabSize))-0.05# from input to hidden
W=0.1*np.random.random((nh,nh))-0.05#from hidden to hidden
V=0.1*np.random.random((vocabSize,nh))-0.05#from hidden to output
b1=np.ones(nh)
b2=np.ones(vocabSize)

#generate the layers
s=np.zeros((seq_len,nh))# hidden layers
o=np.zeros((seq_len,vocabSize))# output layer
p=np.zeros((seq_len,nh))#prob layer
do=np.zeros((seq_len,vocabSize))

smoothLoss=0
losses=[]

while True:
    ixes=[char_to_ix[char] for char in data[position:position+seq_len+1]]
    x=np.zeros((seq_len,vocabSize))
    trainLoss=0
    
    sprev=s[-1].copy()
    #feedforward
    for t in range(seq_len):
        x[t][ixes[t]]=1
        s[t]=tanh(np.dot(W,s[t-1])+np.dot(U,x[t])+b1)
        o[t]=np.dot(V,s[t])+b2
        p[t]=softmax(o[t])
        #loss
        trainLoss+=-np.log(p[t])[ixes[t+1]]
    if time==0:
        smoothLoss=trainLoss
    else:
        smoothLoss=0.999*smoothLoss+0.001*trainLoss
    losses.append(smoothLoss)
    
    #generate the weights gradients
    dU=np.zeros((nh,vocabSize))
    dW=np.zeros((nh,nh))
    dV=np.zeros((vocabSize,nh))
    db1=np.zeros(nh)
    db2=np.zeros(vocabSize)

    if BackPropagationOnly==False:    
        #generate the firsts momentums
        etaU=np.zeros((nh,vocabSize))
        etaW=np.zeros((nh,nh))
        etaV=np.zeros((vocabSize,nh))
        etab1=np.zeros(nh)
        etab2=np.zeros(vocabSize)
        
        #generate the seconds momentums
        vU=np.zeros((nh,vocabSize))
        vW=np.zeros((nh,nh))
        vV=np.zeros((vocabSize,nh))
        vb1=np.zeros(nh)
        vb2=np.zeros(vocabSize)
    
    dsaux=np.zeros(nh)
    #feedforward
    for t in reversed(range(seq_len)):
        do=p[t].copy()
        do[ixes[t+1]]+=-1
        db2+=do
        dV+=np.outer(do,s[t])
        ds=np.dot(do,V)+dsaux
        dsaux=np.dot(ds*tanh(s[t],True),W)
        if t>0: dW+=np.dot(ds*tanh(s[t],True),s[t-1])
        else: dW+=np.dot(ds*tanh(s[t],True),sprev)
        dU+=np.dot(ds*tanh(s[t],True),x[t])
        db2+=ds
        
    for dparam in [dU,dW,dV,db1,db2]:
        np.clip(dparam,-5,5,out=dparam)

    if BackPropagationOnly==True:
        U+=-gamma*dU
        W+=-gamma*dW
        V+=-gamma*dV
        b1+=-gamma*db1
        b2+=-gamma*db2    
    else:
        etaU=beta1*etaU+(1-beta1)*dU
        etaW=beta1*etaW+(1-beta1)*dW
        etaV=beta1*etaV+(1-beta1)*dV
        etab1=beta1*etab1+(1-beta1)*db1
        etab2=beta1*etab2+(1-beta1)*db2
        
        vU=beta2*vU+(1-beta1)*dU**2
        vW=beta2*vW+(1-beta1)*dW**2
        vV=beta2*vV+(1-beta1)*dV**2
        v1=beta2*vb1+(1-beta1)*db1**2
        v2=beta2*vb2+(1-beta1)*db2**2
    
    
        #updating the weights
        U+=-gamma*etaU*dU/(np.sqrt(vU)+1e-8)
        W+=-gamma*etaW*dW/(np.sqrt(vW)+1e-8)
        V+=-gamma*etaV*dV/(np.sqrt(vV)+1e-8)
        b1+=-gamma*etab1*db1/(np.sqrt(vb1)+1e-8)
        b2+=-gamma*etab2*db2/(np.sqrt(vb2)+1e-8)
