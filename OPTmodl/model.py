"""
This code will create the model described in our following paper
MoDL: Model-Based Deep Learning Architecture for Inverse Problems
by H.K. Aggarwal, M.P. Mani, M. Jacob from University of Iowa.

Paper dwonload  Link:     https://arxiv.org/abs/1712.02862

@author: haggarwal
@Edited by Obanmarcos
"""
import tensorflow as tf
import numpy as np
from os.path import expanduser      
home = expanduser("~")      # searches for home user in directory
epsilon=1e-5            # epsilon 
TFeps=tf.constant(1e-5,dtype=tf.float32) # constant value as a tensor object


# function c2r contatenate complex input as new axis two two real inputs
c2r=lambda x:tf.stack([tf.math.real(x),tf.math.imag(x)],axis=-1)
#r2c takes the last dimension of real input and converts to complex
r2c=lambda x:tf.complex(x[...,0],x[...,1])

def createLayer(x, szW, trainning,lastLayer):
    """
    This function create a layer of CNN consisting of convolution, batch-norm,
    and ReLU. Last layer does not have ReLU to avoid truncating the negative
    part of the learned noise and alias patterns.

    DOCS related to tf usage:
        -get_variable creates a new variable with these parameters. 
        - Weights as W, initialized with xavier: random uniform bounded between \pm sqrt(6)/sqrt(n_{i}+n_{i+1}), n_i are incoming connections, n_{i+1} are outgoing connections. These solves vanishing gradients problems
        - tf.nn.conv2d creates 2D convolution and filters:      
            -input tensor has rank 4 or higher and shapes [:-3]    (the first ones) are batch dimension (in our case, [batch_shape, (in_height, in_width, in_channel) where currently we operate on one channel]). 
            - Filters have shape [filter_height, filter_width, in_channels, out_channels]
            - Batch normalization: applies transformation that maintains mean output close to 0 and standard deviation close to 1. During training, layer normalises output using mean and std of the current batch. During inference, layer normalises its output using a moving average of the mean and std of the batches seen during training. These variables update in training mode. this is enabled with trainning boolean parameters
            - ReLu: rectified linear (max(features, 0))
    """

    W=tf.compat.v1.get_variable('W',shape=szW,initializer=tf.initializers.GlorotUniform())
    x = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    xbn=tf.keras.layers.BatchNormalization(trainable=trainning,fused=True,name='BN')(x)

    if not(lastLayer):
        return tf.nn.relu(xbn)
    else:
        return xbn

def dw(inp,trainning,nLay, inChan = 2, outChan = 2):
    """
    This is the Dw block as defined in the Fig. 1 of the MoDL paper
    It creates an n-layer (nLay) residual learning CNN.
    Convolution filters are of size 3x3 and 64 such filters are there.
    nw: It is the learned noise
    dw: it is the output of residual learning after adding the input back.

    DOCS related to tf usage:
        - nw is a dictionary of layers, where c0 stores the input to the network
        - szW is a len=N-1 dict of sizes for convolutional kernels (3,3,64,64).
        Then input layer changes to (3,3,2,64) and output to (3,3,2,64)
        Here input has two channels (for Fourier space), in our case that would be 
        input layer (3,3,1,64) and output layer (3,3,64,1), due that input has only
        one channel for OPT
        - tf.compat.v1.variable_scope allows to easily share variables across different parts of the program, even within different name scopes
        - Creates chained layers with variable_scope 'layerx', passing previous layer
        - 'Residual': sums directly input (shorcut is an identity input copy, works great in same device) and last layer (w/o Relu)
    """
    lastLayer=False
    nw={}
    nw['c'+str(0)]=inp
    szW={}
    szW = {key: (3,3,64,64) for key in range(2,nLay)}
    szW[1]=(3,3,inChan,64)
    szW[nLay]=(3,3,64,outChan)

    for i in np.arange(1,nLay+1):
        if i==nLay:
            lastLayer=True
        with tf.compat.v1.variable_scope('Layer'+str(i)):
            nw['c'+str(i)]=createLayer(nw['c'+str(i-1)],szW[i],trainning,lastLayer)

    with tf.name_scope('Residual'):
        shortcut=tf.identity(inp)
        dw=shortcut+nw['c'+str(nLay)]
    return dw


class Aclass:
    """
    This class is created to do the data-consistency (DC) step as described in paper.

    Docs usage:
        -We have our own data consistency step for our data, this might be the critical step to change.
        - A class depicts the model, here we should discard csm (coil sensitivity maps), specifical to MRI reconstruction
        - 
    """
    def __init__(self, hR, hRT, lam):
        with tf.name_scope('Ainit'):
            
            self.hR = hR
            self.hRT = hRT
            
            #self.cgIter=cgIter
            #self.tol=tol

    def myAtA(self,img, mode='OPT'):
        # For OPT tentative

        if mode == 'OPT':
            with tf.name_scope('AtA'):
                img = img       # masking actually subsamples images
                sino = hR(img)
                iradon = hRT(sino)
                
                return iradon

        if mode == 'MultiMRI':
            # For MRI
            with tf.name_scope('AtA'):
                coilImages=self.csm*img
                kspace=  tf.signal.fft2d(coilImages)/self.SF
                temp=kspace*self.mask
                coilImgs =tf.signal.ifft2d(temp)*self.SF
                coilComb= tf.reduce_sum(coilImgs*tf.math.conj(self.csm),axis=0)
                coilComb=coilComb+self.lam*img
            
            return coilComb

def myCG(A,rhs):
    """
    This is my implementation of CG algorithm in tensorflow that works on
    complex data and runs on GPU. It takes the class object as input.

    - For OPT, should modify CG, but still use this operator for CG - revise notes on previously 
    implemented CG.
    """
    rhs=r2c(rhs)        # This wouldn't apply for real-valued data
    cond=lambda i,rTr,*_: tf.logical_and( tf.less(i,10), rTr>1e-10) # Convergence condition
    
    # Conjugate gradients body, iterated in Tensorflow with while loop
    def body(i,rTr,x,r,p):
        with tf.name_scope('cgBody'):
            Ap=A.myAtA(p)
            alpha = rTr / tf.cast(tf.reduce_sum(tf.math.conj(p)*Ap), float)
            alpha=tf.complex(alpha,0.)
            x = x + alpha * p
            r = r - alpha * Ap
            rTrNew = tf.cast(tf.reduce_sum(tf.math.conj(r)*r), float)
            beta = rTrNew / rTr
            beta=tf.complex(beta,0.)
            p = r + beta * p
        return i+1,rTrNew,x,r,p

    x=tf.zeros_like(rhs)
    i,r,p=0,rhs,rhs
    rTr = tf.cast( tf.reduce_sum(tf.math.conj(r)*r), float)
    loopVar=i,rTr,x,r,p
    out=tf.while_loop(cond,body,loopVar,name='CGwhile',parallel_iterations=1)[2]
    return c2r(out)

def getLambda():
    """
    create a shared variable called lambda.
    
    - Shared variable lambda is the regularisation parameter, used all over the network
        - tf.compat.v1.get_variable_scope : in this case, creates a new variable or reuses if it was previously created in this scope. This is useful because lambda regularisation parameter is being shared across
    """
    with tf.compat.v1.variable_scope(tf.compat.v1.get_variable_scope(), reuse=tf.compat.v1.AUTO_REUSE):
        lam = tf.compat.v1.get_variable(name='lam1', dtype=tf.float32, initializer=.05)
    return lam

def callCG(rhs):
    """
    this function will call the function myCG on each image in a batch

    Comments: 
        - get default graph gets the current thread default graph
    """
    G=tf.get_default_graph()
    getnext=G.get_operation_by_name('getNext')
    _,_,csm,mask=getnext.outputs
    l=getLambda()       # Lambda parameter
    l2=tf.complex(l,0.) # Relevant to MRI
    
    def fn(tmp):
        c,m,r=tmp
        Aobj=Aclass(c,m,l2)
        y=myCG(Aobj,r)  # Solves CG with the CG algorithm
        return y
    
    inp=(csm,mask,rhs)      # MRI specifics
    rec=tf.map_fn(fn,inp,dtype=tf.float32,name='mapFn2')    # applies fn to inputs unstacked on axis 0 (for batch operation)
    
    return rec      

@tf.custom_gradient
def dcManualGradient(x):
    """
    This function impose data consistency constraint. Rather than relying on
    TensorFlow to calculate the gradient for the conjuagte gradient part.
    We can calculate the gradient manually as well by using this function.
    Please see section III (c) in the paper.

     - Data consistency custom gradients:
        This decorator defines function with a custom gradient, which in this case is given by the callCG to calculate the gradient.

    """
    y=callCG(x)
    def grad(inp):
        out=callCG(inp)
        return out
    return y,grad


def dc(rhs,csm,mask,lam1):
    """
    This function is called to create testing model. It apply CG on each image
    in the batch.
    """
    lam2=tf.complex(lam1,0.)
    def fn( tmp ):
        c,m,r=tmp
        Aobj=Aclass( c,m,lam2 )
        y=myCG(Aobj,r)
        return y
        
    inp=(csm,mask,rhs)
    rec=tf.map_fn(fn,inp,dtype=tf.float32,name='mapFn')
    return rec

def makeModel(atb,csm,mask,training,nLayers,K,gradientMethod):
    """
    This is the main function that creates the model.

    Comments:

        - The whole scope is called myModel, where this is implemented. 
        - Iterations i conform the nLayers model and attaches a dw network.
        As x_0 = atb, the model would be 
        {atb, dw1, dc1, ...  }
    """
    out={}
    out['dc0']=atb #x_0 initialization
    with tf.name_scope('myModel'):
        with tf.compat.v1.variable_scope('Wts',reuse=tf.compat.v1.AUTO_REUSE):
            for i in range(1,K+1):
                j=str(i)
                out['dw'+j]=dw(out['dc'+str(i-1)],training,nLayers)
                lam1=getLambda()
                rhs=atb + lam1*out['dw'+j]  
                if gradientMethod=='AG':
                    out['dc'+j]=dc(rhs,csm,mask,lam1)   #Forward pass? 
                elif gradientMethod=='MG':
                    if training:
                        out['dc'+j]=dcManualGradient(rhs) #Calculate gradients manually for faster implementation
                    else:
                        out['dc'+j]=dc(rhs,csm,mask,lam1)
    return out
