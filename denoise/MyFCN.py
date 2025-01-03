import chainer
from chainer import Variable
import chainer.links as L
import chainer.functions as F
import numpy as np
import math
import cv2
from chainer.links.caffe import CaffeFunction
import chainerrl
from chainerrl.agents import a3c

class DilatedConvBlock(chainer.Chain):

    def __init__(self, d_factor, weight, bias):
        super(DilatedConvBlock, self).__init__(
            diconv=L.DilatedConvolution2D( in_channels=64, out_channels=64, ksize=3, stride=1, pad=d_factor, dilate=d_factor, nobias=False, initialW=weight, initial_bias=bias),
            #bn=L.BatchNormalization(64)
        )

        self.train = True

    def __call__(self, x):
        h = F.relu(self.diconv(x))
        #h = F.relu(self.bn(self.diconv(x)))
        return h


class MyFcn(chainer.Chain, a3c.A3CModel):
 
    def __init__(self, n_actions):
        w = chainer.initializers.HeNormal()
        net = CaffeFunction('/content/pixelRL/initial_weight/zhang_cvpr17_denoise_15_gray.caffemodel')
        super(MyFcn, self).__init__(
            conv1=L.Convolution2D( 1, 64, 3, stride=1, pad=1, nobias=False, initialW=net.layer1.W.data, initial_bias=net.layer1.b.data),
            diconv2=DilatedConvBlock(2, net.layer3.W.data, net.layer3.b.data),
            diconv3=DilatedConvBlock(3, net.layer6.W.data, net.layer6.b.data),
            diconv4=DilatedConvBlock(4, net.layer9.W.data, net.layer9.b.data),
            diconv5_pi=DilatedConvBlock(3, net.layer12.W.data, net.layer12.b.data),
            diconv6_pi=DilatedConvBlock(2, net.layer15.W.data, net.layer15.b.data),
            conv7_pi=chainerrl.policies.SoftmaxPolicy(L.Convolution2D( 64, n_actions, 3, stride=1, pad=1, nobias=False, initialW=w)),
            diconv5_V=DilatedConvBlock(3, net.layer12.W.data, net.layer12.b.data),
            diconv6_V=DilatedConvBlock(2, net.layer15.W.data, net.layer15.b.data),
            conv7_V=L.Convolution2D( 64, 1, 3, stride=1, pad=1, nobias=False, initialW=net.layer18.W.data, initial_bias=net.layer18.b.data),
        )
        self.train = True
 
    def pi_and_v(self, x):
         
        h = F.relu(self.conv1(x))
        h = self.diconv2(h)
        h = self.diconv3(h)
        h = self.diconv4(h)
        h_pi = self.diconv5_pi(h)
        h_pi = self.diconv6_pi(h_pi)
        pout = self.conv7_pi(h_pi)
        h_V = self.diconv5_V(h)
        h_V = self.diconv6_V(h_V)
        vout = self.conv7_V(h_V)
       
        return pout, vout
