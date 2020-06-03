import numpy as np
import cv2 as cv
from openvino.inference_engine import IENetwork, IECore

import argparse

parser = argparse.ArgumentParser(description='Run video super resolution with OpenVINO')
parser.add_argument('-i', dest='input', help='Path to input video')
parser.add_argument('-m', dest='model', default='single-image-super-resolution-1033', help='Path to the model')
parser.add_argument('-o', dest='output', help='Path to output')

args = parser.parse_args()

# Setup network
net = IENetwork(args.model + '.xml', args.model + '.bin')

# Read a video stream from file
cap = cv.VideoCapture(args.input)
inp_w  = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
inp_h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv.CAP_PROP_FPS))

out_h, out_w = inp_h * 3, inp_w * 3  # Do not change! This is how model works

c1 = net.layers['79/Cast_11815_const']
c1.blobs['custom'][4] = inp_h
c1.blobs['custom'][5] = inp_w

c2 = net.layers['86/Cast_11811_const']
c2.blobs['custom'][2] = out_h
c2.blobs['custom'][3] = out_w

# Reshape network to specific size
net.reshape({'0': [1, 3, inp_h, inp_w], '1': [1, 3, out_h, out_w]})

ie = IECore()
exec_net = ie.load_network(net, 'CPU')

if(args.output is not None):
    out_stream = cv.VideoWriter(args.output,cv.VideoWriter_fourcc('M', 'P', '4', '2'), fps, (out_w, out_h))


while(cap.isOpened()):
    ret, img = cap.read()
    if img is None:
        break
    # Prepare input
    inp = img.transpose(2, 0, 1)  # interleaved to planar (HWC -> CHW)
    inp = inp.reshape(1, 3, inp_h, inp_w)
    inp = inp.astype(np.float32)

    # Prepare second input - bicubic resize of first input
    resized_img = cv.resize(img, (out_w, out_h), interpolation=cv.INTER_CUBIC)
    resized = resized_img.transpose(2, 0, 1)
    resized = resized.reshape(1, 3, out_h, out_w)
    resized = resized.astype(np.float32)

    outs = exec_net.infer({'0': inp, '1': resized})

    out = next(iter(outs.values()))

    out = out.reshape(3, out_h, out_w).transpose(1, 2, 0)
    out = np.clip(out * 255, 0, 255)
    out = np.ascontiguousarray(out).astype(np.uint8)
    if(args.output is not None):
        out_stream.write(out)

    cv.imshow('Source image', img)
    cv.imshow('Bicubic interpolation', resized_img)
    cv.imshow('Super resolution', out)
    
    if cv.waitKey(25) & 0xFF == ord('q'):
        break
        
cap.release()
if(args.output is not None):
    out_stream.release()


cv.destroyAllWindows()
