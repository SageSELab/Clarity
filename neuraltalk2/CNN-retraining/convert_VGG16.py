# converts a VGG16 model from the keras .h5 format to the caffemodel format via mmdnn (using mmconvert)


# Installation Instructions

# to get mmconvert, do `pip install mmdnn` in a virtual environment
# then do `conda install caffe`
# then do `conda install keras-gpu`

import os

if len(os.sys.argv) < 3:
    print("\nScript to convert a VGG16 model from the keras .h5 format to the caffemodel format via mmdnn (using mmconvert).")
    print("\nusage: python " + __file__ + " <.h5 keras checkpoint> <output caffemodel name>")
    print("\ni.e. python " + __file__ + " imagenet_vgg16.h5 output_model\n")
    exit()
    
keras_h5 = os.sys.argv[1]
out_model = os.sys.argv[2]

if keras_h5[len(keras_h5)-3:len(keras_h5)] != ".h5":
    print("Error: input keras checkpoint must be a .h5 file")
    exit()

print("Converting VGG16 from keras format to caffemodel format...")

os.system("mmconvert -sf keras -iw " + keras_h5 + " -df caffe -om " + out_model)
