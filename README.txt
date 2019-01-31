Step 1 is to convert to an infrence model like this:
retinanet-convert-model resnet50_csv_02.h5 resnet02_inf.h5



# OPTIONAL -- NOT BEING USED RIGHT NOW 
# We don't need the pb/protobuf file format if we stick with keras the whole way through.  Seems to be working
# with the h5 model.
Then we convert from inference to pb apparently.. found this one:
python keras_to_tensorflow.py --input_model=./snapshots-old/resnet02_inf.h5 --output_model=output.pb --quantize=True
... didn't work.  PUll new code from git to make sure we're running the right stuff.
Ran that, exit_ok error in mkdir command... that only happens in python2 though, switch to python3
Now we have invalid keywoard freeze argument which has to do with loading the resnet model I think...
... Follow instructions here: https://github.com/amir-abdi/keras_to_tensorflow/issues/57
.... basically change code in keras_to_tensorflow:

--------------------------------
I figured in out, this is how its done

in keras_to_tensorflow.py
Add
from keras_retinanet import models

Then comment line 62
# model = keras.models.load_model(input_model_path)

and add the following like this
model = models.load_model(input_model_path, backbone_name="resnet50")
------------------------------
THEN it all works.  Excellent!



# BACK TO USEFUL STUFF 
# running keras_proto.py:

Make sure you've got the keras-retinanet stuff installed for the active user
pip3 install . -- user inside the keras-retinet git repo..

The testing_data/images/*.jpg files hould be in the diretory that keras_proto.py is run from.
