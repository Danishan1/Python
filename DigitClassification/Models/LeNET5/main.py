# 1. Load the Dataset
# 2. Check its Datatype and make ready that fro pytorch
# 3. Visualise your data
# 4. Start making Your Model
# 5. Select a Loss Function & optimiser
# 6. Write Code for your training the Model
# 7. Write Code fro testing your Model
# 8. Write Code for evaluating your Model
#
#
#
#
#
#
#
#

from time import time
import customDataset
import torch.nn.functional as F
import torch
from torch import nn

import printImages as prIm
from helperFunction import printTime
import leNET
from modelTraining import ModelTraining


startTime = time()  # Start Time for the Project


# loading Data from binary files to tensors
loader = customDataset.CustomDataset()
# Load the data in batches of 32, these are the iterators of bateches
trainDataloader, testDataloader = loader.load()
# First batch we cam move all batches using next function
trainBatch, trainLabelBatch = next(iter(trainDataloader))


# General LeNET-5 accept (32 x 32) image but we have (28 x 28) size image, therefore we are reshaping
# interpolate function require dimention of (N, C, W, H)
trainBatch = F.interpolate(
    trainBatch, size=(32, 32), mode="bilinear", align_corners=False
)
trainBatch = trainBatch.squeeze()

# Model Training
model = leNET.leNET5(10)
# print(type(model))

ModelTraining(model, trainDataloader, testDataloader)





printTime(startTime, "Model Completed Successfully")
