from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import time
import datasetDownloader as sdl


trainDataPath = "DigitClassification/src/mnistDataset/train-images.idx3-ubyte"
trainlablePath = "DigitClassification/src/mnistDataset/train-labels.idx1-ubyte"
testDataPath = "DigitClassification/src/mnistDataset/t10k-images.idx3-ubyte"
testlablePath = "DigitClassification/src/mnistDataset/t10k-labels.idx1-ubyte"


class mergeDataset(Dataset):
    def __init__(self, xTrain, yTrain):
        self.xTrain = xTrain.clone()
        self.yTrain = yTrain.clone()

    def __len__(self):
        return len(self.xTrain)

    def __getitem__(self, idx):
        return self.xTrain[idx], self.yTrain[idx]


class CustomDataset:
    def load(self, isTime=True):
        startTime = time.time()

        loader = sdl.MnistDataloader(
            trainDataPath, trainlablePath, testDataPath, testlablePath
        )
        (xTrain, yTrain), (xTest, yTest) = loader.load_data()

        # Converting Our data into np.array (because converting Data from np.array is much faster than list)
        xTrain = np.array(xTrain)
        yTrain = np.array(yTrain)
        xTest = np.array(xTest)
        yTest = np.array(yTest)

        # Converting Our data into Tensors
        xTrain = torch.tensor(xTrain)
        yTrain = torch.tensor(yTrain)
        xTest = torch.tensor(xTest)
        yTest = torch.tensor(yTest)

        # Adding Channel to our image
        xTrain = xTrain.unsqueeze(dim=1)
        xTest = xTest.unsqueeze(dim=1)

        # Makeing Batches for Our Data

        batchSize = 32
        trainDataset = mergeDataset(xTrain, yTrain)
        testDataset = mergeDataset(xTest, yTrain)
        trainDataloader = DataLoader(trainDataset, batch_size=batchSize, shuffle=True)
        testDataloader = DataLoader(testDataset, batch_size=batchSize, shuffle=True)

        from helperFunction import printTime

        if isTime == True:
            str = f"Tensor Conversion, Merge Data & Label, and Converted into Batches of Size {batchSize}"
            printTime(startTime, str)

        return (trainDataloader, testDataloader)
