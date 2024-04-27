from torch import nn
from torch.optim import SGD
import torch

import leNET
from helperFunction import getAccuracy

device = "cuda" if torch.cuda.is_available() else "cpu"


class ModelTraining:

    def __init__(self, model, trainDataLoader, testDataLoader):
        
        print(torch.cuda.is_available())

        self.startTime = leNET.time()
        self.trainDataLoader = trainDataLoader
        self.testDataLoader = testDataLoader
        # Model Training

        # Setting Up Hyper parameters
        self.LEARNING_RATE = 0.01
        self.EPOCHS = 10

        # Creating Instances for the Module
        self.model = model
        self.lossFun = nn.CrossEntropyLoss()
        self.optimiser = SGD(params=self.model.parameters(), lr=self.LEARNING_RATE)

        # Putting On GPU if Availabe
        self.model.to(device)
        self.lossFun.to(device)

        for epoch in range(self.EPOCHS):

            trainLoss = 0
            testLoss = 0
            accuracyScore = 0

            # Putting Model on Train Mode
            self.model.train()

            for trainData, trainLabel in self.trainDataLoader:
                trainLoss += self.trainingStep(trainData, trainLabel)

            # Putting Model on Train Mode
            self.model.eval()
            with torch.inference_mode():
                for testData, testLabel in self.testDataLoader:
                    teLoss, aScore = self.testingStep(testData, testLabel)
                    testLoss += teLoss
                    accuracyScore += aScore

            trainLoss /= len(trainDataLoader)
            testBatch = len(testDataLoader)
            # testLoss = testLoss.item()
            testLoss /= testBatch
            accuracyScore /= testBatch
            print(
                f"On Epoch: {epoch} Train Loss: {trainLoss:0.3f} | Test Loss: {testLoss:0.3f} Accuracy Score: {accuracyScore:0.3f}"
            )

            if epoch % 5 == 0:
                file = "DigitClassification/Models/LeNET5/progress.txt"
                with open(file, "a") as file:
                    line = f"On Epoch: {epoch} Train Loss: {trainLoss:0.3f} | Test Loss: {testLoss:0.3f} Accuracy Score: {accuracyScore:0.3f}"
                    file.write(line + "\n")
                    leNET.printTime(
                        self.startTime,
                        f"Successfully Run {epoch} Epoch outof {self.EPOCHS}",
                    )

        file = "DigitClassification/Models/LeNET5/lenNET.pth"
        torch.save(obj=self.model.state_dict(), f=file)
        file2 = "DigitClassification/Models/LeNET5/lenNETOpt.pth"
        torch.save(obj=self.optimiser.state_dict(), f=file2)

    def trainingStep(self, X, y):

        X = X.to(device)
        y = y.to(device)
        X = X.type(torch.float32)

        yPre = self.model(X)
        
        loss = self.lossFun(yPre, y)

        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

        return loss

    #
    def testingStep(self, X, y) -> tuple[float, float]:

        X = X.to(device)
        y = y.to(device)
        X = X.type(torch.float32)

        yPre = self.model(X)
        testLoss = self.lossFun(yPre, y)
        accuracyScore = getAccuracy(y, yPre.argmax(dim=1))

        return testLoss.item(), accuracyScore
