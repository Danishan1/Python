from torch import nn
from torch.optim import SGD
import torch 

import leNET
from helperFunction import getAccuracy

device = "cuda" if torch.cuda.is_available() else "cpu"


class ModelTesting:

    def __init__(self, testDataLoader):

        self.startTime = leNET.time()
        self.testDataLoader = testDataLoader
        self.model = leNET.leNET5(10)
        self.optimiser = SGD(self.model.parameters(), lr=0.1)
        
        file = "DigitClassification/Models/LeNET5/lenNET.pth"
        file2 = "DigitClassification/Models/LeNET5/lenNETOpt.pth"

        self.model = self.model.load_state_dict(torch.load(f=file))
        self.optimiser = self.optimiser.load_state_dict(torch.load(f=file2))
       
        self.model.to(device)

        testLoss = 0
        accuracyScore = 0

        # Putting Model on Train Mode
        self.model.eval()
        with torch.inference_mode():
            for testData, testLabel in self.testDataLoader:
                teLoss, aScore = self.testingStep(testData, testLabel)
                testLoss += teLoss
                accuracyScore += aScore
                
        testBatch = len(testDataLoader)
        testLoss /= testBatch
        accuracyScore /= testBatch
        print(
            f"Test Loss: {testLoss:0.3f} Accuracy Score: {accuracyScore:0.3f}"
        )

            
    def testingStep(self, X, y) -> tuple[float, float]:

        X = X.to(device)
        y = y.to(device)
        X = X.type(torch.float32)

        yPre = self.model(X)
        testLoss = self.lossFun(yPre, y)
        accuracyScore = getAccuracy(y.cpu(), yPre.argmax(dim=1))

        return testLoss.item(), accuracyScore
