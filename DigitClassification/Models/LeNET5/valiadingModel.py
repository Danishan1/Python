import leNET
import torch
import customDataset
import modelTesting

model = leNET.leNET5(10)
model = model.load_state_dict(torch.load("DigitClassification/Models/LeNET5/leNET.py"))
device = "cuda" if torch.cuda.is_available() else "cpu"

model.to(device)

# loading Data from binary files to tensors
loader = customDataset.CustomDataset()
# Load the data in batches of 32, these are the iterators of bateches
trainDataloader, testDataloader = loader.load()

tester = modelTesting.ModelTesting(testDataloader)
