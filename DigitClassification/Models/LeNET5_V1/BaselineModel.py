import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import HelperFun
import SaveData

startTime = HelperFun.time.time()
device = HelperFun.device()


PROGRESS_PATH = "DigitClassification/Models/LeNET5_V1/BaseModelprogress.txt"
line = f"\n###### Baseline Model Training | Device: {torch.cuda.get_device_name(device)} ######\n"
SaveData.addLine2File(PROGRESS_PATH, line)


# Load MNIST data
train = torchvision.datasets.MNIST("data", train=True, download=True)
test = torchvision.datasets.MNIST("data", train=True, download=True)

HelperFun.printTime(startTime, "Data Load Successfully")


# each sample becomes a vector of values 0-1
X_train = train.data.reshape(-1, 784).float() / 255.0
y_train = train.targets
X_test = test.data.reshape(-1, 784).float() / 255.0
y_test = test.targets


class Baseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(784, 784)
        self.act1 = nn.ReLU()
        self.layer2 = nn.Linear(784, 10)

    def forward(self, x):
        x = self.act1(self.layer1(x))
        x = self.layer2(x)
        return x


model = Baseline()

optimizer = optim.SGD(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()
loader = torch.utils.data.DataLoader(
    list(zip(X_train, y_train)), shuffle=True, batch_size=100
)

model = model.to(device)
loss_fn = loss_fn.to(device)

n_epochs = 200
acc = 0
loss = 0

for epoch in range(n_epochs):
    model.train()
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    with torch.inference_mode():
        X_test = X_test.to(device)
        y_pred = model(X_test)
        acc = (torch.argmax(y_pred.cpu(), 1) == y_test).float().mean()
        str = f"Epoch-{epoch+1}: Model Loss {loss*100:0.3f}% Model Accuracy {acc*100:0.2f}%"
        HelperFun.printTime(startTime, str)

        if epoch % 10 == 0:
            SaveData.addData2File(PROGRESS_PATH, epoch, loss, acc, startTime)


PATH = "DigitClassification/Models/LeNET5_V1/BaselineModel.pt"
SaveData.saveModelParams(model, optimizer, PATH, n_epochs, loss, acc, device, startTime)


learnableParms = HelperFun.getNoLearnParamters(model)
SaveData.CompletionReport(
    PROGRESS_PATH,
    "Baseline Model",
    n_epochs,
    loss,
    acc,
    startTime,
    device,
    learnableParms,
)
HelperFun.printTime(startTime, "Model Complexted Successfully. ")
