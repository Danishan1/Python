import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import HelperFun
import  SaveData

startTime = HelperFun.time.time()
device = HelperFun.device()


PROGRESS_PATH = "DigitClassification/Models/LeNET5_V1/CnnModelProgress.txt"
line = f"\n###### CNN Model Training | Device: {torch.cuda.get_device_name(device)} ######\n"
print(line)
SaveData.addLine2File(PROGRESS_PATH, line)


# Load MNIST data
transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0,), (128,)),
    ]
)
train = torchvision.datasets.MNIST(
    "data", train=True, download=True, transform=transform
)
test = torchvision.datasets.MNIST(
    "data", train=True, download=True, transform=transform
)
trainloader = torch.utils.data.DataLoader(train, shuffle=True, batch_size=100)
testloader = torch.utils.data.DataLoader(test, shuffle=True, batch_size=100)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 10, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.dropout = nn.Dropout(0.2)
        self.flat = nn.Flatten()
        self.fc = nn.Linear(27 * 27 * 10, 128)
        self.relu2 = nn.ReLU()
        self.output = nn.Linear(128, 10)

    def forward(self, x):
        x = self.relu1(self.conv(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = self.relu2(self.fc(self.flat(x)))
        x = self.output(x)
        return x


model = CNN().to(device)

optimizer = optim.Adam(model.parameters())
loss_fn = nn.CrossEntropyLoss().to(device)

n_epochs = 200
loss = 0
acc = 0

for epoch in range(n_epochs):
    model.train()
    for X_batch, y_batch in trainloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Validation
    model.eval()
    with torch.inference_mode() :
        acc = 0
        count = 0
        for X_batch, y_batch in testloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            y_pred = model(X_batch)
            acc += (torch.argmax(y_pred, 1) == y_batch).float().sum()
            count += len(y_batch)
        acc = acc / count
        
        str = f"Epoch-{epoch+1}: Model Loss {loss*100:0.3f}% Model Accuracy {acc*100:0.2f}%"
        HelperFun.printTime(startTime, str)
        if  epoch%10 ==0 :
            SaveData.addData2File(PROGRESS_PATH, epoch, loss, acc, startTime)
        


PATH = "DigitClassification/Models/LeNET5_V1/CnnModel.pt"
SaveData.saveModelParams(model, optimizer, PATH, n_epochs, loss, acc, device, startTime)


learnableParms = HelperFun.getNoLearnParamters(model)
SaveData.CompletionReport(
    PROGRESS_PATH,
    "CNN Model",
    n_epochs,
    loss,
    acc,
    startTime,
    device,
    learnableParms,
)
HelperFun.printTime(startTime, "Model Complexted Successfully. ")
