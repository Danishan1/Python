import torch
import time
import HelperFun


def saveModelParams(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    PATH: str,
    n_epochs: int,
    loss: float,
    acc: float,
    device: str,
    startTime: time.time,
):
    torch.save(
        {
            "epoch": n_epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
            "accuracy": acc,
            "device": device,
            "timeTook": time.time() - startTime,
        },
        PATH,
    )


def saveModel(model: torch.nn.Module, PATH):
    model_scripted = torch.jit.script(model)  # Export to TorchScript
    model_scripted.save(PATH)  # Save


def addData2File(PATH: str, epoch, loss, acc, startTime):
    with open(PATH, "a") as file:
        timeStr = HelperFun.conSec2Time(HelperFun.time.time() - startTime)
        line = f"Epoch-{epoch+1} | Model Loss: {loss*100:0.3f}% | Model Accuracy: {acc*100:0.2f}% | Cmpletion Time: {timeStr}"
        file.write(line + "\n")


def addLine2File(PATH: str, line):
    with open(PATH, "a") as file:
        file.write(line + "\n")


def CompletionReport(
    PATH: str, modelName: str, epoch, loss, acc, startTime, device: str, learnParams
):
    with open(PATH, "a") as file:
        timeStr = HelperFun.conSec2Time(HelperFun.time.time() - startTime)
        line = f"\n*********** Model: {modelName} Completion Summary ***********\n"
        line += f"Parameter List :\n - Model Name : {modelName} \n - Epochs: {epoch} \n - Model Loss: {loss*100:0.3f}% \n"
        line += f" - Model Accuracy: {acc*100:0.2f}% \n - Completion Time: {timeStr} \n - Device {torch.cuda.get_device_name(device)} \n"
        line += f" - Learnable Paramters : {learnParams}\n"
        line += f"\n ****** End *******\n"
        file.write(line + "\n")
