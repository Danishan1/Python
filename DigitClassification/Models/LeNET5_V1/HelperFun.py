import time
from torch import cuda
import torch


def conSec2Time(seconds):
    # Calculate hours, minutes, and remaining seconds
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    remaining_seconds = seconds % 60

    # Create the time string
    time_str = ""
    if hours > 0:
        time_str += f"{int(hours)}hr "
    if minutes > 0:
        time_str += f"{int(minutes)}min "
    if remaining_seconds > 0 or (hours == 0 and minutes == 0):
        time_str += f"{remaining_seconds:0.5f}sec"

    return time_str


def printTime(start, str, device=None):
    """Prints difference between start and end time.

    Args:
        start (float): Start time of computation (preferred in timeit format).
        str : msg that you want to print
        device ([type], optional): Device that compute is running on. Defaults to None.

    Returns:
        float: time between start and end in seconds (higher is longer).
    """
    total_time = time.time() - start
    timeStr = conSec2Time(total_time)
    if device is None:
        print(f"{str} | Time taken : {timeStr}")
    else:
        print(f"{str} on ({device}) | Time taken : {timeStr}")
    return total_time


# Device Egnostic Code
device = lambda: "cuda" if cuda.is_available() else "cpu"


# Get Device Name
deviceName = lambda dev: print(
    f"You are working on {dev} and deevice Name is {cuda.get_device_name(dev)}"
)

def getNoLearnParamters(model: torch.nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
