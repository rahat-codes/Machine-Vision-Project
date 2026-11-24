GPU ACTIVATION GUIDE
Machine Vision Final Project

Hardware: NVIDIA GeForce RTX 5050 (8 GB VRAM)
Operating System: Windows 10/11
IDE: Visual Studio Code
Python Environment: Conda
Python Version: 3.11
GPU Framework: PyTorch with CUDA 13.0 build

1. Verify NVIDIA GPU Installation

Open the VS Code terminal (PowerShell) and execute:

powershell
nvidia-smi


Verify that your NVIDIA GPU is detected.

Expected hardware information:

GPU: NVIDIA GeForce RTX 5050
VRAM: Approximately 8 GB
NVIDIA Driver: 591.91
Driver-supported CUDA version: 13.1

Note: The CUDA version displayed by nvidia-smi indicates the maximum CUDA version supported by the installed driver. It does not mean that the CUDA Toolkit is installed.

2. Create a Dedicated Conda Environment

Navigate to the project directory:

powershell
cd "C:\Users\fhv\Documents\rahat-codes\Machine Vision Final Project"


Create a new Conda environment:

powershell
conda create -n vision_gpu python=3.11 -y


Activate the environment:

powershell
conda activate vision_gpu


Verify the Python version:

powershell
python --version


Expected output:

text
Python 3.11.x

3. Upgrade pip

Upgrade the Python package installer:

powershell
python -m pip install --upgrade pip

4. Install Jupyter Notebook and Kernel

Install Jupyter and IPython Kernel:

powershell
pip install jupyter ipykernel


Register the Conda environment as a Jupyter kernel:

powershell
python -m ipykernel install --user --name vision_gpu --display-name "Python (Vision GPU)"


This allows VS Code to recognize and use the dedicated GPU environment.

5. Install PyTorch with CUDA Support

Install GPU-enabled PyTorch packages:

powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130


These packages provide:

PyTorch: Deep learning framework
Torchvision: Computer vision datasets, models, and transformations
Torchaudio: Audio processing support

The CUDA-enabled PyTorch build includes the necessary CUDA runtime components. A separate CUDA Toolkit installation is generally unnecessary for standard PyTorch GPU usage.

6. Install Machine Vision Libraries

Install the required machine vision and data science packages:

powershell
pip install numpy pandas matplotlib seaborn


Install OpenCV and image processing libraries:

powershell
pip install opencv-python pillow scikit-learn


Install Ultralytics for YOLO-based object detection:

powershell
pip install ultralytics

7. Configure VS Code

Install the following VS Code extensions:

Python (Microsoft)
Jupyter (Microsoft)

Open the project folder in VS Code.

Open your .ipynb notebook.

In the upper-right corner:

Click Select Kernel.
Select Python Environments if prompted.
Choose Python (Vision GPU).
Restart the notebook kernel.

Your Jupyter notebook should now use the dedicated Conda environment.

8. Verify GPU Availability

Create a new notebook cell and execute:

python
import torch

print("PyTorch Version:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())

if torch.cuda.is_available():

    print("GPU Name:", torch.cuda.get_device_name(0))
    print("CUDA Runtime:", torch.version.cuda)
    print("GPU Count:", torch.cuda.device_count())

    device = torch.device("cuda")

    x = torch.rand(5000, 5000, device=device)
    y = torch.rand(5000, 5000, device=device)

    result = torch.matmul(x, y)

    print("Calculation Device:", result.device)
    print("GPU ACTIVATION SUCCESSFUL!")

else:
    print("CUDA is unavailable. Check your installation.")


Expected output:

text
CUDA Available: True
GPU Name: NVIDIA GeForce RTX 5050
CUDA Runtime: 13.0
GPU Count: 1
Calculation Device: cuda:0
GPU ACTIVATION SUCCESSFUL!

9. Use GPU in Machine Vision Models

When using PyTorch, explicitly move the model and input tensors to the GPU.

Example:

python
import torch
import torch.nn as nn

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = YourModel().to(device)

images = images.to(device)

outputs = model(images)

print("Model running on:", device)


For YOLO with Ultralytics:

python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

results = model.predict(
    source="image.jpg",
    device=0
)


Note: OpenCV's standard CPU operations do not automatically use the NVIDIA GPU. GPU acceleration in OpenCV requires a CUDA-enabled OpenCV build or another GPU-capable implementation.

10. Monitor GPU Usage

Open another PowerShell terminal and execute:

powershell
nvidia-smi


To monitor GPU usage continuously:

powershell
nvidia-smi -l 1


This updates GPU information every second.

Monitor:

GPU utilization
GPU memory usage
GPU temperature
Running processes

Press Ctrl + C to stop monitoring.

11. Activate the Environment in Future Sessions

Whenever you reopen VS Code or create a new terminal, activate the environment:

powershell
conda activate vision_gpu


Then open your notebook and select:

Python (Vision GPU)

You do not need to reinstall the packages every time.

12. Troubleshooting
Problem 1: CUDA Available = False

Check the following:

powershell
nvidia-smi


Then verify PyTorch:

powershell
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"


Ensure that you installed the CUDA-enabled PyTorch build in the correct Conda environment.

Problem 2: VS Code Cannot Find the Kernel

Register it again:

powershell
conda activate vision_gpu

python -m ipykernel install --user --name vision_gpu --display-name "Python (Vision GPU)"


Restart VS Code if necessary.

Problem 3: Wrong Python Environment

Check the active Python executable:

powershell
where.exe python


It should point to the vision_gpu environment.

Problem 4: CUDA Out of Memory

Your RTX 5050 has approximately 8 GB of VRAM.

Consider:

Reducing batch size.
Reducing image resolution.
Using smaller deep learning models.
Clearing unused GPU tensors.

Example:

python
import torch

torch.cuda.empty_cache()

13. Environment Summary
Component	Configuration
GPU	NVIDIA RTX 5050
VRAM	8 GB
NVIDIA Driver	591.91
Python	3.11
Environment	vision_gpu
Deep Learning	PyTorch
CUDA Build	cu130
Notebook	Jupyter
IDE	VS Code
Object Detection	Ultralytics YOLO
Image Processing	OpenCV
Final Checklist
NVIDIA GPU detected using nvidia-smi
Conda environment created
Conda environment activated
Jupyter and IPython Kernel installed
PyTorch CUDA-enabled version installed
Machine vision libraries installed
VS Code Python extension installed
VS Code Jupyter extension installed
Correct Jupyter kernel selected
CUDA availability verified
GPU computation tested successfully

Project Status: Ready for GPU-accelerated Machine Vision development.

Important: Always use the vision_gpu environment when running your Machine Vision Final Project notebooks.