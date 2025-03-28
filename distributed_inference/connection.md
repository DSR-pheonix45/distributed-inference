Setting Up Distributed Inference Across Two Laptops
Here's a step-by-step guide to set up your distributed inference system with one laptop as the master node and another as the worker node.

Step 1: Push the Project to GitHub
On your first laptop (future master node):

# Initialize git repository
cd ~/peer-connection-project
git init

# Create a .gitignore file
echo "*.pyc" > .gitignore
echo "__pycache__/" >> .gitignore
echo "*.pb2.py" >> .gitignore
echo "*.pb2_grpc.py" >> .gitignore
echo "metrics.json" >> .gitignore

# Add all files
git add .

# Commit changes
git commit -m "Initial commit of distributed inference system"

# Create a new repository on GitHub through the web interface
# Then link your local repository to the remote one
git remote add origin https://github.com/yourusername/distributed-inference.git
git branch -M main
git push -u origin main

Step 2: Clone the Repository on the Second Laptop (Worker Node)
On your second laptop:

# Clone the repository
git clone https://github.com/yourusername/distributed-inference.git
cd distributed-inference

Step 3: Set Up Both Laptops
On Both Laptops:
Install CUDA and set environment variables:
# Install CUDA keyring
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-toolkit-12-0

# Set environment variables
echo 'export CUDA_HOME=/usr/local/cuda' >> ~/.bashrc
echo 'export PATH=$PATH:$CUDA_HOME/bin' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

Install required Python packages:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install deepspeed accelerate grpcio grpcio-tools protobuf nvidia-ml-py3 transformers

Compile the protocol buffer:
cd distributed_inference
python3 -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. inference.proto

Step 4: Configure Networking
Ensure both laptops are on the same network.

Find the IP address of the worker laptop:

# On the worker laptop
ip addr show
# Look for the IP address on your main network interface (usually wlan0 or eth0)

Make sure port 50051 is accessible (you might need to configure your firewall):
# On the worker laptop
sudo ufw allow 50051/tcp

Step 5: Start the Worker Node
On the worker laptop:

cd distributed-inference/distributed_inference
python3 worker_node.py

You should see: "Worker node started on port 50051"

Step 6: Register the Worker Node on the Master
On the master laptop:

cd ~/peer-connection-project/distributed_inference
python3 master_node.py --register --ip <worker_laptop_ip> --username <worker_laptop_username>

Replace <worker_laptop_ip> with the actual IP address of your worker laptop.

Step 7: Run Inference
On the master laptop:

python3 master_node.py --generate --prompt "Explain quantum computing in simple terms"

Troubleshooting
Connection Issues
If the master can't connect to the worker:

Check if both laptops are on the same network
Verify the worker's IP address is correct
Ensure port 50051 is open on the worker
Try pinging the worker from the master:
ping <worker_laptop_ip>

Test the gRPC connection with a tool like grpcurl
Model Loading Issues
If you encounter issues loading the model:

Check if you have enough disk space for the model
Ensure you have internet access to download the model
Try downloading the model manually first:
from transformers import AutoModelForCausalLM, AutoTokenizer
model_name = "google/gemma-2b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

CUDA Issues
If CUDA isn't working properly:

Verify CUDA installation:
nvcc --version

Check if PyTorch can see your GPU:
import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))

Monitoring the System
To monitor the system in real-time:

On the master laptop, check the metrics.json file for GPU utilization and memory usage data
You can set up a simple monitoring script:
import json
import time
import os

while True:
    if os.path.exists("metrics.json"):
        with open("metrics.json", "r") as f:
            try:
                metrics = json.load(f)
                for metric in metrics:
                    if metric["measurement"] == "gpu_utilization":
                        worker = metric["tags"]["worker"]
                        value = metric["fields"]["value"]
                        print(f"{worker} GPU utilization: {value}%")
            except json.JSONDecodeError:
                print("Error reading metrics file")
    time.sleep(5)

Save this as monitor.py and run it with python3 monitor.py on the master laptop.