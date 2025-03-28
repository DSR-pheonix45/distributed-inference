A distributed inference system for running large language models across multiple GPU nodes. This system shards model layers across worker nodes and coordinates inference through a master node.

Table of Contents
Overview
System Requirements
Installation
Usage
Register Worker Nodes
Start Worker Nodes
Run Inference
Monitoring
Code Explanation
Worker Node
Master Node
Troubleshooting
Overview
This distributed inference system partitions a large language model (Gemma-2B) across multiple GPU nodes for efficient inference. The system uses:

DeepSpeed: For model sharding
gRPC: For communication between nodes
PyTorch: For model loading and inference
CUDA: For GPU acceleration
Metrics collection: For performance monitoring
System Requirements
CUDA-capable GPUs on all nodes
Ubuntu 20.04 or later
Python 3.8 or later
Network connectivity between all nodes
Installation
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
pip install deepspeed accelerate grpcio grpcio-tools protobuf nvidia-ml-py3

Clone this repository:
git clone https://github.com/yourusername/distributed-inference.git
cd distributed-inference

Compile the protocol buffer:
python3 -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. inference.proto

Usage
Register Worker Nodes
On the master node, register each worker node:

python3 master_node.py --register --ip <worker_ip_address> --username <worker_username>

Start Worker Nodes
On each worker node, start the worker service:

python3 worker_node.py

Run Inference
On the master node, run inference with your prompt:

python3 master_node.py --generate --prompt "Your input text here"

Monitoring
The system automatically collects GPU utilization and memory usage metrics from all worker nodes. These metrics are stored in metrics.json.

To visualize these metrics:

Install Grafana
Configure a JSON data source pointing to the metrics.json file
Create dashboards to visualize GPU utilization and memory usage
Code Explanation
Worker Node
The worker node (worker_node.py) implements a gRPC service that:

Receives model layers: The InitializeWorker method receives serialized model layers from the master node and loads them onto the GPU.
def InitializeWorker(self, request, context):
    try:
        # Deserialize the model layer
        buffer = io.BytesIO(request.model_layer)
        self.layer = torch.load(buffer, map_location=self.device)
        self.layer_id = request.layer_id
        
        # Set to evaluation mode
        self.layer.eval()
        
        return inference_pb2.InitResponse(
            success=True,
            message=f"Successfully initialized layer {self.layer_id}"
        )
    except Exception as e:
        return inference_pb2.InitResponse(
            success=False,
            message=f"Error initializing layer: {str(e)}"
        )

Processes input tensors: The ProcessLayer method receives input tensors, passes them through the assigned model layer, and returns the output.
def ProcessLayer(self, request, context):
    try:
        if self.layer is None:
            return inference_pb2.LayerResponse(
                success=False,
                error_message="Layer not initialized"
            )
        
        # Deserialize input tensor
        buffer = io.BytesIO(request.input_tensor)
        input_tensor = torch.load(buffer, map_location=self.device)
        
        # Process through the layer
        with torch.no_grad():
            output = self.layer(input_tensor)
        
        # Serialize output tensor
        output_buffer = io.BytesIO()
        torch.save(output, output_buffer)
        
        return inference_pb2.LayerResponse(
            output_tensor=output_buffer.getvalue(),
            success=True
        )
    except Exception as e:
        return inference_pb2.LayerResponse(
            success=False,
            error_message=f"Error processing layer: {str(e)}"
        )

Reports metrics: The Heartbeat method reports GPU utilization and memory usage metrics to the master node.
def Heartbeat(self, request, context):
    try:
        # Get GPU utilization
        utilization = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
        memory_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
        
        gpu_util = utilization.gpu
        memory_util = (memory_info.used / memory_info.total) * 100
        
        return inference_pb2.HeartbeatResponse(
            timestamp=int(time.time()),
            gpu_utilization=gpu_util,
            memory_utilization=memory_util
        )
    except Exception as e:
        return inference_pb2.HeartbeatResponse(
            timestamp=int(time.time()),
            gpu_utilization=0.0,
            memory_utilization=0.0
        )

Master Node
The master node (master_node.py) coordinates the distributed inference:

Worker registration: The register_worker function adds worker nodes to the system.
def register_worker(ip, username):
    workers_file = "workers.json"
    workers = []
    
    if os.path.exists(workers_file):
        with open(workers_file, 'r') as f:
            workers = json.load(f)
    
    workers.append({
        "ip": ip,
        "username": username
    })
    
    with open(workers_file, 'w') as f:
        json.dump(workers, f, indent=2)
    
    print(f"Registered worker at {ip}")

Model loading and sharding: The _load_and_shard_model method loads the model, partitions it, and distributes layers to worker nodes.
def _load_and_shard_model(self):
    print(f"Loading model {self.model_name}...")
    
    # Load tokenizer
    self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(self.model_name)
    
    # Extract transformer layers
    transformer_layers = model.model.layers
    num_layers = len(transformer_layers)
    num_workers = len(self.workers)
    
    # Distribute layers among workers
    layers_per_worker = num_layers // num_workers
    remainder = num_layers % num_workers
    
    start_idx = 0
    for worker_id in range(num_workers):
        # Calculate how many layers this worker gets
        worker_layers = layers_per_worker + (1 if worker_id < remainder else 0)
        end_idx = start_idx + worker_layers
        
        # Assign layer indices to this worker
        for layer_idx in range(start_idx, end_idx):
            self.layer_mapping[layer_idx] = worker_id
            
            # Serialize and send layer to worker
            layer = transformer_layers[layer_idx]
            buffer = io.BytesIO()
            torch.save(layer, buffer)
            
            response = self.worker_stubs[worker_id].InitializeWorker(
                inference_pb2.InitRequest(
                    model_layer=buffer.getvalue(),
                    layer_id=layer_idx
                )
            )
            
            if response.success:
                print(f"Layer {layer_idx} initialized on worker {worker_id}")
            else:
                print(f"Failed to initialize layer {layer_idx} on worker {worker_id}: {response.message}")
        
        start_idx = end_idx
    
    # Save input embedding and output layers locally
    self.input_embeddings = model.model.embed_tokens
    self.norm = model.model.norm
    self.output_head = model.lm_head
    
    print("Model loaded and sharded successfully")

Inference execution: The generate method processes input through the distributed model layers.
def generate(self, prompt, max_length=100):
    # Tokenize input
    input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    
    # Get input embeddings
    hidden_states = self.input_embeddings(input_ids)
    
    # Process through each layer sequentially
    for layer_idx in range(len(self.layer_mapping)):
        worker_id = self.layer_mapping[layer_idx]
        
        # Serialize tensor
        buffer = io.BytesIO()
        torch.save(hidden_states, buffer)
        
        # Send to worker
        response = self.worker_stubs[worker_id].ProcessLayer(
            inference_pb2.LayerRequest(
                input_tensor=buffer.getvalue(),
                layer_id=layer_idx,
                is_final=(layer_idx == len(self.layer_mapping) - 1)
            )
        )
        
        if not response.success:
            raise Exception(f"Error processing layer {layer_idx}: {response.error_message}")
        
        # Deserialize output
        buffer = io.BytesIO(response.output_tensor)
        hidden_states = torch.load(buffer)
    
    # Apply final norm and lm_head
    hidden_states = self.norm(hidden_states)
    logits = self.output_head(hidden_states)
    
    # Generate output tokens
    generated_ids = []
    for _ in range(max_length):
        next_token_logits = logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
        generated_ids.append(next_token.item())
        
        # Break if EOS token
        if next_token.item() == self.tokenizer.eos_token_id:
            break
            
        # Get embedding for next token
        next_token_embedding = self.input_embeddings(next_token)
        
        # Process through layers again
        hidden_states = next_token_embedding
        for layer_idx in range(len(self.layer_mapping)):
            worker_id = self.layer_mapping[layer_idx]
            
            # Serialize tensor
            buffer = io.BytesIO()
            torch.save(hidden_states, buffer)
            
            # Send to worker
            response = self.worker_stubs[worker_id].ProcessLayer(
                inference_pb2.LayerRequest(
                    input_tensor=buffer.getvalue(),
                    layer_id=layer_idx,
                    is_final=(layer_idx == len(self.layer_mapping) - 1)
                )
            )
            
            if not response.success:
                raise Exception(f"Error processing layer {layer_idx}: {response.error_message}")
            
            # Deserialize output
            buffer = io.BytesIO(response.output_tensor)
            hidden_states = torch.load(buffer)
        
        # Apply final norm and lm_head
        hidden_states = self.norm(hidden_states)
        logits = self.output_head(hidden_states)
    
    # Decode and return
    output_text = self.tokenizer.decode(generated_ids)
    return output_text

Monitoring: The _monitor_workers method collects performance metrics from worker nodes.
def _monitor_workers(self):
    while True:
        for worker_id, stub in self.worker_stubs.items():
            try:
                response = stub.Heartbeat(
                    inference_pb2.HeartbeatRequest(
                        timestamp=int(time.time())
                    )
                )
                
                self.monitoring_data[worker_id] = {
                    "timestamp": response.timestamp,
                    "gpu_utilization": response.gpu_utilization,
                    "memory_utilization": response.memory_utilization
                }
            except Exception as e:
                print(f"Error monitoring worker {worker_id}: {str(e)}")
        
        # Export metrics for Grafana
        self._export_metrics()
        
        # Sleep for 5 seconds
        time.sleep(5)

Step-by-Step Execution
1. Set up the environment on all nodes
Make sure CUDA and all required packages are installed on both master and worker nodes.

2. Create the protocol buffer and compile it
On both master and worker nodes:

mkdir -p distributed_inference
cd distributed_inference
# Create inference.proto file with the content provided
python3 -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. inference.proto

3. Create worker and master node scripts
Create worker_node.py and master_node.py with the code provided and make them executable:

chmod +x worker_node.py master_node.py

4. Register worker nodes
On the master node:

python3 master_node.py --register --ip 192.168.1.101 --username worker1
python3 master_node.py --register --ip 192.168.1.102 --username worker2
# Add more workers as needed

5. Start worker nodes
On each worker node:

python3 worker_node.py

You should see: "Worker node started on port 50051"

6. Run inference
On the master node:

python3 master_node.py --generate --prompt "Explain quantum computing in simple terms"

The system will:

Load the Gemma-2B model
Shard the model across worker nodes
Process the input through the distributed model
Return the generated text
Troubleshooting
CUDA not found: Ensure CUDA is properly installed and environment variables are set
Connection errors: Check network connectivity between nodes
Out of memory errors: Reduce batch size or distribute across more workers
Slow inference: Check network bandwidth between node