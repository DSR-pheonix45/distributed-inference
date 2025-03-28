#!/usr/bin/env python3
import torch
import deepspeed
import grpc
import json
import io
import time
import threading
import inference_pb2
import inference_pb2_grpc
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Any
import numpy as np
import os

class DistributedInferenceManager:
    def __init__(self, model_name="google/gemma-2b"):
        self.model_name = model_name
        self.workers = self._load_workers()
        self.worker_stubs = {}
        self.worker_channels = {}
        self.layer_mapping = {}
        self.tokenizer = None
        self.monitoring_data = {}
        
        # Connect to workers
        self._connect_to_workers()
        
        # Load and shard model
        self._load_and_shard_model()
        
        # Start monitoring thread
        self.monitoring_thread = threading.Thread(target=self._monitor_workers)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
    
    def _load_workers(self):
        try:
            with open("workers.json", "r") as f:
                return json.load(f)
        except FileNotFoundError:
            print("No workers.json file found. Please register workers first.")
            return []
    
    def _connect_to_workers(self):
        for i, worker in enumerate(self.workers):
            channel = grpc.insecure_channel(f"{worker['ip']}:50051")
            self.worker_channels[i] = channel
            self.worker_stubs[i] = inference_pb2_grpc.InferenceServiceStub(channel)
            print(f"Connected to worker {i} at {worker['ip']}")
    
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
    
    def _export_metrics(self):
        # Simple file-based metrics export for Grafana
        metrics = []
        timestamp = int(time.time())
        
        for worker_id, data in self.monitoring_data.items():
            metrics.append({
                "measurement": "gpu_utilization",
                "tags": {"worker": f"worker_{worker_id}"},
                "time": timestamp,
                "fields": {"value": data.get("gpu_utilization", 0)}
            })
            
            metrics.append({
                "measurement": "memory_utilization",
                "tags": {"worker": f"worker_{worker_id}"},
                "time": timestamp,
                "fields": {"value": data.get("memory_utilization", 0)}
            })
        
        with open("metrics.json", "w") as f:
            json.dump(metrics, f)
    
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

# Create a simple worker registration script
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

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Distributed Inference Manager")
    parser.add_argument("--register", action="store_true", help="Register a worker")
    parser.add_argument("--ip", help="Worker IP address")
    parser.add_argument("--username", help="Worker username")
    parser.add_argument("--generate", action="store_true", help="Generate text")
    parser.add_argument("--prompt", help="Input prompt for generation")
    
    args = parser.parse_args()
    
    if args.register:
        if not args.ip or not args.username:
            print("Please provide --ip and --username to register a worker")
        else:
            register_worker(args.ip, args.username)
    elif args.generate:
        if not args.prompt:
            print("Please provide --prompt for generation")
        else:
            manager = DistributedInferenceManager()
            output = manager.generate(args.prompt)
            print(f"Generated: {output}")
    else:
        print("Please specify an action: --register or --generate")
