import grpc
import torch
import inference_pb2
import inference_pb2_grpc
import io
import concurrent.futures
import time
import pynvml
from concurrent import futures

class InferenceServicer(inference_pb2_grpc.InferenceServiceServicer):
    def __init__(self):
        self.layer = None
        self.layer_id = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        
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

def serve(port=50051):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    inference_pb2_grpc.add_InferenceServiceServicer_to_server(
        InferenceServicer(), server)
    server.add_insecure_port(f'[::]:{port}')
    server.start()
    print(f"Worker node started on port {port}")
    try:
        while True:
            time.sleep(86400)  # Sleep for a day
    except KeyboardInterrupt:
        server.stop(0)

if __name__ == '__main__':
    serve()