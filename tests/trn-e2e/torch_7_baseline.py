import numpy as np
import torch
import torch.nn as nn
import time

class LogitsBaseline(nn.Module):
    """Torch baseline for LM head logits computation."""
    def __init__(self, vocab_size=64128, hidden_size=2048):
        super().__init__()
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
    
    def forward(self, hidden_states):
        """
        Compute logits: hidden_states @ lm_head.weight.T
        Args:
            hidden_states: (1, 1, 2048)
        Returns:
            logits: (1, 1, 64128)
        """
        return self.lm_head(hidden_states)

def get_test_data(dtype):
    """Create test data for logits kernel."""
    # hidden_states: (1, 1, 2048)
    hidden_states = np.random.randn(1, 1, 2048).astype(dtype)
    # lm_head_weight: (2048, 64128)
    lm_head_weight = np.random.randn(2048, 64128).astype(dtype)
    return (hidden_states, lm_head_weight)

def forward_torch(hidden_states, lm_head_weight):
    """
    Compute logits using torch: hidden_states @ lm_head_weight
    
    Args:
        hidden_states: (1, 1, 2048)
        lm_head_weight: (2048, 64128)
    Returns:
        logits: (1, 1, 64128)
    """
    baseline = LogitsBaseline(vocab_size=64128, hidden_size=2048)
    # Set the weight (note: nn.Linear stores weight as (out_features, in_features))
    # So we need to transpose lm_head_weight from (2048, 64128) to (64128, 2048)
    baseline.lm_head.weight.data = torch.from_numpy(lm_head_weight.T)
    hidden_states_torch = torch.from_numpy(hidden_states)
    with torch.no_grad():
        output = baseline(hidden_states_torch)
    return output.numpy()

def benchmark_torch(hidden_states, lm_head_weight, warmup=10, iters=100):
    """Benchmark torch implementation."""
    baseline = LogitsBaseline(vocab_size=64128, hidden_size=2048)
    baseline.lm_head.weight.data = torch.from_numpy(lm_head_weight.T)
    hidden_states_torch = torch.from_numpy(hidden_states)
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = baseline(hidden_states_torch)
    
    # Benchmark
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(iters):
            _ = baseline(hidden_states_torch)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end = time.perf_counter()
    
    avg_time_ms = (end - start) / iters * 1000.0
    return avg_time_ms

if __name__ == "__main__":
    np.random.seed(0)
    
    print("Benchmarking torch performance...")
    print("Shapes: hidden_states (1, 1, 2048), lm_head_weight (2048, 64128)")
    print()
    
    # Benchmark with float32
    test_data_f32 = get_test_data(np.float32)
    hidden_states_f32, lm_head_weight_f32 = test_data_f32
    
    torch_time_f32 = benchmark_torch(hidden_states_f32, lm_head_weight_f32)
    print(f"Torch (float32) latency: {torch_time_f32:.3f} ms")
    
    # Benchmark with bfloat16 if available
    if hasattr(torch, 'bfloat16'):
        test_data_bf16 = get_test_data(np.float32)  # Use float32 as input, torch will convert
        hidden_states_bf16, lm_head_weight_bf16 = test_data_bf16
        hidden_states_torch_bf16 = torch.from_numpy(hidden_states_bf16).to(torch.bfloat16)
        lm_head_weight_torch_bf16 = torch.from_numpy(lm_head_weight_bf16.T).to(torch.bfloat16)
        
        baseline_bf16 = LogitsBaseline(vocab_size=64128, hidden_size=2048)
        baseline_bf16.lm_head.weight.data = lm_head_weight_torch_bf16
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = baseline_bf16(hidden_states_torch_bf16)
        
        # Benchmark
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(100):
                _ = baseline_bf16(hidden_states_torch_bf16)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end = time.perf_counter()
        
        torch_time_bf16 = (end - start) / 100 * 1000.0
        print(f"Torch (bfloat16) latency: {torch_time_bf16:.3f} ms")
