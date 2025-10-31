import os

import neuronxcc.nki as nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
import neuronxcc.nki.typing as nt
import numpy as np
import torch
import torch.nn as nn
import torch_xla.core.xla_model as xm
import benchmark_lib
import logging
import time

logger = logging.getLogger(__name__)


@torch.no_grad()
class FlashAttentionTiled(nn.Module):
  """
  Flash attention with tiled computation and causal masking.
  
  Input shapes:
    q: (seq_len, d_head)
    k: (d_head, seq_len)
    v: (seq_len, d_head)
  
  Output shape:
    (seq_len, d_head + 2) - concatenation of [o, l, m] where:
      o: (seq_len, d_head) - attention output
      l: (seq_len, 1) - log-sum-exp normalizer
      m: (seq_len, 1) - max values
  """
  def __init__(self):
    super().__init__()
  
  def forward(self, q, k, v):
    """
    Args:
      q: Query tensor (seq_len, d_head)
      k: Key tensor (d_head, seq_len)
      v: Value tensor (seq_len, d_head)
      
    Returns:
      Concatenated tensor (seq_len, d_head + 2) containing [o, l, m]
    """
    # Convert numpy to torch if needed
    q = torch.as_tensor(q)
    k = torch.as_tensor(k)
    v = torch.as_tensor(v)
    
    seq_len, d_head = q.shape
    dtype = q.dtype
    device = q.device
    
    # Tile sizes
    b_p_size = 128  # Query tile size
    b_f_size = 512  # K/V tile size
    num_q_tiles = seq_len // b_p_size
    num_k_tiles = seq_len // b_f_size
    
    # Initialize output tensors
    o = torch.zeros(seq_len, d_head, dtype=dtype, device=device)
    l = torch.zeros(seq_len, 1, dtype=dtype, device=device)
    m = torch.zeros(seq_len, 1, dtype=dtype, device=device)
    
    # Process each query tile
    for q_idx in range(num_q_tiles):
      q_start = q_idx * b_p_size
      q_end = q_start + b_p_size
      q_tile = q[q_start:q_end]
      
      # Compute attention scores for all K tiles
      qk_scores = torch.zeros(b_p_size, seq_len, dtype=torch.float32, device=device)
      
      for k_idx in range(num_k_tiles):
        k_start = k_idx * b_f_size
        k_end = k_start + b_f_size
        k_tile = k[:, k_start:k_end]
        
        # Q @ K^T with causal mask
        scores = torch.matmul(q_tile.float(), k_tile.float())
        
        # Causal mask: q_pos >= k_pos
        q_pos = q_start + torch.arange(b_p_size, device=device).unsqueeze(1)
        k_pos = k_start + torch.arange(b_f_size, device=device).unsqueeze(0)
        mask = q_pos >= k_pos
        scores = torch.where(mask, scores, -9984.0)
        
        qk_scores[:, k_start:k_end] = scores
      
      # Softmax with numerical stability
      max_scores = qk_scores.max(dim=1, keepdim=True).values
      attn_weights = torch.exp(qk_scores - max_scores)
      sum_weights = attn_weights.sum(dim=1, keepdim=True)
      
      # Compute output: attention_weights @ V
      o_tile = torch.matmul(attn_weights, v).to(dtype)
      
      # Store results
      o[q_start:q_end] = o_tile
      l[q_start:q_end] = (torch.log(sum_weights) + max_scores).to(dtype)
      m[q_start:q_end] = max_scores.to(dtype)
    
    # Concatenate outputs into a single tensor
    return torch.cat([o, l, m], dim=1)


def test(q, k, v, kernel_dtype=None, acc_type=None, seq_len=2048, d_head=128):
  """Legacy function wrapper for FlashAttentionTiled module."""
  model = FlashAttentionTiled()
  output = model(q, k, v)
  # Split the concatenated output back into o, l, m
  o = output[:, :d_head]
  l = output[:, d_head:d_head+1]
  m = output[:, d_head+1:d_head+2]
  return o.numpy(), l.numpy(), m.numpy()

@nki.jit
def flash_attention_core(q, k, v,
                          kernel_dtype, acc_type,
                          seq_len=2048,
                          d_head=128
                          ):
  """
  The flash attention core function to calcualte self attention for all q tiles with K and V.
  The q, k, and v start in HBM and will be loaded into SBUF as tiles. The block size of K and V
  is defined in the seq_len parameter. Returns o, l, m:
  o: (seq_len, d_head) - output attention values for all q tiles
  l: (seq_len, 1) - log-sum-exp normalizer for all q tiles
  m: (seq_len, 1) - max values for all q tiles
  """
  B_P_SIZE = 128
  B_F_SIZE = 512
  LARGE_TILE_SZ = seq_len
  REDUCTION_TILE = min(2048, LARGE_TILE_SZ // 2)
  num_k_tile_per_large_tile = LARGE_TILE_SZ // B_F_SIZE
  num_q_tiles = seq_len // B_P_SIZE

  i_q_p = nl.arange(B_P_SIZE)[:, None]
  i_q_f = nl.arange(B_F_SIZE)[None, :]
  i_d_p = nl.arange(d_head)[:, None]
  i_d_f = nl.arange(d_head)[None, :]
  i_f_128 = nl.arange(B_P_SIZE)[None, :]
  i_f_k_tiles = nl.arange(num_k_tile_per_large_tile)[None, :]

  # Create local storage for output, l, and m for all q tiles
  o = nl.ndarray((seq_len, d_head), dtype=kernel_dtype, buffer=nl.shared_hbm)
  l = nl.ndarray((seq_len, 1), dtype=kernel_dtype, buffer=nl.shared_hbm)
  m = nl.ndarray((seq_len, 1), dtype=kernel_dtype, buffer=nl.shared_hbm)

  # Loop over all q tiles
  for q_tile_idx in nl.affine_range(num_q_tiles):
    # Load q tile from HBM into SBUF
    q_local_tile = nl.load(q[q_tile_idx * B_P_SIZE + i_q_p, i_d_f], dtype=kernel_dtype)

    # mask are used to only apply computation to the lower half of the matrix,
    # which reduce the arthimetic intensity by half
    forward_mask = q_tile_idx * B_P_SIZE >= 0

    qk_res_buf = nl.ndarray((nl.par_dim(B_P_SIZE), LARGE_TILE_SZ), buffer=nl.sbuf, dtype=acc_type)
    max_local = nl.ndarray((nl.par_dim(B_P_SIZE), num_k_tile_per_large_tile), dtype=acc_type)
    for k_i in nl.affine_range(num_k_tile_per_large_tile):
      # Load k tile from HBM into SBUF
      k_tile = nl.load(k[i_d_p, k_i * B_F_SIZE + i_q_f], dtype=kernel_dtype)
      
      qk_psum = nl.zeros((nl.par_dim(B_P_SIZE), B_F_SIZE),
                          dtype=np.float32, buffer=nl.psum)  # (128, 512)
      multiplication_required_selection = k_i * B_F_SIZE <= q_tile_idx * B_P_SIZE
      qk_psum[i_q_p, i_q_f] += nl.matmul(q_local_tile, k_tile, transpose_x=True,
                                         mask=multiplication_required_selection) # (p(128), 512)

      left_diagonal_selection = q_tile_idx * B_P_SIZE >= (k_i + 1) * B_F_SIZE
      diagonal_and_right_selection = (q_tile_idx * B_P_SIZE < (k_i + 1) * B_F_SIZE) & forward_mask

      q_pos = q_tile_idx * B_P_SIZE + i_q_p
      k_pos = k_i * B_F_SIZE + i_q_f
      pred = q_pos >= k_pos
      # For tiles on and to the right of the diagonal, need to do affine_select.
      # Magic number -9984.0 to replace -inf similar to what Tensorizer uses
      qk_res_buf[i_q_p, k_i * B_F_SIZE + i_q_f] = nisa.affine_select(
        pred=pred,
        on_true_tile=qk_psum[i_q_p, i_q_f], on_false_value=-9984.0, dtype=kernel_dtype,
        mask=diagonal_and_right_selection)

      # For tiles on the left of the diagonal, direct copy, no select required.
      qk_res_buf[i_q_p, k_i * B_F_SIZE + i_q_f] = \
        nl.copy(qk_psum[i_q_p, i_q_f], dtype=kernel_dtype, mask=left_diagonal_selection)

      # Calculate max of the current tile
      max_local[i_q_p, k_i] = nisa.tensor_reduce(np.max, qk_res_buf[i_q_p, k_i * B_F_SIZE + i_q_f], axis=(1,),
                                          dtype=acc_type, negate=False, mask=forward_mask)

    max_ = nisa.tensor_reduce(np.max, max_local[i_q_p, i_f_k_tiles], axis=(1, ),
                      dtype=acc_type, negate=False, mask=forward_mask)
    nl.store(m[q_tile_idx * B_P_SIZE:q_tile_idx * B_P_SIZE+B_P_SIZE, :], value=nl.copy(max_, dtype=kernel_dtype))
    m_current = max_

    p_local = nl.ndarray((nl.par_dim(B_P_SIZE), LARGE_TILE_SZ), dtype=kernel_dtype)
    i_r_f = nl.arange(REDUCTION_TILE)[None,: ]
    p_partial_sum = nl.ndarray((nl.par_dim(B_P_SIZE), LARGE_TILE_SZ // REDUCTION_TILE), dtype=acc_type)
    for k_r_i in nl.affine_range(LARGE_TILE_SZ // REDUCTION_TILE):
      # compute exp(qk-max)
      p_local[i_q_p, k_r_i * REDUCTION_TILE + i_r_f] = \
        nisa.activation(np.exp,
                        qk_res_buf[i_q_p, k_r_i * REDUCTION_TILE + i_r_f],
                        bias=-1 * m_current,
                        scale=1.0,
                        dtype=kernel_dtype,
                        mask=forward_mask)

      # Compute partial row-tile sum of exp(qk-max))
      p_partial_sum[i_q_p, k_r_i] = nl.sum(p_local[i_q_p, k_r_i * REDUCTION_TILE + i_r_f], axis=1, dtype=acc_type, mask=forward_mask)

    p_local_transposed = nl.ndarray((nl.par_dim(B_P_SIZE), LARGE_TILE_SZ), dtype=kernel_dtype)
    for i_p_t in nl.affine_range(LARGE_TILE_SZ // 512):
      p_local_t_tmp = nl.ndarray((nl.par_dim(B_P_SIZE), 512), buffer=nl.psum, dtype=np.float32)
      for i_p_t_local in nl.affine_range(512//128):
        p_local_t_tmp[i_q_p, i_p_t_local*128 + i_f_128] = nisa.nc_transpose(p_local[i_q_p, i_p_t*512+i_p_t_local * B_P_SIZE + i_f_128], mask=forward_mask)
      i_f_512 = nl.arange(512)[None, :]
      p_local_transposed[i_q_p, i_p_t * 512 + i_f_512 ] = nl.copy(p_local_t_tmp[i_q_p, i_f_512], dtype=kernel_dtype, mask=forward_mask)

    ps = nl.sum(p_partial_sum, axis=1, dtype=acc_type, mask=forward_mask)
    pv_psum = nl.zeros((nl.par_dim(B_P_SIZE), d_head), dtype=np.float32, buffer=nl.psum)
    for k_i in nl.affine_range(LARGE_TILE_SZ // B_P_SIZE):
      # Load v tile from HBM into SBUF
      v_tile = nl.load(v[k_i * B_P_SIZE + i_q_p, i_d_f], dtype=kernel_dtype)
      
      pv_psum[i_q_p, i_d_f] += nl.matmul(p_local_transposed[i_q_p, k_i * B_P_SIZE + i_f_128],
                                         v_tile,
                                         transpose_x=True,
                                         mask=forward_mask) # (128, 128) (p(Br), d)

    nl.store(o[q_tile_idx * B_P_SIZE:q_tile_idx * B_P_SIZE+B_P_SIZE, :], value=nl.copy(pv_psum[i_q_p, i_d_f], dtype=kernel_dtype))
    nl.store(l[q_tile_idx * B_P_SIZE:q_tile_idx * B_P_SIZE+B_P_SIZE, :], value=nl.add(nl.log(ps), max_))
  
  return o, l, m

def test_nki(ref_func, compiled_module):
  """Test the Neuron-compiled PyTorch module against the NKI reference implementation."""
  d_head = 128
  seq_len = 2048
  
  logger.info(f"Testing flash attention with d_head={d_head} and seq_len={seq_len}...")
  
  # Create random inputs
  q_np = np.random.rand(seq_len, d_head).astype(nl.float32)
  k_np = np.random.rand(d_head, seq_len).astype(nl.float32)
  v_np = np.random.rand(seq_len, d_head).astype(nl.float32)
  
  # Convert to torch tensors
  q = torch.from_numpy(q_np)
  k = torch.from_numpy(k_np)
  v = torch.from_numpy(v_np)
  
  # Run the NKI reference
  o_ref, l_ref, m_ref = ref_func(
    q_np, k_np, v_np,
    kernel_dtype=nl.bfloat16,
    acc_type=nl.float32,
    seq_len=seq_len,
    d_head=d_head
  )
  
  # Run the Neuron-compiled version
  logger.info("Testing Neuron-compiled module...")
  output_neuron = compiled_module(q, k, v)
  # Split the concatenated output back into o, l, m
  o_neuron_torch = output_neuron[:, :d_head]
  l_neuron_torch = output_neuron[:, d_head:d_head+1]
  m_neuron_torch = output_neuron[:, d_head+1:d_head+2]
  o_neuron = o_neuron_torch.numpy()
  l_neuron = l_neuron_torch.numpy()
  m_neuron = m_neuron_torch.numpy()
  
  fail = False
  if not np.allclose(o_ref.astype(nl.float32), o_neuron.astype(nl.float32), atol=0.01, rtol=0.001):
    logger.error("FAIL: o_ref != o_neuron (Neuron-compiled)")
    logger.error(f"o_ref shape: {o_ref.shape}, o_neuron shape: {o_neuron.shape}")
    logger.error(f"o_ref:\n{o_ref.astype(nl.float32)[:5,:5]}")
    logger.error(f"o_neuron:\n{o_neuron.astype(nl.float32)[:5,:5]}")
    fail = True
  if not np.allclose(l_ref.astype(nl.float32), l_neuron.astype(nl.float32), atol=0.01, rtol=0.001):
    logger.error("FAIL: l_ref != l_neuron (Neuron-compiled)")
    logger.error(f"l_ref shape: {l_ref.shape}, l_neuron shape: {l_neuron.shape}")
    logger.error(f"l_ref:\n{l_ref.astype(nl.float32)[:5]}")
    logger.error(f"l_neuron:\n{l_neuron.astype(nl.float32)[:5]}")
    fail = True
  if not np.allclose(m_ref.astype(nl.float32), m_neuron.astype(nl.float32), atol=0.01, rtol=0.001):
    logger.error("FAIL: m_ref != m_neuron (Neuron-compiled)")
    logger.error(f"m_ref shape: {m_ref.shape}, m_neuron shape: {m_neuron.shape}")
    logger.error(f"m_ref:\n{m_ref.astype(nl.float32)[:5]}")
    logger.error(f"m_neuron:\n{m_neuron.astype(nl.float32)[:5]}")
    fail = True
  if fail:
    return False
  
  logger.info("✓ All tests passed (Neuron-compiled)")
  
  return True

if __name__ == "__main__":
  # Configure logging
  logging.basicConfig(
    format='[%(levelname)s %(name)s] %(message)s',
    level=logging.INFO
  )
  
  # Configuration
  d_head = 128
  seq_len = 2048
  
  # Get XLA device
  device = xm.xla_device()
  logger.info(f"Using device: {device}")
  
  # Instantiate the PyTorch module
  logger.info(f"Instantiating FlashAttentionTiled")
  model = FlashAttentionTiled().to(device)
  
  # Create inputs and move to XLA device
  q = torch.randn(seq_len, d_head, dtype=torch.float32).to(device)
  k = torch.randn(d_head, seq_len, dtype=torch.float32).to(device)
  v = torch.randn(seq_len, d_head, dtype=torch.float32).to(device)
  
  # Run tests
  logger.info("=" * 80)
  logger.info("TESTING")
  logger.info("=" * 80)
  
  # Create test inputs
  q_np = np.random.rand(seq_len, d_head).astype(nl.float32)
  k_np = np.random.rand(d_head, seq_len).astype(nl.float32)
  v_np = np.random.rand(seq_len, d_head).astype(nl.float32)
  
  # Convert to torch tensors on XLA device
  q_test = torch.from_numpy(q_np).to(device)
  k_test = torch.from_numpy(k_np).to(device)
  v_test = torch.from_numpy(v_np).to(device)
  
  # Run the NKI reference
  o_ref, l_ref, m_ref = flash_attention_core(
    q_np, k_np, v_np,
    kernel_dtype=nl.bfloat16,
    acc_type=nl.float32,
    seq_len=seq_len,
    d_head=d_head
  )
  
  # Run the XLA model
  logger.info("Testing XLA model...")
  output_xla = model(q_test, k_test, v_test)
  
  # Mark step to execute the computation
  xm.mark_step()
  
  # Split the concatenated output back into o, l, m and move to CPU
  o_xla = output_xla[:, :d_head].cpu().numpy()
  l_xla = output_xla[:, d_head:d_head+1].cpu().numpy()
  m_xla = output_xla[:, d_head+1:d_head+2].cpu().numpy()
  
  fail = False
  if not np.allclose(o_ref.astype(nl.float32), o_xla.astype(nl.float32), atol=0.01, rtol=0.001):
    logger.error("FAIL: o_ref != o_xla")
    logger.error(f"o_ref shape: {o_ref.shape}, o_xla shape: {o_xla.shape}")
    logger.error(f"o_ref:\n{o_ref.astype(nl.float32)[:5,:5]}")
    logger.error(f"o_xla:\n{o_xla.astype(nl.float32)[:5,:5]}")
    fail = True
  if not np.allclose(l_ref.astype(nl.float32), l_xla.astype(nl.float32), atol=0.01, rtol=0.001):
    logger.error("FAIL: l_ref != l_xla")
    logger.error(f"l_ref shape: {l_ref.shape}, l_xla shape: {l_xla.shape}")
    logger.error(f"l_ref:\n{l_ref.astype(nl.float32)[:5]}")
    logger.error(f"l_xla:\n{l_xla.astype(nl.float32)[:5]}")
    fail = True
  if not np.allclose(m_ref.astype(nl.float32), m_xla.astype(nl.float32), atol=0.01, rtol=0.001):
    logger.error("FAIL: m_ref != m_xla")
    logger.error(f"m_ref shape: {m_ref.shape}, m_xla shape: {m_xla.shape}")
    logger.error(f"m_ref:\n{m_ref.astype(nl.float32)[:5]}")
    logger.error(f"m_xla:\n{m_xla.astype(nl.float32)[:5]}")
    fail = True
  
  if fail:
    logger.error("Test failed")
    exit(1)
  
  logger.info("✓ All tests passed (XLA)")
  
  # Run benchmarks
  logger.info("\n" + "=" * 80)
  logger.info("BENCHMARKING")
  logger.info("=" * 80)
  
  num_warmup = 2
  num_timed = 10
  
  # Warmup iterations
  logger.info(f"Running {num_warmup} warmup iterations...")
  for _ in range(num_warmup):
    output = model(q, k, v)
    xm.mark_step()
  
  # Timed iterations
  logger.info(f"Running {num_timed} timed iterations...")
  start_time = time.time()
  for _ in range(num_timed):
    output = model(q, k, v)
    xm.mark_step()
  end_time = time.time()
  
  avg_runtime = (end_time - start_time) / num_timed
  
  logger.info("=" * 80)
  logger.info(f"BENCHMARK RESULTS:")
  logger.info(f"  XLA runtime per iteration: {avg_runtime:.4f}s")
  logger.info(f"  Latency: {avg_runtime * 1000:.3f} ms")
  logger.info("=" * 80)
