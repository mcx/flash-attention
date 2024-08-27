import torch
#from flash_attn_interface import flash_attn_func, flash_attn_varlen_func, flash_attn_with_kvcache
import flash_attn_interface as fa3
import flash_attn as fa2
import torch.utils.benchmark as benchmark

import argparse
import math

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--causal', action='store_true')
parser.add_argument('--splits', type=int, default=1)

args = parser.parse_args()

def benchmark_fa_kv(fn, repeats=10, desc='', verbose=True, **kwinputs):
    """Use Pytorch Benchmark on the forward pass of an arbitrary function."""
    if verbose:
        print(desc, '- Forward pass')
    t = benchmark.Timer(
            stmt='fn(**kwinputs)',
            globals={'fn': fn, 'kwinputs': kwinputs},
            num_threads=torch.get_num_threads(),
            )
    m = t.timeit(repeats)
    if verbose:
        print(m)
    return t, m

def main():
    # *SAMPLE CONFIG*
    # Model arch params:
    nheads_q = 64
    nheads_kv = 8
    headdim = 128
    #dtype = torch.bfloat16
    dtype = torch.float16

    # Cache settings:
    num_caches = 8
    cache_seqlen = 1024 * 16

    # Batching settings
    ntokens = 1024
    max_queries_per_batch = 4
    small_request_ntokens = 16

    # Input settings
    query_seqlens = [900, 12, 3]
    num_queries = len(query_seqlens)
    # Need to add empty queries to fill out `max_queries_per_batch`
    num_padding_queries = max_queries_per_batch - num_queries
    context_seqlens = [4096, 5120*2, 6145*2]

    # Validation
    assert sum(query_seqlens) <= ntokens
    assert all(s < small_request_ntokens for s in query_seqlens[1:])
    assert num_queries <= max_queries_per_batch
    assert all(s < cache_seqlen for s in context_seqlens)

    torch.manual_seed(5434)

    # Allocate some tensors
    k_cache = torch.randn(
        (num_caches, cache_seqlen, nheads_kv, headdim), device="cuda", dtype=dtype
    )
    v_cache = torch.randn(
        (num_caches, cache_seqlen, nheads_kv, headdim), device="cuda", dtype=dtype
    )

    q_buf_large = torch.randn(
        (1, ntokens, nheads_q, headdim), device="cuda", dtype=dtype
    )
    cache_seqlen_large = torch.tensor(
        [context_seqlens[0]], dtype=torch.int32, device="cuda"
    )
    cache_idx_large = torch.tensor([1], dtype=torch.int32, device="cuda")

    q_buf_small = torch.randn(
        (max_queries_per_batch - 1, small_request_ntokens, nheads_q, headdim),
        device="cuda",
        dtype=dtype,
    )
    cache_seqlens_small = torch.tensor(
        context_seqlens[1:] + [0] * num_padding_queries, dtype=torch.int32, device="cuda"
    )
    cache_idxs_small = torch.randperm(num_caches, dtype=torch.int32, device="cuda")[
        : max_queries_per_batch - 1
    ]

    # Call flash attn
    # First for the single full-sized query
    out0 = fa3.flash_attn_with_kvcache(
        q=q_buf_large,
        k_cache=k_cache,
        v_cache=v_cache,
        cache_seqlens=cache_seqlen_large,
        cache_batch_idx=cache_idx_large,
        causal=bool(args.causal),
        #num_splits=args.splits
        num_splits=1
    )

    # Second for n-1 small queries
    out1 = fa3.flash_attn_with_kvcache(
        q=q_buf_small,
        k_cache=k_cache,
        v_cache=v_cache,
        cache_seqlens=cache_seqlens_small,
        cache_batch_idx=cache_idxs_small,
        causal=bool(args.causal),
        num_splits=args.splits
    )

      # Call flash attn
    # First for the single full-sized query
    out2 = fa2.flash_attn_with_kvcache(
        q=q_buf_large,
        k_cache=k_cache,
        v_cache=v_cache,
        cache_seqlens=cache_seqlen_large,
        cache_batch_idx=cache_idx_large,
        causal=bool(args.causal),
        num_splits=args.splits
    )
    print ((out0[0] - out2).abs().max().item())


    # Second for n-1 small queries
    out3 = fa2.flash_attn_with_kvcache(
        q=q_buf_small,
        k_cache=k_cache,
        v_cache=v_cache,
        cache_seqlens=cache_seqlens_small,
        cache_batch_idx=cache_idxs_small,
        causal=bool(args.causal),
        #num_splits=args.splits
        num_splits=1
    )

    # Second for n-1 small queries
    out3, _, _  = fa3.flash_attn_with_kvcache(
        q=q_buf_small,
        k_cache=k_cache,
        v_cache=v_cache,
        cache_seqlens=cache_seqlens_small,
        cache_batch_idx=cache_idxs_small,
        causal=bool(args.causal),
        #num_splits=args.splits
        num_splits=1
    )

     # Second for n-1 small queries
    out4, _, _  = fa3.flash_attn_with_kvcache(
        q=q_buf_small,
        k_cache=k_cache,
        v_cache=v_cache,
        cache_seqlens=cache_seqlens_small,
        cache_batch_idx=cache_idxs_small,
        causal=bool(args.causal),
        #num_splits=args.splits
        num_splits=1
    )

    print ('fa3-split\n', out1[0], 'fa3\n', out3)

    print (out1[1][0,0,0,0,0], out1[2][0,0,0,0])
    print (out1[1][1,0,0,0,0], out1[2][1,0,0,0])
    out00 = out1[1][0,0,0,0,0]
    lse00 = out1[2][0,0,0,0]
    out01 = out1[1][1,0,0,0,0]
    lse01 = out1[2][1,0,0,0]
    lse_max = max(lse00, lse01)
    lse = math.log(math.exp(lse00) + math.exp(lse01))
    lse_sum = math.exp(lse00 - lse_max) + math.exp(lse01 - lse_max)
    lse_logsum = math.log(lse_sum) + lse_max;
    newout00 = out00 * math.exp(lse00 - lse_logsum)
    newout01 = out01 * math.exp(lse01 - lse_logsum)
    combined = newout00 + newout01
    print(newout00, newout01, 'combined', combined)
    print ('lse', lse)
    print ('diff\n', (out1[0] - out3).abs().max().item())
    print ('diff\n', (out1[0] - out3).abs().mean().item())

    return

    benchmark_fa_kv(fa3.flash_attn_with_kvcache, repeats=10, desc='', verbose=True,  
        q=q_buf_large,
        k_cache=k_cache,
        v_cache=v_cache,
        cache_seqlens=cache_seqlen_large,
        cache_batch_idx=cache_idx_large,
        causal=bool(args.causal),
        #num_splits=args.splits
        num_splits=1
    )

    benchmark_fa_kv(fa3.flash_attn_with_kvcache, repeats=10, desc='', verbose=True,  
        q=q_buf_small,
        k_cache=k_cache,
        v_cache=v_cache,
        cache_seqlens=cache_seqlens_small,
        cache_batch_idx=cache_idxs_small,
        causal=bool(args.causal),
        num_splits=args.splits
    )

    print ('fa2 ')

    for k in [1]:
        benchmark_fa_kv(fa2.flash_attn_with_kvcache, repeats=10, desc='', verbose=True,
            q=q_buf_large,
            k_cache=k_cache,
            v_cache=v_cache,
            cache_seqlens=cache_seqlen_large,
            cache_batch_idx=cache_idx_large,
        causal=bool(args.causal),
        num_splits=args.splits
    )

    for k in [1]:
        benchmark_fa_kv(fa2.flash_attn_with_kvcache, repeats=10, desc='', verbose=True,
            q=q_buf_small,
            k_cache=k_cache,
            v_cache=v_cache,
            cache_seqlens=cache_seqlens_small,
            cache_batch_idx=cache_idxs_small,
        causal=bool(args.causal),
        num_splits=args.splits
    )

if __name__ == "__main__":
    main()
