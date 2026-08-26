# OffloaderV1 Investigation Notes

This note records the current findings. It does not introduce a PR3 code fix.

## Local status

- PR1 addresses Gemma RMSNorm non-persistent-buffer handling.
- PR2 addresses duplicate state entries for tied/aliased parameters.
- Both fixes currently exist only as local commits and have not been landed upstream.
- No PR3 implementation is included in this note.

## PR3 finding: stale GDN tensor view

`Qwen3_5GatedDeltaNet` creates a view during initialization:

```python
conv_weights = self.conv1d.weight.view(...)
self.attn = RadixLinearAttention(conv_weights=conv_weights, ...)
```

`RadixLinearAttention.conv_weights` is an ordinary tensor attribute. It is not a
registered parameter or buffer, so it is not part of `state_dict()` and is not
updated by `OffloaderV1`'s `functional_call` state mapping.

The relevant lifecycle is:

```text
layer construction
  -> make_layers() applies OffloaderV1
  -> OffloaderV1 replaces conv1d.weight storage with CPU storage
  -> checkpoint loading writes the real weight into the CPU Parameter
  -> the original CUDA conv_weights view remains stale
  -> GDN uses RadixLinearAttention.conv_weights
```

The GDN backend passes `layer.conv_weights` to the causal convolution kernel,
so this stale view can produce incorrect output without a device-mismatch
exception.

## Experimental confirmation

With Ornith-1.5-35B-A3B-FP8, TP=2, `--cpu-offload-gb 4`, and CUDA graphs
disabled:

- The unmodified path produced `!!!!!!!!!!!!!!!!` and token IDs consisting of
  zeros.
- A temporary change in `Qwen3_5GatedDeltaNet.forward()` regenerated
  `conv_weights` from the current `self.conv1d.weight` immediately before
  calling `self.attn(...)`.
- The same test then returned normal model output.
- The temporary change was reverted after the test.

This confirms the stale-view mechanism as the current PR3 root cause. The
formal fix is intentionally left undecided between a model-side dynamic view
and an explicit OffloaderV1 rebind/restore mechanism.

Relevant files:

- `python/sglang/srt/utils/offloader.py`
- `python/sglang/srt/models/qwen3_5.py`
- `python/sglang/srt/layers/radix_linear_attention.py`
- `python/sglang/srt/layers/attention/linear/gdn_backend.py`
