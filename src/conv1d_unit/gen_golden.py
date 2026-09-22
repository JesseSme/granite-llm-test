"""Generate golden sample for conv1d_unit cocotb test.

Uses PyTorch to compute causal depthwise 1D convolution and saves
input/output pairs as hex files for the RTL testbench.

The hardware implements:
  output[c,t] = weight[c,0]*input[c,t] + weight[c,1]*input[c,t-1]
              + weight[c,2]*input[c,t-2] + weight[c,3]*input[c,t-3] + bias[c]

The input stream provides raw (unpadded) input. The circular buffer
stores 3 previous timesteps, initialized to zero (causal padding).
"""

import torch
import struct
import os


def float_to_bfloat16(val):
    """Convert Python float to 16-bit bfloat16 pattern."""
    f32 = struct.pack('f', val)
    i32 = struct.unpack('I', f32)[0]
    rounding_bias = (1 << 15) + ((i32 >> 16) & 1)
    i32 += rounding_bias
    bf16 = i32 >> 16
    return bf16 & 0xFFFF


def bfloat16_to_float(val):
    """Convert 16-bit bfloat16 to float."""
    val = val & 0xFFFF
    float32_val = val << 16
    return struct.unpack('f', struct.pack('I', float32_val))[0]


def main():
    torch.manual_seed(42)

    CHANNELS = 1536
    KERNEL = 4
    PADDING = 3
    SEQ_LEN = 32

    conv = torch.nn.Conv1d(
        in_channels=CHANNELS,
        out_channels=CHANNELS,
        kernel_size=KERNEL,
        groups=CHANNELS,
        bias=True,
        padding=PADDING,
        padding_mode='zeros'
    )

    torch.manual_seed(123)
    conv.weight.data = torch.randn(CHANNELS, 1, KERNEL, dtype=torch.bfloat16)
    conv.bias.data = torch.randn(CHANNELS, dtype=torch.bfloat16)

    # x is the RAW input stream (no extra padding)
    # Shape: (1, CHANNELS, SEQ_LEN)
    torch.manual_seed(456)
    x = torch.randn(1, CHANNELS, SEQ_LEN, dtype=torch.bfloat16)

    # PyTorch conv with padding=3 handles causal zero-padding internally
    with torch.no_grad():
        y = conv(x)  # shape: (1, CHANNELS, SEQ_LEN + PADDING)

    # Remove the extra PADDING positions from the right
    # y[:, :, :SEQ_LEN] is the causal output for positions 0..SEQ_LEN-1
    y_causal = y[:, :, :SEQ_LEN]  # shape: (1, CHANNELS, SEQ_LEN)

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Save weights: channel-major, tap order
    # IMPORTANT: RTL implements convolution as:
    #   output[t] = w[0]*x[t] + w[1]*x[t-1] + w[2]*x[t-2] + w[3]*x[t-3]
    # PyTorch Conv1d implements cross-correlation (reversed kernel):
    #   output[t] = w[0]*x[t+3] + w[1]*x[t+2] + w[2]*x[t+1] + w[3]*x[t]
    # So we reverse the kernel to match RTL convention.
    weights_path = os.path.join(script_dir, "golden_weights.hex")
    with open(weights_path, 'w') as f:
        for c in range(CHANNELS):
            for k in range(KERNEL):
                # Reverse kernel: RTL w[k] = PyTorch w[KERNEL-1-k]
                val = float_to_bfloat16(conv.weight.data[c, 0, KERNEL-1-k].item())
                f.write(f"{val:04X}\n")

    # Save biases
    biases_path = os.path.join(script_dir, "golden_biases.hex")
    with open(biases_path, 'w') as f:
        for c in range(CHANNELS):
            val = float_to_bfloat16(conv.bias.data[c].item())
            f.write(f"{val:04X}\n")

    # Save inputs in stream order: for each timestep, all channels
    inputs_path = os.path.join(script_dir, "golden_inputs.hex")
    with open(inputs_path, 'w') as f:
        for t in range(SEQ_LEN):
            for c in range(CHANNELS):
                val = float_to_bfloat16(x[0, c, t].item())
                f.write(f"{val:04X}\n")

    # Save outputs in stream order
    outputs_path = os.path.join(script_dir, "golden_outputs.hex")
    with open(outputs_path, 'w') as f:
        for t in range(SEQ_LEN):
            for c in range(CHANNELS):
                val = float_to_bfloat16(y_causal[0, c, t].item())
                f.write(f"{val:04X}\n")

    print(f"Generated golden sample:")
    print(f"  Weights: {weights_path} ({CHANNELS * KERNEL} values)")
    print(f"  Biases:  {biases_path} ({CHANNELS} values)")
    print(f"  Inputs:  {inputs_path} ({SEQ_LEN * CHANNELS} values)")
    print(f"  Outputs: {outputs_path} ({SEQ_LEN * CHANNELS} values)")
    print(f"  Sequence length: {SEQ_LEN}, Channels: {CHANNELS}")

    # Verify first channel manually
    print("\nVerification (channel 0):")
    w = conv.weight.data[0, 0, :].tolist()
    b = conv.bias.data[0].item()
    for t in range(4):
        inp_vals = [float_to_bfloat16(x[0, 0, max(0, t - k)].item()) for k in range(KERNEL)]
        out_val = float_to_bfloat16(y_causal[0, 0, t].item())
        print(f"  t={t}: inputs={[f'{v:04X}' for v in inp_vals]}, "
              f"expected=0x{out_val:04X} ({bfloat16_to_float(out_val):.4f})")


if __name__ == "__main__":
    main()
