"""PRKT FP16 layout shared by the ONNX exporter, Ultra converter, and v1 repacker.

Tensor order and alignment must match assign_weight_pointers() in src/weights.cpp.
"""

import struct

MAGIC         = struct.pack("<I", 0x544B5250)  # "PRKT"
TENSOR_ALIGN  = 256


def align_up(x: int, a: int) -> int:
    return (x + a - 1) & ~(a - 1)


# ---------------------------------------------------------------------------
# Fixed tensor order — must match assign_weight_pointers() in weights.cpp
# ---------------------------------------------------------------------------

def tensor_order() -> list[str]:
    names = []

    # Subsampling (pre_encode)
    for i in [0, 2, 3, 5, 6]:
        names += [
            f"encoder/pre_encode.conv.{i}.weight",
            f"encoder/pre_encode.conv.{i}.bias",
        ]
    names += [
        "encoder/pre_encode.out.weight",
        "encoder/pre_encode.out.bias",
    ]

    # 24 conformer blocks
    for i in range(24):
        p = f"encoder/layers.{i}"
        names += [
            f"{p}.norm_feed_forward1.weight",
            f"{p}.norm_feed_forward1.bias",
            f"{p}.feed_forward1.linear1.weight",
            f"{p}.feed_forward1.linear2.weight",
            f"{p}.norm_self_att.weight",
            f"{p}.norm_self_att.bias",
            f"{p}.self_attn.linear_q.weight",
            f"{p}.self_attn.linear_k.weight",
            f"{p}.self_attn.linear_v.weight",
            f"{p}.self_attn.linear_pos.weight",
            f"{p}.self_attn.pos_bias_u",
            f"{p}.self_attn.pos_bias_v",
            f"{p}.self_attn.linear_out.weight",
            f"{p}.norm_conv.weight",
            f"{p}.norm_conv.bias",
            f"{p}.conv.pointwise_conv1.weight",
            f"{p}.conv.depthwise_conv.weight",
            f"{p}.conv.depthwise_conv.bias",
            f"{p}.conv.pointwise_conv2.weight",
            f"{p}.norm_feed_forward2.weight",
            f"{p}.norm_feed_forward2.bias",
            f"{p}.feed_forward2.linear1.weight",
            f"{p}.feed_forward2.linear2.weight",
            f"{p}.norm_out.weight",
            f"{p}.norm_out.bias",
        ]

    # Decoder + joint
    names += [
        "decoder/decoder.prediction.embed.weight",
        "decoder/decoder.dec_rnn.lstm.weight_ih",
        "decoder/decoder.dec_rnn.lstm.weight_hh",
        "decoder/decoder.dec_rnn.lstm.bias",
        "decoder/decoder.dec_rnn.lstm.1.weight_ih",
        "decoder/decoder.dec_rnn.lstm.1.weight_hh",
        "decoder/decoder.dec_rnn.lstm.1.bias",
        "decoder/joint.enc.weight",
        "decoder/joint.enc.bias",
        "decoder/joint.pred.weight",
        "decoder/joint.pred.bias",
        "decoder/joint.joint_net.joint_net.2.weight",
        "decoder/joint.joint_net.2.bias",
    ]

    return names
