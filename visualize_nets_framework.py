"""Generate a detailed framework diagram from the networks in ``nets``.

The diagram is intentionally described here instead of importing the previous
visualization script.  Its stages, feature names, operations, and tensor
sizes mirror ``nets/funiegan_features.py``, ``nets/crackformer.py`` and
``nets/fusion.py`` for a 256 x 256 input.
"""
import argparse
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


WIDTH, HEIGHT = 2600, 1800
COLORS = {
    "background": "#FAFBFC",
    "text": "#17202A",
    "muted": "#52606D",
    "border": "#AAB4BE",
    "funie_fill": "#D9EEF7",
    "funie_line": "#2479A4",
    "fusion_fill": "#FCE8B2",
    "fusion_line": "#B7791F",
    "crack_fill": "#E4F1DE",
    "crack_line": "#397A35",
    "decoder_fill": "#E9E3F8",
    "decoder_line": "#6553A6",
    "output_fill": "#D7E8F8",
    "output_line": "#316B9A",
    "loss_fill": "#F5DCE3",
    "loss_line": "#A43A58",
    "warning_fill": "#FFF0D5",
    "warning_line": "#AF6B00",
    "arrow": "#59636E",
}
FONT_ROOT = Path("C:/Windows/Fonts")


def get_font(size, bold=False):
    names = ("segoeuib.ttf", "msyhbd.ttc") if bold else ("segoeui.ttf", "msyh.ttc")
    for name in names:
        path = FONT_ROOT / name
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


TITLE = get_font(42, True)
SECTION = get_font(27, True)
LABEL = get_font(19, True)
BODY = get_font(16)
SMALL = get_font(14)
TINY = get_font(12)


def text_center(draw, rect, value, font=BODY, fill=None):
    bounds = draw.textbbox((0, 0), value, font=font)
    x = (rect[0] + rect[2] - bounds[2] - bounds[0]) / 2
    y = (rect[1] + rect[3] - bounds[3] - bounds[1]) / 2
    draw.text((x, y), value, font=font, fill=fill or COLORS["text"])


def node(draw, rect, title, detail, fill, outline, note=None, title_size=None):
    draw.rounded_rectangle(rect, radius=10, fill=fill, outline=outline, width=3)
    title_font = title_size or LABEL
    text_center(draw, (rect[0] + 4, rect[1] + 5, rect[2] - 4, rect[1] + 34),
                title, title_font, outline)
    text_center(draw, (rect[0] + 4, rect[1] + 34, rect[2] - 4, rect[3] - 5),
                detail, SMALL, COLORS["muted"])
    if note:
        text_center(draw, (rect[0], rect[3] + 3, rect[2], rect[3] + 21),
                    note, TINY, COLORS["muted"])


def section(draw, rect, title, fill, outline):
    draw.rounded_rectangle(rect, radius=18, fill=fill, outline=outline, width=2)
    draw.text((rect[0] + 22, rect[1] + 15), title, font=SECTION, fill=outline)


def arrow(draw, start, end, color=None, width=3, dashed=False):
    color = color or COLORS["arrow"]
    x1, y1 = start
    x2, y2 = end
    if dashed:
        distance = max(abs(x2 - x1), abs(y2 - y1))
        pieces = max(2, int(distance / 18))
        for i in range(0, pieces, 2):
            a = i / pieces
            b = min((i + 1) / pieces, 1.0)
            draw.line((x1 + (x2 - x1) * a, y1 + (y2 - y1) * a,
                       x1 + (x2 - x1) * b, y1 + (y2 - y1) * b),
                      fill=color, width=width)
    else:
        draw.line((x1, y1, x2, y2), fill=color, width=width)
    angle = math.atan2(y2 - y1, x2 - x1)
    head = 14
    points = [(x2, y2)]
    for offset in (-0.55, 0.55):
        points.append((x2 - head * math.cos(angle + offset),
                       y2 - head * math.sin(angle + offset)))
    draw.polygon(points, fill=color)


def line_label(draw, xy, value, fill=None, font=SMALL):
    bounds = draw.textbbox((0, 0), value, font=font)
    draw.text((xy[0] - (bounds[2] - bounds[0]) / 2, xy[1]), value,
              font=font, fill=fill or COLORS["muted"])


def render(output):
    image = Image.new("RGB", (WIDTH, HEIGHT), COLORS["background"])
    draw = ImageDraw.Draw(image)
    draw.text((55, 27), "FUnIE-GAN + CrackFormer-II integrated framework",
              font=TITLE, fill=COLORS["text"])
    draw.text((58, 86),
              "Generated from the actual forward path in nets/fusion.py (input: 3 x 256 x 256)",
              font=BODY, fill=COLORS["muted"])

    section(draw, (40, 130, 2560, 515),
            "1. FUnIE-GAN enhancement branch (GeneratorFunieGANFeatures)",
            "#EDF7FB", COLORS["funie_line"])
    section(draw, (40, 545, 2560, 1015),
            "2. CrackFormer-II encoder + five FeatureFusionBlock injections",
            "#F1F8EE", COLORS["crack_line"])
    section(draw, (40, 1045, 2560, 1455),
            "3. CrackFormer-II decoder, LA attention and original Fuse blocks",
            "#F2F0FA", COLORS["decoder_line"])
    section(draw, (40, 1490, 2560, 1760),
            "4. Training outputs and joint objective", "#FFF4F6", COLORS["loss_line"])

    # ------------------------------------------------------------------
    # FUnIE-GAN: exact UNetDown/UNetUp sequence and returned feature taps.
    # ------------------------------------------------------------------
    input_box = (70, 285, 235, 370)
    node(draw, input_box, "Input", "3 x 256 x 256", "white", COLORS["arrow"])
    enc_specs = [
        ("d1", "32 x 128 x 128", "Conv4,s2 + LReLU"),
        ("d2", "128 x 64 x 64", "Conv4,s2 + BN + LReLU"),
        ("d3", "256 x 32 x 32", "Conv4,s2 + BN + LReLU"),
        ("d4", "256 x 16 x 16", "Conv4,s2 + BN + LReLU"),
        ("d5", "256 x 8 x 8", "Conv4,s2 + LReLU"),
    ]
    enc_boxes = []
    for index, (title, detail, note) in enumerate(enc_specs):
        rect = (300 + index * 205, 245, 465 + index * 205, 340)
        enc_boxes.append(rect)
        node(draw, rect, title, detail, COLORS["funie_fill"], COLORS["funie_line"], note)
    arrow(draw, (input_box[2], 327), (enc_boxes[0][0], 292), COLORS["funie_line"])
    for left, right in zip(enc_boxes[:-1], enc_boxes[1:]):
        arrow(draw, (left[2], 292), (right[0], 292), COLORS["funie_line"])

    dec_specs = [
        ("u1", "512 x 16 x 16", "ConvT4,s2 + BN + ReLU; concat d4"),
        ("u2", "512 x 32 x 32", "ConvT4,s2 + BN + ReLU; concat d3"),
        ("u3", "256 x 64 x 64", "ConvT4,s2 + BN + ReLU; concat d2"),
        ("u4", "64 x 128 x 128", "ConvT4,s2 + BN + ReLU; concat d1"),
    ]
    dec_boxes = []
    for index, (title, detail, note) in enumerate(dec_specs):
        rect = (950 + index * 205, 380, 1115 + index * 205, 465)
        dec_boxes.append(rect)
        node(draw, rect, title, detail, COLORS["funie_fill"], COLORS["funie_line"], note)
    arrow(draw, (enc_boxes[-1][2], 292), (dec_boxes[0][0], 422), COLORS["funie_line"])
    for left, right in zip(dec_boxes[:-1], dec_boxes[1:]):
        arrow(draw, (left[2], 422), (right[0], 422), COLORS["funie_line"])
    enhanced = (1900, 245, 2165, 340)
    node(draw, enhanced, "Enhanced image", "3 x 256 x 256", COLORS["output_fill"], COLORS["output_line"], "Upsample + ZeroPad + Conv4 + Tanh")
    arrow(draw, (dec_boxes[-1][2], 422), (enhanced[0], 292), COLORS["funie_line"])

    # U-Net skips are drawn as thin, dashed paths so they do not obscure flow.
    skip_pairs = [(enc_boxes[3], dec_boxes[0]), (enc_boxes[2], dec_boxes[1]),
                  (enc_boxes[1], dec_boxes[2]), (enc_boxes[0], dec_boxes[3])]
    for index, (source, target) in enumerate(skip_pairs):
        y = 350 + index * 13
        sx = (source[0] + source[2]) // 2
        tx = (target[0] + target[2]) // 2
        draw.line((sx, source[3], sx, y), fill=COLORS["funie_line"], width=2)
        draw.line((sx, y, tx, y), fill=COLORS["funie_line"], width=2)
        arrow(draw, (tx, y), (tx, target[1]), COLORS["funie_line"], 2, dashed=True)
    line_label(draw, (735, 165), "UNetUp returns concatenated tensors: u1=256+256, u2=256+256, u3=128+128, u4=32+32", COLORS["funie_line"])

    # ------------------------------------------------------------------
    # CrackFormer encoder and the actual feature source for each fusion.
    # ------------------------------------------------------------------
    centers = [300, 760, 1220, 1680, 2140]
    fusion_specs = [
        ("Fusion 1", "u4: 64 -> 64", "stage1"),
        ("Fusion 2", "u3: 256 -> 128", "stage2"),
        ("Fusion 3", "u2: 512 -> 256", "stage3"),
        ("Fusion 4", "u1: 512 -> 512", "stage4"),
        ("Fusion 5", "d5: 256 -> 512", "stage5"),
    ]
    stage_specs = [
        ("Stage 1", "64 x 128 x 128", "ConvRelu(3,64) + Trans_EB + MaxPool"),
        ("Stage 2", "128 x 64 x 64", "Trans_EB(64,128) + Trans_EB + MaxPool"),
        ("Stage 3", "256 x 32 x 32", "Trans_EB x 3 + MaxPool"),
        ("Stage 4", "512 x 16 x 16", "Trans_EB x 3 + MaxPool"),
        ("Stage 5", "512 x 8 x 8", "Trans_EB x 3 + MaxPool"),
    ]
    for index, ((ftitle, fdetail, _), (stitle, sdetail, snote), cx) in enumerate(zip(fusion_specs, stage_specs, centers)):
        fbox = (cx - 125, 620, cx + 125, 710)
        sbox = (cx - 145, 790, cx + 145, 895)
        node(draw, fbox, ftitle, fdetail, COLORS["fusion_fill"], COLORS["fusion_line"], "FeatureFusionBlock")
        node(draw, sbox, stitle, sdetail, COLORS["crack_fill"], COLORS["crack_line"], snote)
        arrow(draw, (cx, fbox[3]), (cx, sbox[1]), COLORS["fusion_line"], 3)
        # A small blue source port makes the exact mapping explicit.
        source_port = (cx - 72, 565, cx + 72, 603)
        source_name = fusion_specs[index][1].split(":")[0]
        node(draw, source_port, source_name, "FUnIE tap", COLORS["funie_fill"], COLORS["funie_line"], title_size=TINY)
        arrow(draw, (cx, source_port[3]), (cx, fbox[1]), COLORS["funie_line"], 2)
    # Crack encoder flow is Stage 1 -> Stage 5.
    for left, right in zip(centers, centers[1:]):
        arrow(draw, (left + 145, 842), (right - 145, 842), COLORS["crack_line"])
    arrow(draw, (enhanced[0] + 120, enhanced[3]), (centers[0] + 145, 842), COLORS["crack_line"])
    line_label(draw, (1300, 935), "Actual order: enhanced -> down1 -> Fusion1 -> down2 -> Fusion2 -> ... -> down5 -> Fusion5", COLORS["crack_line"])
    draw.text((65, 967), "FeatureFusionBlock: adapted = GELU(GroupNorm(1x1 Conv(FUnIE))); gate = sigmoid(1x1 Conv([crack, adapted])); fused = crack + tanh(strength) * gate * adapted", font=SMALL, fill=COLORS["fusion_line"])

    # ------------------------------------------------------------------
    # CrackFormer decoder and original Fuse/LA path.
    # ------------------------------------------------------------------
    up_centers = [350, 790, 1230, 1670, 2110]
    up_specs = [
        ("up5", "512 x 16 x 16", "MaxUnpool2d + Trans_EB x 3"),
        ("up4", "256 x 32 x 32", "MaxUnpool2d + Trans_EB x 3"),
        ("up3", "128 x 64 x 64", "MaxUnpool2d + Trans_EB x 3"),
        ("up2", "64 x 128 x 128", "MaxUnpool2d + Trans_EB x 2"),
        ("up1", "64 x 256 x 256", "MaxUnpool2d + Trans_EB x 2"),
    ]
    up_boxes = []
    for cx, (title, detail, note) in zip(up_centers, up_specs):
        rect = (cx - 135, 1110, cx + 135, 1205)
        up_boxes.append(rect)
        node(draw, rect, title, detail, COLORS["decoder_fill"], COLORS["decoder_line"], note)
    arrow(draw, (centers[-1], 895), (up_centers[0], 1110), COLORS["decoder_line"])
    for left, right in zip(up_boxes, up_boxes[1:]):
        arrow(draw, (left[2], 1157), (right[0], 1157), COLORS["decoder_line"])
    line_label(draw, (1220, 1070), "MaxUnpool uses saved MaxPool indices and unpool shapes", COLORS["decoder_line"])

    fuse_boxes = []
    for index, (cx, up) in enumerate(zip(up_centers, up_boxes)):
        rect = (cx - 130, 1270, cx + 130, 1350)
        fuse_boxes.append(rect)
        node(draw, rect, f"Fuse {5 - index}", "encoder + decoder -> 1 ch", COLORS["decoder_fill"], COLORS["decoder_line"], f"LA{5 - index} mask; scale {16 // (2 ** index)}")
        arrow(draw, ((up[0] + up[2]) // 2, up[3]), ((rect[0] + rect[2]) // 2, rect[1]), COLORS["decoder_line"], 2)
        # Dashed line denotes the encoder-side tensor used by Fuse, not the
        # already pooled stage output.
        arrow(draw, (centers[4 - index], 895), ((rect[0] + rect[2]) // 2, rect[1]), COLORS["crack_line"], 2, dashed=True)
    # All f1..f5 are resized to full resolution before final concatenation.
    concat = (2260, 1200, 2480, 1280)
    node(draw, concat, "Concat", "f5..f1: 5 x 256 x 256", COLORS["output_fill"], COLORS["output_line"])
    final = (2260, 1320, 2480, 1400)
    node(draw, final, "Final logits", "1 x 256 x 256", COLORS["output_fill"], COLORS["output_line"], "1x1 Conv (5 -> 1)")
    for fuse in fuse_boxes:
        arrow(draw, (fuse[2], (fuse[1] + fuse[3]) // 2), (concat[0], concat[1] + 40), COLORS["decoder_line"], 2)
    arrow(draw, ((concat[0] + concat[2]) // 2, concat[3]), ((final[0] + final[2]) // 2, final[1]), COLORS["output_line"])

    # ------------------------------------------------------------------
    # Outputs and training objective.
    # ------------------------------------------------------------------
    enhanced_out = (180, 1580, 680, 1670)
    seg_out = (870, 1580, 1370, 1670)
    total_out = (1600, 1580, 2100, 1670)
    node(draw, enhanced_out, "Enhancement target", "L1 + edge(enhanced, reference)", COLORS["loss_fill"], COLORS["loss_line"])
    node(draw, seg_out, "Segmentation target", "final + side outputs vs label", COLORS["loss_fill"], COLORS["loss_line"])
    node(draw, total_out, "Joint objective", "weighted sum -> backprop", COLORS["loss_fill"], COLORS["loss_line"])
    arrow(draw, (enhanced_out[2], 1625), (total_out[0], 1625), COLORS["loss_line"])
    arrow(draw, (seg_out[2], 1625), (total_out[0], 1625), COLORS["loss_line"])
    arrow(draw, (enhanced[0] + 50, enhanced[3]), (enhanced_out[0] + 180, enhanced_out[1]), COLORS["loss_line"], 2, dashed=True)
    arrow(draw, (final[0] + 70, final[3]), (seg_out[2] - 120, seg_out[1]), COLORS["loss_line"], 2, dashed=True)

    warning = (2180, 1530, 2525, 1710)
    draw.rounded_rectangle(warning, 10, fill=COLORS["warning_fill"], outline=COLORS["warning_line"], width=2)
    draw.text((2200, 1545), "Code note", font=LABEL, fill=COLORS["warning_line"])
    draw.text((2200, 1582), "Down3/4/5 define nn3,", font=SMALL, fill=COLORS["muted"])
    draw.text((2200, 1604), "but forward() calls nn2", font=SMALL, fill=COLORS["muted"])
    draw.text((2200, 1626), "for the third block.", font=SMALL, fill=COLORS["muted"])
    draw.text((2200, 1653), "Diagram follows execution.", font=SMALL, fill=COLORS["muted"])

    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    print(f"Saved framework diagram: {output.resolve()}")


def validate_forward(size=256, device="cpu"):
    """Optionally verify output sizes against the actual imported network."""
    import torch
    from nets import FunieCrackFusion

    model = FunieCrackFusion().to(device).eval()
    sample = torch.zeros(1, 3, size, size, device=device)
    with torch.no_grad():
        result = model(sample)
    print("Forward validation:")
    print("  enhanced:", tuple(result["enhanced"].shape))
    print("  logits:", tuple(result["logits"].shape))
    print("  side_outputs:", [tuple(x.shape) for x in result["side_outputs"]])


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=root / "outputs" / "fusion_framework_from_nets.png")
    parser.add_argument("--validate", action="store_true",
                        help="also run one forward pass through nets.FunieCrackFusion")
    parser.add_argument("--size", type=int, default=256,
                        help="validation input size; must be divisible by 32")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    args = parser.parse_args()
    render(args.output)
    if args.validate:
        validate_forward(args.size, args.device)


if __name__ == "__main__":
    main()