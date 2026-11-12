"""MiniMax H3 prompt encoding, ported from ComfyUI comfy/text_encoders/minimax.py.

The H3 presentation is not chat-templated: token ids are the raw prompt text with no special tokens, and the
conditioning is the unnormalized hidden state after layer 50 (the checkpoint is truncated there and has no final
norm). With pictures each one comes first as "<Picture i>: " + a vision block, then the prompt. Every text segment
is tokenized on its own, as ComfyUI does.
"""

from backend.args import dynamic_args
from backend.text_processing import emphasis
from backend.text_processing._comfy import EMBEDDINGS, INF, SDClipModel, SDTokenizer

from ..profiles import PH3_IMAGE_HIDDEN_STATES, PICTURE_LABEL

PAD = 151643
VISION_START = 151652
VISION_END = 151653


def tap_layers(text_encoder):
    """What Forge's clip wrapper asks the encoder for: H3 takes the last hidden state, P-H3-Image's text connector
    the five it was trained on (the wrapper then returns the connector's output)."""
    return list(PH3_IMAGE_HIDDEN_STATES) if hasattr(text_encoder, "text_connector") else "last"


class VisionSpanClipModel(SDClipModel):
    # remembers where the vision embeddings of the last prompt landed, (start, length) per image
    image_spans: list[tuple[int, int]] = []

    def process_tokens(self, tokens, device):
        embeds, attention_mask, num_tokens, embeds_info = super().process_tokens(tokens, device)
        self.image_spans = [(e["index"], e["size"]) for e in embeds_info if e["type"] == "image"]
        return embeds, attention_mask, num_tokens, embeds_info


def vision_spans(image_spans) -> list[tuple[int, int]]:
    """Token ranges [start, stop) of the whole vision blocks, flanking <|vision_start|> / <|vision_end|> included;
    the DiT gives them the video modality tag (0) instead of the text one (1)."""
    return [(max(0, start - 1), start + size + 1) for start, size in image_spans]


class MiniMaxH3TextEngine:
    def __init__(self, text_encoder, tokenizer):
        self.text_encoder = VisionSpanClipModel(text_encoder, layer=tap_layers(text_encoder), special_tokens={"pad": PAD},
                                                layer_norm_hidden_state=False, enable_attention_masks=False,
                                                return_attention_masks=False)
        self.tokenizer = SDTokenizer(tokenizer, pad_with_end=False, has_start_token=False, has_end_token=False,
                                     pad_to_max_length=False, max_length=INF, min_length=1, pad_token=PAD)
        # vision block ranges of the last encoded prompt; the pictures come first, so the prompt and the negative
        # prompt share them
        self.vision_spans: list[tuple[int, int]] = []

    @property
    def emphasis(self) -> "emphasis.Emphasis":
        return emphasis.EmphasisNone()

    def tokenize(self, texts: str | list[str]) -> EMBEDDINGS | list[EMBEDDINGS]:
        return self.tokenizer.tokenizer(texts, add_special_tokens=False)["input_ids"]

    def __call__(self, texts: list[str], images: list = (), picture_label: str = PICTURE_LABEL) -> list:
        if any(emphasis.uses_emphasis(text) for text in texts):
            dynamic_args.last_extra_generation_params["Emphasis"] = "None"

        zs = []
        cache = {}
        self.vision_spans = []
        for line in texts:
            if line not in cache:
                cache[line] = self.text_encoder.encode_token_weights(self.tokens(line, images, picture_label))[0]
                self.vision_spans = vision_spans(self.text_encoder.image_spans)
            zs.extend(cache[line])  # (L, D) per prompt; the prompt parser stacks the batch
        return zs

    def tokens(self, text: str, images=(), picture_label: str = PICTURE_LABEL) -> list[list[tuple]]:
        """images: (1, h, w, 3) tensors in [0, 1], in <Picture i> order; picture_label names each one before its
        vision block ("<Picture %d>: " for H3 and P-H3-Image, "Picture %d: " for MiniMax-H3-Image)."""
        entries = []

        def add_text(segment):
            if not segment:
                return
            batches = self.tokenizer.tokenize_with_weights(segment, disable_weights=True)
            if len(batches) != 1:
                raise ValueError("[MiniMax H3] the prompt is longer than the text encoder accepts")
            entries.extend(batches[0])

        for i, image in enumerate(images):
            add_text(picture_label % (i + 1))
            embed = {"type": "image", "data": image, "original_type": "image"}
            entries.extend([(VISION_START, 1.0), (embed, 1.0), (VISION_END, 1.0)])
        add_text(text)
        return [entries or [(PAD, 1.0)]]
