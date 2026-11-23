"""Build a tiny, randomly initialised chat model + tokenizer entirely offline.

Used by tests and the CI smoke-training job: it exercises the real code path (chat template,
masking, PEFT, Trainer, saving, merging, generation) in seconds on a CPU, without downloading a
multi-GB checkpoint. It learns nothing useful - it only proves the pipeline runs end to end.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

CHATML_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)
SPECIALS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]


def build_tiny_model(
    out_dir: str | Path,
    corpus: Iterable[str],
    *,
    vocab_size: int = 600,
    hidden_size: int = 64,
    layers: int = 2,
    seed: int = 0,
) -> Path:
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    raw = Tokenizer(models.BPE(unk_token=None))
    raw.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    raw.decoder = __import__("tokenizers").decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIALS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    raw.train_from_iterator(list(corpus), trainer=trainer)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=raw,
        eos_token="<|im_end|>",
        pad_token="<|endoftext|>",
        bos_token=None,
        unk_token=None,
    )
    tokenizer.chat_template = CHATML_TEMPLATE
    tokenizer.save_pretrained(out)

    torch.manual_seed(seed)
    config = Qwen2Config(
        vocab_size=len(tokenizer),
        hidden_size=hidden_size,
        intermediate_size=hidden_size * 2,
        num_hidden_layers=layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=4096,
        tie_word_embeddings=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    Qwen2ForCausalLM(config).save_pretrained(out)
    return out
