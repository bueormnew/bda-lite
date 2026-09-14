from collections.abc import Iterable
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers

SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]


def train_bpe(texts: Iterable[str], output: str | Path, vocab_size: int = 8192) -> Tokenizer:
    """Train a compact text BPE; it is not a byte-level tokenizer."""
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.normalizer = normalizers.Sequence([normalizers.NFD(), normalizers.Lowercase(), normalizers.StripAccents()])
    tokenizer.pre_tokenizer = pre_tokenizers.Metaspace()
    tokenizer.decoder = decoders.Metaspace()
    tokenizer.train_from_iterator(texts, trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=2, special_tokens=SPECIAL_TOKENS))
    tokenizer.save(str(output))
    return tokenizer
