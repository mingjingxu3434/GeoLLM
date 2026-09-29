import hashlib
import re
import torch

_TOKEN_RE = re.compile(r"[A-Za-z0-9_%-]+|[^\s]", re.UNICODE)

def stable_hash_tokenize(text: str, vocab_size: int = 4096, max_len: int = 384) -> torch.Tensor:
    """Deterministic dependency-free fallback tokenizer.

    The manuscript does not release a tokenizer checkpoint. This function is intentionally
    reproducible across Python processes, unlike built-in hash(). Replace it with the exact
    tokenizer when an official language checkpoint is available.
    """
    words = _TOKEN_RE.findall(text.lower())[:max_len]
    ids = []
    for w in words:
        digest = hashlib.blake2b(w.encode('utf-8'), digest_size=8).digest()
        ids.append((int.from_bytes(digest, 'little') % (vocab_size - 1)) + 1)
    return torch.tensor(ids or [1], dtype=torch.long)
