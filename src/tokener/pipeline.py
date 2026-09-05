"""Command-line version of the tokenizer comparison in ``notebooks/``.

Downloads the corpora, compares a general-purpose tokenizer (gpt2) with a
medical one (microsoft/biogpt) on medical and general text, then trains a
byte-level BPE tokenizer on PubMed abstracts and evaluates all three on
held-out medical text.

Usage:
    tokener                                  # full run (notebook-sized slices)
    tokener --n-train 1000 --vocab-size 10000  # smaller/faster run
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass

from datasets import load_dataset
from huggingface_hub import hf_hub_download, list_repo_files
from tokenizers import Tokenizer, pre_tokenizers
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer
from transformers import AutoTokenizer

GENERAL_TOKENIZER_ID = "openai-community/gpt2"
MEDICAL_TOKENIZER_ID = "microsoft/biogpt"
SCIENTIFIC_PAPERS_PARQUET_REV = "refs/convert/parquet"

DEMO_SENTENCE = (
    "The patient was prescribed amoxicillin-clavulanate for "
    "community-acquired pneumonia and referred for echocardiography."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--n-metrics", type=int, default=2000, help="docs per corpus for the 2x2 metrics")
    parser.add_argument("--n-train", type=int, default=5000, help="PubMed abstracts used to train the BPE tokenizer")
    parser.add_argument("--n-heldout", type=int, default=1500, help="held-out PubMed abstracts for final evaluation")
    parser.add_argument("--vocab-size", type=int, default=30000)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        default="medical_bpe_tokenizer.json",
        help="where to save the trained tokenizer",
    )
    return parser.parse_args()


def load_corpora(args: argparse.Namespace) -> dict[str, list[str]]:
    """PubMed abstracts (metrics/train/held-out slices) + WikiText-2 lines."""
    pubmed_files = [
        hf_hub_download("scientific_papers", f, repo_type="dataset", revision=SCIENTIFIC_PAPERS_PARQUET_REV)
        for f in sorted(
            list_repo_files("scientific_papers", repo_type="dataset", revision=SCIENTIFIC_PAPERS_PARQUET_REV)
        )
        if f.startswith("pubmed/train/") and f.endswith(".parquet")
    ]
    pubmed = load_dataset("parquet", data_files=pubmed_files, split="train")

    wikitext_file = hf_hub_download(
        "wikitext", "wikitext-2-raw-v1/train-00000-of-00001.parquet", repo_type="dataset"
    )
    wikitext = load_dataset("parquet", data_files=wikitext_file, split="train")

    shuffled = pubmed.shuffle(seed=args.seed)
    n = args.n_metrics
    return {
        "PubMed (metrics)": shuffled.select(range(n))["abstract"],
        "PubMed (train)": shuffled.select(range(n, n + args.n_train))["abstract"],
        "PubMed (held-out)": shuffled.select(
            range(n + args.n_train, n + args.n_train + args.n_heldout)
        )["abstract"],
        "WikiText-2 (general)": [t for t in wikitext["text"] if t.strip()][:n],
    }


def load_tokenizers() -> tuple[Tokenizer, AutoTokenizer]:
    """gpt2 via the raw `tokenizers` library; biogpt via transformers (legacy fairseq style)."""
    general = Tokenizer.from_pretrained(GENERAL_TOKENIZER_ID)
    medical = AutoTokenizer.from_pretrained(MEDICAL_TOKENIZER_ID)
    return general, medical


def token_list(tok: Tokenizer | AutoTokenizer, text: str) -> list[str]:
    """Tokens for `text` from either a raw `tokenizers.Tokenizer` or a transformers tokenizer."""
    if isinstance(tok, Tokenizer):
        return tok.encode(text, add_special_tokens=False).tokens
    return tok.tokenize(text)


@dataclass(frozen=True)
class CorpusStats:
    tokenizer: str
    corpus: str
    fertility: float
    tokens_per_1k_chars: float


def corpus_stats(tok: Tokenizer | AutoTokenizer, texts: list[str], tokenizer_name: str, corpus_name: str) -> CorpusStats:
    n_tokens = sum(len(token_list(tok, t)) for t in texts)
    n_words = sum(len(t.split()) for t in texts)
    n_chars = sum(len(t) for t in texts)
    return CorpusStats(
        tokenizer=tokenizer_name,
        corpus=corpus_name,
        fertility=n_tokens / n_words,
        tokens_per_1k_chars=1000.0 * n_tokens / n_chars,
    )


def format_table(rows: list[CorpusStats]) -> str:
    header = ("tokenizer", "corpus", "fertility (tokens/word)", "tokens/1k chars")
    body = [(r.tokenizer, r.corpus, f"{r.fertility:.3f}", f"{r.tokens_per_1k_chars:.1f}") for r in rows]
    widths = [max(len(h), *(len(b[i]) for b in body)) for i, h in enumerate(header)]

    def line(cols: tuple[str, ...]) -> str:
        return "  ".join(c.ljust(w) for c, w in zip(cols, widths))

    return "\n".join([line(header), line(tuple("-" * w for w in widths)), *(line(b) for b in body)])


def train_medical_bpe(texts: list[str], vocab_size: int, min_frequency: int) -> Tokenizer:
    """GPT-2-style byte-level BPE trained from scratch on the given texts."""
    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)  # match GPT-2's own setting
    tokenizer.decoder = ByteLevelDecoder()  # must pair with the ByteLevel pre-tokenizer
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),  # all 256 bytes -> lossless decoding
        special_tokens=["<|endoftext|>"],
        show_progress=False,
    )
    tokenizer.train_from_iterator(texts, trainer=trainer)
    return tokenizer


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    print("Loading corpora (first run downloads ~350 MB of parquet files)...")
    corpora = load_corpora(args)
    tok_general, tok_medical = load_tokenizers()

    print(f"\ngpt2 vocab size: {tok_general.get_vocab_size()} | biogpt vocab size: {tok_medical.vocab_size}")
    print("\nSame sentence, two tokenizers:")
    for name, tok in [("gpt2 (general)", tok_general), ("biogpt (medical)", tok_medical)]:
        toks = token_list(tok, DEMO_SENTENCE)
        print(f"  {name:17s} ({len(toks):2d} tokens): {toks}")

    print("\n2x2 comparison — crossover proves domain mismatch, not tokenizer quality:")
    rows = [
        corpus_stats(tok, texts, tname, cname)
        for cname, texts in [
            ("PubMed abstracts (medical)", corpora["PubMed (metrics)"]),
            ("WikiText-2 (general)", corpora["WikiText-2 (general)"]),
        ]
        for tname, tok in [("gpt2", tok_general), ("biogpt", tok_medical)]
    ]
    print(format_table(rows))

    print(f"\nTraining byte-level BPE on {args.n_train} PubMed abstracts (vocab_size={args.vocab_size})...")
    trained = train_medical_bpe(corpora["PubMed (train)"], args.vocab_size, args.min_frequency)
    round_trip = trained.decode(trained.encode(DEMO_SENTENCE, add_special_tokens=False).ids)
    print(f"Round-trip lossless: {round_trip == DEMO_SENTENCE}")
    trained.save(args.output)
    print(f"Saved trained tokenizer to {args.output}")

    print("\nHeld-out PubMed comparison (abstracts no tokenizer was trained on):")
    heldout_rows = [
        corpus_stats(tok, corpora["PubMed (held-out)"], name, "held-out PubMed")
        for name, tok in [
            ("gpt2 (general)", tok_general),
            ("biogpt (medical)", tok_medical),
            ("our trained BPE", trained),
        ]
    ]
    print(format_table(heldout_rows))


if __name__ == "__main__":
    main()
