# M8: Tokenizer from Scratch

## Problem Statement

Implement a tokenizer (Byte Pair Encoding or similar subword algorithm) from scratch. Train it on a corpus to build a vocabulary and merge table. Support encode (text to token ids) and decode (token ids to text) with round-trip correctness.

## Mathematical Formulation

### BPE Algorithm
(Placeholder for merge rule selection based on frequency)

### Vocabulary Construction
(Placeholder for initial vocabulary and merge iterations)

### Encoding/Decoding
(Placeholder for greedy segmentation and lookup)

## Intuition

(Placeholder for intuition on subword vs character vs word tokenization, and why BPE balances vocabulary size and sequence length)

## Implementation Plan

1. Initialize vocabulary from character-level (or UTF-8 bytes)
2. Compute pair frequencies across corpus
3. Iteratively merge most frequent pair; add to vocabulary
4. Repeat until target vocabulary size or merge count
5. Implement encode: greedily apply merge rules to segment text
6. Implement decode: join tokens, handle special tokens
7. Verify round-trip: decode(encode(text)) == text for sample inputs
8. Document vocabulary size, merge count, and sample tokenizations

(No code.)

## Required Experiments

- Train tokenizer on a representative corpus
- Report vocabulary size and number of merges
- Test encode/decode on diverse inputs: short, long, with punctuation, with numbers
- Compare sequence length: character-level vs your tokenizer vs word-level (if applicable)

## Required Visualizations

- Table: sample strings and their tokenization (tokens and ids)
- (Optional) Merge tree or visualization of top merges
- (Optional) Distribution of token lengths or sequence length reduction

## Reflection Questions

1. Why does BPE often produce better results than word-level tokenization for rare words?
2. How does vocabulary size affect downstream model size and sequence length?
3. What are the tradeoffs of training a custom tokenizer vs using a pretrained one?

## Completion Checklist

- [ ] Tokenizer trains and produces vocabulary
- [ ] Encode and decode implemented with round-trip correctness
- [ ] Sample tokenizations documented in `experiments_log.md`
- [ ] Journal entry in `learning_journal.md`
