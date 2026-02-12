# Phase 3: LLM Training

## What This Phase Is About

Phase 3 moves from toy models to real LLM training pipelines. You will implement a tokenizer (BPE or similar), pretrain a small GPT from scratch, fine-tune it with LoRA, and build a retrieval-augmented generation (RAG) system. The focus is on production-relevant techniques and tooling.

## Mathematical Focus

- Byte Pair Encoding: merge rules, vocabulary construction
- Pretraining objective: next-token prediction, perplexity
- LoRA: low-rank decomposition of weight updates
- RAG: retrieval scoring, context concatenation, generation with external knowledge

## Engineering Focus

- Tokenizer implementation: encode, decode, merge table
- Training at scale: batching, checkpointing, logging
- Parameter-efficient fine-tuning: LoRA configuration, merging
- RAG pipeline: embedding, retrieval, prompt construction

## Expected Competencies After Completion

- Build and train a tokenizer from a corpus
- Pretrain a small GPT and interpret perplexity
- Apply LoRA for fine-tuning and merge weights for inference
- Construct a RAG system with retrieval and generation stages

## Milestones in This Phase

| Milestone | Description |
|-----------|-------------|
| M8 | Tokenizer from scratch |
| M9 | Small GPT Pretraining |
| M10 | LoRA Fine-Tuning |
| M11 | RAG System |

## Exit Criteria

You have completed Phase 3 when:

- [ ] M8: Tokenizer encodes/decodes; vocabulary and merge rules documented
- [ ] M9: Small GPT pretrained; perplexity and samples reported
- [ ] M10: LoRA fine-tuning on downstream task; comparison to full fine-tune documented
- [ ] M11: RAG pipeline operational; evaluation on QA task documented
