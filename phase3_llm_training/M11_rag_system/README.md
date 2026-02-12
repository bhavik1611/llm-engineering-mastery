# M11: RAG System

## Problem Statement

Build a Retrieval-Augmented Generation (RAG) system that retrieves relevant documents from a knowledge base and conditions an LLM on them to answer questions. Implement indexing, retrieval (e.g., dense retrieval), and generation. Evaluate on a question-answering task.

## Mathematical Formulation

### Embedding and Similarity
(Placeholder for document and query embeddings, similarity scoring)

### Retrieval
(Placeholder for top-k selection, reranking if used)

### Prompt Construction
(Placeholder for context format, concatenation with question)

## Intuition

(Placeholder for intuition on why RAG reduces hallucination, when retrieval helps vs hurts, and the role of chunk size)

## Implementation Plan

1. Choose or create a knowledge base (e.g., documents, Wikipedia subsets)
2. Chunk documents (fixed size or semantic)
3. Embed chunks using a sentence/document encoder
4. Build an index (in-memory or vector DB)
5. For a query: embed query, retrieve top-k chunks, rank
6. Construct prompt: context + question
7. Generate answer using LLM
8. Evaluate: accuracy, relevance, or manual assessment
9. (Optional) Add reranking step

(No code.)

## Required Experiments

- Build RAG pipeline and run on QA dataset
- Report retrieval accuracy: is the relevant chunk in top-k?
- Report end-to-end QA accuracy or F1
- Ablate: top-k (1, 3, 5, 10), chunk size, or number of chunks in context

## Required Visualizations

- Pipeline diagram: index, retrieve, prompt, generate
- Table: retrieval accuracy and QA accuracy vs top-k
- (Optional) Example queries with retrieved context and generated answer

## Reflection Questions

1. How does chunk size affect retrieval quality and context relevance?
2. When does adding more retrieved chunks hurt rather than help?
3. What are the failure modes of RAG (e.g., retrieval misses, irrelevant context)?

## Completion Checklist

- [ ] RAG pipeline implemented end-to-end
- [ ] Retrieval and QA evaluation documented in `experiments_log.md`
- [ ] Ablation (top-k or chunk size) documented
- [ ] Pipeline diagram in `visualizations/`
- [ ] Journal entry in `learning_journal.md`
