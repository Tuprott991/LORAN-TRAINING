## Model Architecture

### LORAN: Multi-Vector Retrieval with Late Interaction

The LORAN model is a lightweight neural retrieval architecture inspired by ColBERT but optimized for efficiency and adaptability.

#### Architecture Components

**1. Lexical Encoder**
```
Input Text → Tokenization → Embedding Layer (d_lex_emb=512)
           ↓
   Layer Normalization
           ↓
   Multi-Head Self-Attention (8 heads)
      - Supports Flash Attention 2
      - SDPA (Scaled Dot-Product Attention)
      - Fallback to standard attention
           ↓
   Attention Pooling → Token Representations (d_lex=192)
```

**2. Low-Rank Projection**
```
Token Vectors (d_lex=192) → Low-Rank Factorization (rank=256)
                          ↓
                    U (rank × d_lex)  ×  V (rank × m_teacher)
                          ↓
                Final Embeddings (m_teacher=768)
                          ↓
                   L2 Normalization
```

**3. Late Interaction Scoring**

For single-vector mode (documents):
```
similarity(query, doc) = mean(query_vectors) · doc_vector
```

For multi-vector mode (queries):
```
similarity(query, doc) = mean_i(max_j(q_i · d_j))
```
Where:
- q_i are the top-K query token vectors (K=4)
- d_j are document token vectors (typically K=1 for efficiency)

#### Key Features

- **Multi-Vector Representation**: Queries use 4 token vectors for nuanced matching
- **Single-Vector Documents**: Documents compressed to 1 vector for fast retrieval
- **Flash Attention Support**: Optional Flash Attention 2 for 2-3x speedup
- **Gradient Checkpointing**: Enables training with larger batch sizes
- **Orthogonal Regularization**: Ensures diverse learned representations
- **Mixed Precision Training**: BF16/FP16 support for faster training

#### Model Hyperparameters

```yaml
d_lex_emb: 512      # Embedding dimension
d_lex: 192          # Lexical representation dimension
rank: 256           # Low-rank projection dimension
heads: 8            # Number of attention heads
topk_q: 4           # Query token vectors to keep
topk_d: 1           # Document token vectors to keep
m_teacher: 768      # Final embedding dimension
```

#### Loss Functions

The model is trained with a composite loss:

1. **Retrieval Loss (λ_ret=1.0)**: InfoNCE contrastive loss
2. **Lexical Loss (λ_lex=0.25)**: Term matching similarity
3. **Entropy Regularization (λ_ent=0.0015)**: Prevents attention collapse
4. **Orthogonality Loss (λ_ortho=0.001)**: Promotes diverse representations
5. **Knowledge Distillation (optional)**: Learns from teacher models like BGE-M3

## Training Pipeline

**make sure you pip install -r requirements.txt before training LORAN**

### Phase 1: Pre-training on MSMARCO v2.1

The model is pre-trained on the [MSMARCO v2.1](https://microsoft.github.io/msmarco/) passage ranking dataset, which contains millions of real web search queries paired with relevant passages.

#### Dataset Preparation

1. **Download the datasets:**
   - [Training data (MSMARCO)](https://drive.google.com/file/d/1ToLibb6URSbDVgxhkwJg_VE1fFSjY5J0/view?usp=drive_link)
   - [Test data (MSMARCO)](https://drive.google.com/file/d/12bTqwscrfzylAicm0Fz4SKKt1L_GGqR7/view?usp=drive_link)

2. **Place the files:**
data/
├── train.tsv
└── test.tsv


#### Training Command

Run pre-training with the provided configuration:

```bash
python train_longmatrix.py --config config/large_allmini.yaml
```

### Phase 2: Fine-tuning on Synthetic Memory Data

After pre-training, the model is fine-tuned on synthetically generated conversational memory data to improve retrieval of personal context and long-term interactions.

Step 1: Generate Synthetic Data
Generate 1,000 memory records (~5,000 query-memory pairs):

```bash
python synthetic_data_generation/data_generator.py --num_records 1000
```

Step 2: Convert to TSV Format
Convert the generated JSON data to TSV format:

```bash
python utils/data_converter.py --input path/to/queries.json
```

Step 3: Fine-tune the Model
Fine-tune using the memory-specific configuration and pre-trained checkpoint:

```bash
python run_longmatrix.py --config config/config_memories.yaml --resume longmatrix.pt
```

Training Tips
- Ensure sufficient GPU memory (40GB+ recommended for default batch sizes)
- Use mixed precision training (BF16/FP16) for faster training
- Monitor validation metrics to prevent overfitting on synthetic data
- Adjust learning rate in config files based on your hardware

