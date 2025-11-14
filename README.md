### AI Dataset Verifier: Pre-Encryption Quality Assessment Engine

Overview: Within our decentralized marketplace platform, datasets submitted for listing as NFTs undergo rigorous pre-minting validation through an integrated AI-driven verification process. The core implementation, `main.py`, is a high-performance FastAPI-based microservice designed to analyze uploaded datasets—either as ZIP archives or individual files—evaluating them across key dimensions: quality, completeness, consistency, and relevance. This mechanism identifies and mitigates data quality issues, ensuring that only validated, high-integrity datasets proceed to blockchain inscription. Developed with hackathon efficiency in mind, the system employs BERT for text perplexity analysis, ResNet-50 for image classification confidence, and SentenceTransformer for semantic alignment, all containerized for straightforward integration into the backend architecture.

#### Operational Workflow: From Submission to Validation Outcome (Sub-10-Second Processing)
1. Data Ingestion and Parsing  
   - Supports ZIP archives and standalone files in formats including JSON, CSV, Parquet, Excel, TXT, PNG, and JPEG.  
   - Extracts textual content via recursive JSON traversal and tabular data flattening, while preprocessing images using PIL for compatibility.  
   - Manages exceptions efficiently: Unsupported formats are immediately flagged with an "unsupported" issue to maintain workflow integrity.

2. Anomaly Detection and Issue Identification  
   - Tabular Data Analysis: Detects columns exceeding 30% missing values, row duplication rates above 10%, or numerical outliers beyond 3 standard deviations (Z-score > 3σ).  
   - Textual Content Validation: Identifies empty or nonsensical entries (alphanumeric content <10%), excessively brief segments (<1 character), or overly verbose passages (>500 characters).  
   - Image Quality Checks: Rejects undersized images (<50 pixels on the minor dimension), predominantly dark compositions (>95% black pixels), or corrupted files.  
   - Results are compiled into a structured `Issue` array, e.g., `{file: "data.csv", type: "missing", details: "Column 'price' exhibits 45% NaN values"}`.

3. Multi-Modal Scoring Framework  
   - Text Processing Pipeline (BERT-Masked Language Model): Evaluates perplexity on up to five representative snippets (truncated to 512 tokens), deriving quality as `100 - (average perplexity × 2)`. Completeness is scaled by sample volume, and consistency penalizes variance in perplexity scores.  
   - Image Processing Pipeline (ResNet-50 on ImageNet): Computes top-5 softmax confidence across up to three images, with quality equated to `average confidence × 100`. Completeness adjusts based on image count, maintaining consistency via variance assessment.  
   - Relevance Assessment (SentenceTransformer): Calculates cosine similarity between the dataset description and content embeddings, normalized from [-1, 1] to [0, 100]. Implementation Note: This component dynamically aligns scores with the submitter's description, enhancing topical fidelity.  
   - Score Aggregation: Applies a weighted fusion (60% text, 40% image) for hybrid datasets, defaulting to the primary modality where applicable.

4. Outcome Generation and API Response  
   - Consolidates results into a `VerifyResponse` object: `{scores: {quality: 85, ...}, status: "VERIFIED" (if quality ≥ 50) or "FAILED", issues: [...]}`.  
   - Includes a root-level health endpoint (`/`) for container orchestration monitoring; compatible with CUDA acceleration or CPU fallback to support varied deployment environments.

#### Technical Architecture and Optimization Strategies
- Core Dependencies: FastAPI for API orchestration; PyTorch and Hugging Face Transformers for machine learning inference; Pandas, PyArrow, and OpenPyXL for data manipulation; SentenceTransformers for embedding generation.  
- Performance Enhancements: Sample limitations (e.g., five text excerpts, three images) ensure inference latency below 10 seconds; utilizes model evaluation mode and gradient disabling (`no_grad()`) for resource efficiency.  
- Integration Interface: Accessible via `POST /verify` endpoint accepting multipart uploads with optional description metadata. Operates pre-encryption to preserve data security, deferring key management to upstream backend processes.  
- Demonstration Scenario: Submitting a flawed ZIP (e.g., duplicated CSVs paired with low-resolution PNGs) yields escalated issues and depressed scores. In contrast, a well-curated multimodal dataset achieves "VERIFIED" status with scores exceeding 90, qualifying it for NFT minting.

This verification engine serves as the foundational trust mechanism for our NFT marketplace, transforming unrefined datasets into quantifiable, blockchain-ready assets. In the Web3 ecosystem, such quality assurance elevates data as a premium commodity. We invite collaboration: clone the repository, deploy via containerization, and iterate toward broader adoption.