# TraceVision

### AI-Powered Cross-Temporal Face Re-Identification & Visual Retrieval System

TraceVision is an end-to-end computer vision system designed to identify and retrieve the same person across different video frames and temporal observations.

The system combines **person detection and tracking, person re-identification, face re-identification, embedding memory, vector retrieval, candidate ranking, and multi-modal evidence fusion** into a unified pipeline. It is designed with an application-oriented architecture and can be exposed through a REST API and an interactive Streamlit dashboard.

---

## Key Features

* Person detection and multi-object tracking with YOLO and ByteTrack
* Person-level visual re-identification using OSNet
* Face detection and face re-identification using InsightFace / ArcFace
* Track-level embedding memory
* Quality-aware embedding storage
* Temporal face and person representation aggregation
* FAISS-based vector similarity retrieval
* Candidate history across multiple observations
* Candidate ranking based on similarity, consistency, recency, and observation frequency
* Multi-modal evidence fusion between person and face Re-ID
* Track finalization and identity decision
* Session and audit support
* REST API using FastAPI
* Interactive dashboard using Streamlit
* Automated unit and integration testing
* Real-video end-to-end runtime verification

---

## System Architecture

```text
                         ┌─────────────────────┐
                         │     Input Video     │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Person Detection &  │
                         │      Tracking       │
                         │   YOLO + ByteTrack  │
                         └──────────┬──────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
                    ▼                               ▼
          ┌──────────────────┐            ┌──────────────────┐
          │   Person Re-ID   │            │    Face Re-ID    │
          │      OSNet       │            │ InsightFace /    │
          │                  │            │     ArcFace      │
          └────────┬─────────┘            └────────┬─────────┘
                   │                               │
                   ▼                               ▼
          ┌──────────────────┐            ┌──────────────────┐
          │ Person Embedding │            │ Face Embedding   │
          │     Memory       │            │     Memory       │
          └────────┬─────────┘            └────────┬─────────┘
                   │                               │
                   ▼                               ▼
          ┌──────────────────┐            ┌──────────────────┐
          │ Track-level      │            │ Track-level      │
          │ Aggregation      │            │ Aggregation      │
          └────────┬─────────┘            └────────┬─────────┘
                   │                               │
                   └───────────────┬───────────────┘
                                   ▼
                         ┌─────────────────────┐
                         │   FAISS Retrieval   │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Candidate History   │
                         │    & Ranking        │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │  Evidence Fusion    │
                         │ Person + Face Re-ID │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Final Track Result  │
                         └──────────┬──────────┘
                                    │
                       ┌────────────┴────────────┐
                       ▼                         ▼
              ┌─────────────────┐       ┌─────────────────┐
              │   FastAPI REST  │       │ Streamlit       │
              │       API       │       │   Dashboard     │
              └─────────────────┘       └─────────────────┘
```

---

## Core Pipeline

For each video frame, TraceVision performs the following processing sequence:

```text
Video Frame
    ↓
Person Detection / Tracking
    ↓
Person Track Association
    ↓
Person Re-ID
    ↓
Face Detection
    ↓
Face ↔ Person Track Association
    ↓
Face Quality Assessment
    ↓
Face Re-ID
    ↓
Embedding Memory
    ↓
Track-level Embedding Aggregation
    ↓
FAISS Similarity Retrieval
    ↓
Candidate History
    ↓
Candidate Ranking
    ↓
Person + Face Evidence Fusion
    ↓
Final Identity Result
```

The system uses temporal observations rather than relying exclusively on a single frame. Multiple high-quality observations are accumulated in track memory before producing stronger retrieval and ranking evidence.

---

## Technology Stack

| Component             | Technology                 |
| --------------------- | -------------------------- |
| Programming Language  | Python                     |
| Person Detection      | Ultralytics YOLO           |
| Multi-Object Tracking | ByteTrack                  |
| Person Re-ID          | Torchreid / OSNet          |
| Face Detection        | InsightFace                |
| Face Recognition      | ArcFace                    |
| Vector Retrieval      | FAISS                      |
| Numerical Computing   | NumPy                      |
| Image Processing      | OpenCV                     |
| API                   | FastAPI                    |
| Dashboard             | Streamlit                  |
| Testing               | Pytest                     |
| Environment           | Python Virtual Environment |

---

## Project Structure

```text
Tracevision/
│
├── app/
│   ├── api/
│   │   ├── main.py
│   │   └── routes/
│   │
│   ├── dashboard/
│   │   └── dashboard.py
│   │
│   └── services/
│       └── tracevision_service.py
│
├── src/
│   ├── aggregation/
│   │   ├── candidate_history.py
│   │   ├── candidate_ranker.py
│   │   ├── evidence_fusion.py
│   │   └── face_embedding_aggregator.py
│   │
│   ├── pipeline/
│   │   ├── factory.py
│   │   ├── face_reid_pipeline.py
│   │   ├── person_reid_pipeline.py
│   │   └── tracevision_pipeline.py
│   │
│   ├── recognition/
│   │   ├── base_face_detector.py
│   │   ├── base_face_embedder.py
│   │   ├── face_crop.py
│   │   ├── face_memory_manager.py
│   │   ├── face_quality.py
│   │   └── face_track_associator.py
│   │
│   ├── retrieval/
│   │   └── face_retrieval_service.py
│   │
│   └── tracking/
│       └── models.py
│
├── tests/
│   ├── test_api.py
│   ├── test_pipeline_integration.py
│   ├── test_retrieval.py
│   ├── test_runtime_video.py
│   └── ...
│
├── scripts/
│   └── diagnose_runtime_face_embedding.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── reference_gallery/
│
├── models/
│
├── notebooks/
│
├── pytest.ini
├── requirements.txt
└── README.md
```

> The exact repository contents may evolve as the project moves from research and experimentation toward deployment.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/Tracevision.git
cd Tracevision
```

### 2. Create a virtual environment

Windows:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.venv\Scripts\Activate.ps1
```

Linux / macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running the REST API

Start the FastAPI application:

```powershell
uvicorn app.api.main:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

Health endpoint:

```text
http://127.0.0.1:8000/health
```

Interactive Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

---

## Running the Streamlit Dashboard

Open another terminal, activate the virtual environment, and run:

```powershell
streamlit run app/dashboard/dashboard.py
```

The dashboard provides an interactive interface for interacting with TraceVision results and system outputs.

---

## API Endpoints

| Method | Endpoint                                           | Purpose                     |
| ------ | -------------------------------------------------- | --------------------------- |
| GET    | `/`                                                | Service information         |
| GET    | `/health`                                          | Health check                |
| GET    | `/results/tracks`                                  | Retrieve track results      |
| GET    | `/results/tracks/{track_id}`                       | Retrieve a specific track   |
| GET    | `/results/tracks/{track_id}/face-debug`            | Face processing diagnostics |
| GET    | `/results/tracks/{track_id}/face-retrieval-debug`  | Face retrieval diagnostics  |
| GET    | `/results/tracks/{track_id}/face-embedding-debug`  | Face embedding diagnostics  |
| GET    | `/results/tracks/{track_id}/face-individual-debug` | Individual face diagnostics |
| POST   | `/process/video`                                   | Process a video             |

---

## Retrieval & Ranking

TraceVision does not rely on a single similarity score.

Multiple observations are accumulated for each tracked person and evaluated using:

* Mean similarity
* Latest similarity
* Similarity consistency
* Observation frequency
* Mean retrieval rank

The resulting candidate score is used to produce a more stable identity ranking across temporal observations.

---

## Multi-Modal Evidence Fusion

TraceVision combines two independent identity signals:

```text
Person Re-ID Evidence
        +
Face Re-ID Evidence
        ↓
Evidence Fusion
        ↓
Final Candidate Ranking
```

The current fusion configuration gives greater weight to face-based evidence:

```text
Person Re-ID: 40%
Face Re-ID:   60%
```

Candidates are fused only when supported by both modalities, allowing the system to use complementary visual evidence rather than relying on a single representation.

---

## Quality-Aware Processing

Low-quality face observations are filtered before embedding storage.

The pipeline evaluates factors such as:

* Face detection confidence
* Crop dimensions
* Image quality
* Blur
* Minimum quality thresholds

This prevents poor observations from unnecessarily contaminating track-level identity memory.

---

## Temporal Memory

Instead of treating each frame independently, TraceVision maintains embedding observations associated with individual tracks.

```text
Frame 1 ──→ Embedding ──┐
Frame 2 ──→ Embedding ──┤
Frame 3 ──→ Embedding ──┼──→ Track Memory
Frame 4 ──→ Embedding ──┤
Frame N ──→ Embedding ──┘
                         │
                         ▼
                 Aggregated Identity
                 Representation
```

This provides a more stable representation when individual frames contain blur, occlusion, pose variation, or other visual noise.

---

## Testing

Run the complete automated test suite:

```powershell
pytest -q
```

The project currently includes tests covering:

* FAISS retrieval
* Candidate ranking
* Evidence fusion
* Track finalization
* API endpoints
* Pipeline integration

### Real Video Runtime Test

TraceVision also includes a real-video smoke test:

```powershell
pytest tests/test_runtime_video.py -v -s
```

The runtime test executes the actual pipeline against a real video input and verifies end-to-end processing through pipeline finalization.

---

## Current Validation

The current implementation has been validated with:

```text
Automated tests:       18 passed
Real video smoke test: 100 frames
Runtime pipeline:      Passed
API test suite:        Passed
Pipeline integration:  Passed
```

The real-video validation covers the complete processing flow from video frames through tracking, Re-ID, retrieval, evidence fusion, and finalization.

---

## Design Principles

TraceVision is designed around several principles:

### Modular Architecture

Detection, tracking, embedding, retrieval, ranking, and fusion are implemented as separate components.

### Temporal Evidence

Identity decisions are strengthened through multiple observations instead of depending on a single frame.

### Quality-Aware Memory

Only sufficiently reliable observations are retained in track memory.

### Multi-Modal Retrieval

Person and face representations provide complementary identity evidence.

### Retrieval Before Decision

The system separates vector retrieval, candidate ranking, evidence fusion, and final identity finalization.

### Application-Oriented Design

The research pipeline is exposed through a REST API and an interactive dashboard, making the system suitable for further deployment and integration.

---

## Future Development

Potential future improvements include:

* GPU acceleration
* Larger reference galleries
* More advanced person Re-ID models
* Improved face quality estimation
* Adaptive temporal memory
* Advanced candidate calibration
* Persistent vector databases
* Multi-camera identity tracking
* Docker-based deployment
* Cloud deployment
* Authentication and API security
* Production monitoring and observability
* Larger-scale evaluation benchmarks

---

## Disclaimer

TraceVision is a computer vision research and engineering project intended for experimentation, evaluation, and development of visual retrieval and re-identification systems.

Face recognition and re-identification technologies can involve significant privacy, security, and ethical considerations. Any real-world deployment should comply with applicable laws, regulations, consent requirements, data protection policies, and organizational governance.

---

## Author

**Md. Nayim Howlader**

Data Analyst | Data Science & Machine Learning

Interested in:

* Data Science
* Machine Learning
* Computer Vision
* Machine Learning Engineering
* AI Systems
* Applied Research

---

## License

This project is currently intended for research and portfolio purposes.

A formal open-source license can be added before public redistribution or production use.

```
```
# **`Md. Nayim Howlader`**
## **`BSc (Honours), Statistics`,**
## **`Dhaka College, Dhaka`**
