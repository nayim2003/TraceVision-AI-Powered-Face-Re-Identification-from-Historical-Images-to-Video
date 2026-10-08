
import streamlit as st
import requests


API_URL = "http://127.0.0.1:8000"


st.set_page_config(
    page_title="TraceVision",
    page_icon="🔎",
    layout="wide",
)


# =========================================================
# HEADER
# =========================================================

st.title("TraceVision")
st.caption(
    "AI-Powered Cross-Temporal Face Re-Identification "
    "and Visual Retrieval System"
)

st.divider()


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("System")

if st.sidebar.button("Check API"):

    try:
        response = requests.get(
            f"{API_URL}/health",
            timeout=5,
        )

        if response.ok:
            data = response.json()

            st.sidebar.success(
                f"API: {data['status']}"
            )

        else:
            st.sidebar.error("API unavailable.")

    except requests.RequestException:
        st.sidebar.error("API unavailable.")


st.sidebar.divider()

st.sidebar.subheader("Configuration")

top_k = st.sidebar.slider(
    "Top-K Candidates",
    min_value=1,
    max_value=20,
    value=10,
)

similarity_threshold = st.sidebar.slider(
    "Similarity Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.70,
    step=0.05,
)


# =========================================================
# MAIN TABS
# =========================================================

tab1, tab2, tab3 = st.tabs(
    [
        "Dashboard",
        "Track Results",
        "System Information",
    ]
)


# =========================================================
# DASHBOARD
# =========================================================

with tab1:

    st.subheader("System Overview")

    col1, col2, col3, col4 = st.columns(4)

    try:

        response = requests.get(
            f"{API_URL}/results/tracks",
            timeout=5,
        )

        tracks = (
            response.json()
            if response.ok
            else []
        )

    except requests.RequestException:

        tracks = []

    finalized_tracks = len(tracks)

    candidate_tracks = sum(
        1
        for track in tracks
        if track.get("has_candidates", False)
    )

    total_person_embeddings = sum(
        track.get(
            "person_embedding_count",
            0,
        )
        for track in tracks
    )

    total_face_embeddings = sum(
        track.get(
            "face_embedding_count",
            0,
        )
        for track in tracks
    )

    col1.metric(
        "Finalized Tracks",
        finalized_tracks,
    )

    col2.metric(
        "Tracks with Candidates",
        candidate_tracks,
    )

    col3.metric(
        "Person Embeddings",
        total_person_embeddings,
    )

    col4.metric(
        "Face Embeddings",
        total_face_embeddings,
    )

    st.divider()

    st.subheader("Pipeline")

    st.code(
        """
Video
  ↓
Person Detection
  ↓
ByteTrack
  ↓
Person Re-ID
  ↓
Face Detection
  ↓
Face Re-ID
  ↓
Vector Retrieval
  ↓
Candidate Ranking
  ↓
Evidence Fusion
  ↓
Human Review
  ↓
Audit
        """,
        language="text",
    )


# =========================================================
# TRACK RESULTS
# =========================================================

with tab2:

    st.subheader("Track Results")

    if not tracks:

        st.info(
            "No finalized track results available."
        )

    else:

        for track in tracks:

            with st.expander(
                f"Track {track['track_id']} — "
                f"{track['state']}"
            ):

                c1, c2, c3, c4 = st.columns(4)

                c1.metric(
                    "Observations",
                    track["observation_count"],
                )

                c2.metric(
                    "Person Embeddings",
                    track["person_embedding_count"],
                )

                c3.metric(
                    "Face Embeddings",
                    track["face_embedding_count"],
                )

                c4.metric(
                    "Candidates",
                    "Yes"
                    if track["has_candidates"]
                    else "No",
                )

                if st.button(
                    "View Details",
                    key=f"track_{track['track_id']}",
                ):

                    try:

                        detail_response = requests.get(
                            f"{API_URL}/results/tracks/"
                            f"{track['track_id']}",
                            timeout=5,
                        )

                        if detail_response.ok:

                            st.json(
                                detail_response.json()
                            )

                        else:

                            st.error(
                                "Unable to load track details."
                            )

                    except requests.RequestException:

                        st.error(
                            "API unavailable."
                        )


# =========================================================
# SYSTEM INFORMATION
# =========================================================

with tab3:

    st.subheader("TraceVision System")

    st.write(
        {
            "Version": "0.1.0",
            "Detection": "YOLO",
            "Tracking": "ByteTrack",
            "Person Re-ID": "OSNet",
            "Face Re-ID": "ArcFace",
            "Vector Search": "FAISS",
            "Backend": "FastAPI",
            "Dashboard": "Streamlit",
            "Storage": "SQLite / PostgreSQL",
        }
    )

    st.divider()

    st.subheader("Configuration")

    st.write(
        {
            "Top-K": top_k,
            "Similarity Threshold": similarity_threshold,
        }
    )

    st.warning(
        "Candidate matches are retrieval/ranking results "
        "for authorized reference data and require human review. "
        "Similarity scores are not identity probabilities."
    )

