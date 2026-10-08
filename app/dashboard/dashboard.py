
import requests
import streamlit as st


# ==================================================
# Page Configuration
# ==================================================

st.set_page_config(
    page_title="TraceVision",
    page_icon="🔎",
    layout="wide",
)


# ==================================================
# Configuration
# ==================================================

API_BASE_URL = "http://127.0.0.1:8000"


# ==================================================
# API Helpers
# ==================================================

def get_track_results():
    """Fetch completed track results from the API."""

    response = requests.get(
        f"{API_BASE_URL}/results/tracks",
        timeout=30,
    )

    response.raise_for_status()

    return response.json()


def process_video(uploaded_file, frame_skip):
    """Upload and process a video through the FastAPI backend."""

    response = requests.post(
        f"{API_BASE_URL}/process/video",
        params={
            "frame_skip": frame_skip,
        },
        files={
            "file": (
                uploaded_file.name,
                uploaded_file.getvalue(),
                uploaded_file.type or "video/mp4",
            )
        },
        timeout=3600,
    )

    response.raise_for_status()

    return response.json()


# ==================================================
# Header
# ==================================================

st.title("TraceVision")

st.caption(
    "AI-Powered Cross-Temporal Face Re-Identification "
    "and Visual Retrieval System"
)

st.divider()


# ==================================================
# Video Processing
# ==================================================

st.subheader("Video Processing")

uploaded_file = st.file_uploader(
    "Upload a video",
    type=[
        "mp4",
        "avi",
        "mov",
        "mkv",
    ],
)

frame_skip = st.number_input(
    "Frame Skip",
    min_value=1,
    max_value=100,
    value=5,
    step=1,
    help="Process every Nth frame.",
)


if uploaded_file is not None:

    st.video(
        uploaded_file
    )

    if st.button(
        "🚀 Process Video",
        type="primary",
        use_container_width=True,
    ):

        with st.spinner(
            "Processing video... This may take some time."
        ):

            try:

                result = process_video(
                    uploaded_file=uploaded_file,
                    frame_skip=int(frame_skip),
                )

                st.session_state["last_processing_result"] = result

                st.success(
                    "Video processing completed successfully."
                )

            except requests.RequestException as exc:

                st.error(
                    "Unable to process the video."
                )

                st.code(str(exc))


# ==================================================
# Processing Summary
# ==================================================

processing_result = st.session_state.get(
    "last_processing_result"
)


if processing_result is not None:

    result_data = processing_result.get(
        "result",
        {},
    )

    st.subheader(
        "Processing Summary"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Frames",
            result_data.get(
                "total_frames",
                0,
            ),
        )

    with col2:
        st.metric(
            "Processed Frames",
            result_data.get(
                "processed_frames",
                0,
            ),
        )

    with col3:
        st.metric(
            "Finalized Tracks",
            result_data.get(
                "finalized_tracks",
                0,
            ),
        )

    with col4:
        st.metric(
            "Status",
            result_data.get(
                "status",
                "unknown",
            ),
        )


st.divider()


# ==================================================
# Load Track Results
# ==================================================

try:

    tracks = get_track_results()

except requests.RequestException as exc:

    st.error(
        "Unable to connect to the TraceVision API."
    )

    st.code(str(exc))

    st.stop()


# ==================================================
# Track Statistics
# ==================================================

total_tracks = len(tracks)

matched_tracks = sum(
    1
    for track in tracks
    if track.get("decision") == "matched"
)

unknown_tracks = sum(
    1
    for track in tracks
    if track.get("decision") == "unknown"
)


col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Total Tracks",
        total_tracks,
    )

with col2:
    st.metric(
        "Matched",
        matched_tracks,
    )

with col3:
    st.metric(
        "Unknown",
        unknown_tracks,
    )


st.divider()


# ==================================================
# Track Results
# ==================================================

st.subheader("Track Results")


if not tracks:

    st.info(
        "No completed track results are available."
    )

else:

    table_data = []

    for track in tracks:

        fusion_score = track.get(
            "fusion_score"
        )

        table_data.append(
            {
                "Track ID": track.get(
                    "track_id"
                ),
                "Decision": track.get(
                    "decision"
                ),
                "Subject": (
                    track.get("subject_id")
                    if track.get("subject_id")
                    else "—"
                ),
                "Fusion Score": (
                    round(
                        fusion_score,
                        4,
                    )
                    if fusion_score is not None
                    else "—"
                ),
                "Observations": track.get(
                    "observation_count"
                ),
                "Person Embeddings": track.get(
                    "person_embedding_count"
                ),
                "Face Embeddings": track.get(
                    "face_embedding_count"
                ),
            }
        )

    st.dataframe(
        table_data,
        use_container_width=True,
        hide_index=True,
    )


# ==================================================
# Track Details
# ==================================================

st.divider()

st.subheader("Track Details")


track_ids = [
    track.get("track_id")
    for track in tracks
]


if not track_ids:

    st.info(
        "No track is available for detailed inspection."
    )

else:

    selected_track_id = st.selectbox(
        "Select Track",
        track_ids,
    )

    selected_track = next(
        (
            track
            for track in tracks
            if track.get("track_id")
            == selected_track_id
        ),
        None,
    )

    if selected_track is not None:

        decision = selected_track.get(
            "decision"
        )

        subject_id = selected_track.get(
            "subject_id"
        )

        fusion_score = selected_track.get(
            "fusion_score"
        )

        best_candidate = selected_track.get(
            "best_candidate"
        )


        # ------------------------------------------
        # Decision Banner
        # ------------------------------------------

        if decision == "matched":

            st.success(
                f"✓ MATCHED — {subject_id}"
            )

        else:

            st.warning(
                "⚠ UNKNOWN — No fused identity match"
            )


        # ------------------------------------------
        # Primary Evidence
        # ------------------------------------------

        st.markdown(
            "### Final Decision"
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Decision",
                decision or "unknown",
            )

        with col2:

            st.metric(
                "Subject",
                subject_id or "—",
            )

        with col3:

            st.metric(
                "Fusion Score",
                (
                    f"{fusion_score:.4f}"
                    if fusion_score is not None
                    else "—"
                ),
            )


        # ------------------------------------------
        # Evidence Breakdown
        # ------------------------------------------

        if best_candidate is not None:

            st.markdown(
                "### Evidence Breakdown"
            )

            person_score = best_candidate.get(
                "person_score"
            )

            face_score = best_candidate.get(
                "face_score"
            )

            person_rank = best_candidate.get(
                "person_rank"
            )

            face_rank = best_candidate.get(
                "face_rank"
            )

            person_observations = (
                best_candidate.get(
                    "person_observations"
                )
            )

            face_observations = (
                best_candidate.get(
                    "face_observations"
                )
            )


            col1, col2 = st.columns(2)

            with col1:

                st.markdown(
                    "#### Person Re-ID"
                )

                st.metric(
                    "Person Score",
                    (
                        f"{person_score:.4f}"
                        if person_score is not None
                        else "—"
                    ),
                )

                st.metric(
                    "Person Rank",
                    (
                        person_rank
                        if person_rank is not None
                        else "—"
                    ),
                )

                st.metric(
                    "Observations",
                    (
                        person_observations
                        if person_observations is not None
                        else "—"
                    ),
                )


            with col2:

                st.markdown(
                    "#### Face Re-ID"
                )

                st.metric(
                    "Face Score",
                    (
                        f"{face_score:.4f}"
                        if face_score is not None
                        else "—"
                    ),
                )

                st.metric(
                    "Face Rank",
                    (
                        face_rank
                        if face_rank is not None
                        else "—"
                    ),
                )

                st.metric(
                    "Observations",
                    (
                        face_observations
                        if face_observations is not None
                        else "—"
                    ),
                )


        # ------------------------------------------
        # Track Statistics
        # ------------------------------------------

        st.markdown(
            "### Track Statistics"
        )

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.metric(
                "Observations",
                selected_track.get(
                    "observation_count",
                    0,
                ),
            )

        with col2:

            st.metric(
                "Person Embeddings",
                selected_track.get(
                    "person_embedding_count",
                    0,
                ),
            )

        with col3:

            st.metric(
                "Face Embeddings",
                selected_track.get(
                    "face_embedding_count",
                    0,
                ),
            )

        with col4:

            st.metric(
                "Track State",
                selected_track.get(
                    "state",
                    "unknown",
                ),
            )


        # ------------------------------------------
        # Raw Result
        # ------------------------------------------

        with st.expander(
            "View Raw Track Result"
        ):

            st.json(
                selected_track
            )

