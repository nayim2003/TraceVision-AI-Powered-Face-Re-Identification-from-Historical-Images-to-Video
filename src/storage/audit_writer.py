
from pathlib import Path
import json
from datetime import datetime, timezone


class AuditWriter:
    """
    Writes finalized TraceVision track results
    as session-specific JSON audit records.
    """

    def __init__(self, output_dir: str = "data/processed/audit"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write(self, final_result, session_id: str = "default"):
        session_dir = self.output_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        result = final_result.to_dict()

        audit_record = {
            "session_id": session_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "record_type": "track_final_result",
            "result": result,
        }

        filename = f"track_{final_result.track_id}.json"
        output_path = session_dir / filename

        with output_path.open("w", encoding="utf-8") as file:
            json.dump(
                audit_record,
                file,
                indent=4,
                ensure_ascii=False,
            )

        return output_path
