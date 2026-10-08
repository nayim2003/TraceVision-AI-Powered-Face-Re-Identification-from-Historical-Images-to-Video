import json
from pathlib import Path


class SessionWriter:
    """
    Persists TraceVision session metadata.
    """

    def __init__(
        self,
        output_dir: str = "data/processed/sessions",
    ):
        self.output_dir = Path(output_dir)

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    def write(self, session):
        """
        Write a session record to JSON.
        """

        filename = (
            f"{session.session_id}.json"
        )

        output_path = (
            self.output_dir / filename
        )

        with output_path.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                session.to_dict(),
                file,
                indent=4,
                ensure_ascii=False,
            )

        return output_path
