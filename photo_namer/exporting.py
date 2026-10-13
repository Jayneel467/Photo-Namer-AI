"""In-memory export of renamed image copies and a CSV caption manifest."""

import csv
import io
import zipfile

from photo_namer.captioning import caption_to_filename, unique_filenames


def _safe_csv_value(value: str) -> str:
    """Prevent values from being interpreted as spreadsheet formulas."""
    if value.lstrip().startswith(("=", "+", "-", "@")):
        return "'" + value
    return value


def build_export_archive(items: list[dict], results: dict[str, dict]) -> bytes:
    """Build a ZIP of successful outputs without writing images to disk."""
    buffer = io.BytesIO()
    successful = [item for item in items if results.get(item["id"], {}).get("caption")]
    output_names = unique_filenames(
        caption_to_filename(
            results[item["id"]]["caption"], item["image_format"]
        )
        for item in successful
    )

    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        manifest = io.StringIO(newline="")
        writer = csv.writer(manifest)
        writer.writerow(("original_filename", "new_filename", "caption"))
        for item, output_name in zip(successful, output_names):
            caption = results[item["id"]]["caption"]
            archive.writestr(output_name, item["data"])
            writer.writerow(
                (
                    _safe_csv_value(item["filename"]),
                    output_name,
                    _safe_csv_value(caption),
                )
            )
        archive.writestr("captions.csv", manifest.getvalue().encode("utf-8-sig"))

    return buffer.getvalue()