"""Offline tests for image validation and safe filename generation."""

import io
import csv
import unittest
import zipfile

from PIL import Image

from photo_namer.captioning import (
    ImageValidationError,
    caption_to_filename,
    unique_filenames,
    validate_image_data,
)
from photo_namer.exporting import build_export_archive


def image_bytes(image_format="PNG"):
    buffer = io.BytesIO()
    Image.new("RGB", (8, 6), color=(20, 80, 140)).save(buffer, format=image_format)
    return buffer.getvalue()


class ImageValidationTests(unittest.TestCase):
    def test_decodes_supported_image_and_reports_actual_format(self):
        decoded, image_format = validate_image_data(image_bytes("JPEG"))

        self.assertEqual(image_format, "JPEG")
        self.assertEqual(decoded.mode, "RGB")
        self.assertEqual(decoded.size, (8, 6))

    def test_rejects_empty_and_non_image_payloads(self):
        for payload in (b"", b"not an image"):
            with self.subTest(payload=payload):
                with self.assertRaises(ImageValidationError):
                    validate_image_data(payload)

    def test_rejects_unsupported_image_format(self):
        with self.assertRaisesRegex(ImageValidationError, "JPEG, PNG, or WEBP"):
            validate_image_data(image_bytes("BMP"))


class FilenameTests(unittest.TestCase):
    def test_filename_is_safe_and_uses_decoded_format(self):
        name = caption_to_filename("A cat / on a sunny day!", "JPEG")

        self.assertEqual(name, "a-cat-on-a-sunny-day.jpg")
        self.assertNotIn("/", name)
        self.assertNotIn("..", name)

    def test_filename_has_a_fallback_and_length_limit(self):
        self.assertEqual(caption_to_filename("你好", "PNG"), "meaningful-photo.png")
        self.assertLessEqual(len(caption_to_filename("very " * 100, "WEBP")), 77)

    def test_duplicate_names_get_stable_numeric_suffixes(self):
        self.assertEqual(
            unique_filenames(["sunset.jpg", "sunset.jpg", "SUNSET.JPG", "tree.png"]),
            ["sunset.jpg", "sunset-2.jpg", "SUNSET-3.jpg", "tree.png"],
        )

    def test_duplicate_paths_are_reduced_to_basename(self):
        self.assertEqual(unique_filenames(["../../trip.jpg"]), ["trip.jpg"])


class ExportTests(unittest.TestCase):
    def test_export_includes_safe_unique_names_and_manifest(self):
        items = [
            {"id": "one", "filename": "=formula.png", "data": b"one", "image_format": "PNG"},
            {"id": "two", "filename": "second.png", "data": b"two", "image_format": "PNG"},
            {"id": "failed", "filename": "failed.png", "data": b"bad", "image_format": "PNG"},
        ]
        results = {
            "one": {"caption": "A sunset"},
            "two": {"caption": "A sunset"},
            "failed": {"error": "RuntimeError"},
        }

        with zipfile.ZipFile(io.BytesIO(build_export_archive(items, results))) as archive:
            self.assertEqual(
                archive.namelist(),
                ["a-sunset.png", "a-sunset-2.png", "captions.csv"],
            )
            self.assertEqual(archive.read("a-sunset.png"), b"one")
            rows = list(
                csv.reader(io.StringIO(archive.read("captions.csv").decode("utf-8-sig")))
            )

        self.assertEqual(rows[1], ["'=formula.png", "a-sunset.png", "A sunset"])
        self.assertEqual(rows[2], ["second.png", "a-sunset-2.png", "A sunset"])


if __name__ == "__main__":
    unittest.main()