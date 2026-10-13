"""Streamlit interface for private, batch photo captioning and renaming."""

from __future__ import annotations

import hashlib
import os

import streamlit as st

from photo_namer.captioning import (
    MAX_BATCH_BYTES,
    MAX_BATCH_IMAGES,
    ImageValidationError,
    caption_to_filename,
    load_captioner,
    validate_image_data,
)
from photo_namer.exporting import build_export_archive


st.set_page_config(
    page_title="Photo Namer AI",
    page_icon="📸",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_ID = os.getenv("PHOTO_NAMER_MODEL_ID", "Salesforce/blip-image-captioning-base")


@st.cache_resource(show_spinner=False)
def get_captioner(model_id: str):
    return load_captioner(model_id)


def _build_items(uploaded_files) -> tuple[list[dict], list[tuple[str, str]]]:
    items = []
    errors = []
    total_bytes = sum(file.size for file in uploaded_files)

    if len(uploaded_files) > MAX_BATCH_IMAGES:
        errors.append(
            ("Batch size", f"Select no more than {MAX_BATCH_IMAGES} images at a time.")
        )
        return items, errors
    if total_bytes > MAX_BATCH_BYTES:
        errors.append(("Batch size", "The total selected image size must be 40 MB or less."))
        return items, errors

    for index, uploaded_file in enumerate(uploaded_files):
        data = uploaded_file.getvalue()
        try:
            image, image_format = validate_image_data(data)
            digest = hashlib.sha256(data).hexdigest()
            items.append(
                {
                    "id": f"{digest}:{index}",
                    "filename": uploaded_file.name,
                    "data": data,
                    "image": image,
                    "image_format": image_format,
                }
            )
        except ImageValidationError as exc:
            errors.append((uploaded_file.name, str(exc)))

    return items, errors


st.markdown(
    """
    <style>
      .block-container {max-width: 1180px; padding-top: 2.2rem; padding-bottom: 3rem;}
      .hero {padding: 1.5rem 1.7rem; border: 1px solid rgba(120,130,150,.22);
             border-radius: 18px; background: linear-gradient(120deg, #f4f7ff, #f8fbff);}
      .hero p {font-size: 1.04rem; max-width: 760px; color: #526078;}
      @media (prefers-color-scheme: dark) {
        .hero {background: linear-gradient(120deg, #18233a, #182b32);}
        .hero p {color: #c5cfdf;}
      }
    </style>
    <div class="hero">
      <h1>Give every photo a meaningful name.</h1>
      <p>Generate clear captions and safe, descriptive filenames in batches.
      Inference runs locally on this machine using BLIP—no image API key required.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Workspace")
    st.markdown("**Model**  \nSalesforce BLIP · image-captioning-base")
    st.caption(f"Model identifier: `{MODEL_ID}`")
    st.divider()
    st.markdown("**Batch limits**")
    st.caption("Up to 20 images · 10 MB per image · 40 MB per batch")
    st.divider()
    st.markdown("**Privacy by design**")
    st.caption(
        "Images are processed in memory and are not written to disk. "
        "The model is downloaded from Hugging Face on first use; image pixels "
        "stay on this machine."
    )

st.subheader("1 · Choose photos")
st.write("Supported formats: JPEG, PNG, and WEBP. Use sample or non-sensitive images first.")
uploaded_files = st.file_uploader(
    "Add one or more images",
    type=("jpg", "jpeg", "png", "webp"),
    accept_multiple_files=True,
    help="Image contents are validated locally before inference.",
)

items, validation_errors = _build_items(uploaded_files or [])
if validation_errors:
    for filename, message in validation_errors:
        st.warning(f"{filename}: {message}")

if items:
    st.success(f"{len(items)} valid image{'s' if len(items) != 1 else ''} ready.")

if "caption_results" not in st.session_state:
    st.session_state.caption_results = {}
current_ids = {item["id"] for item in items}
st.session_state.caption_results = {
    key: value
    for key, value in st.session_state.caption_results.items()
    if key in current_ids
}

can_generate = bool(items) and not any(label == "Batch size" for label, _ in validation_errors)
if st.button("Generate captions and names", type="primary", disabled=not can_generate):
    try:
        with st.spinner("Loading the local model and captioning your photos…"):
            captioner = get_captioner(MODEL_ID)
            new_results = {}
            for item in items:
                try:
                    caption = captioner.caption(item["image"])
                    new_results[item["id"]] = {"caption": caption}
                except Exception as exc:  # Keep one bad inference from stopping a batch.
                    new_results[item["id"]] = {"error": type(exc).__name__}
            st.session_state.caption_results.update(new_results)
    except Exception as exc:
        st.error(
            "The local vision model could not be loaded. Check your model download "
            f"and runtime setup. ({type(exc).__name__})"
        )

results = st.session_state.caption_results
visible_results = [item for item in items if item["id"] in results]
if visible_results:
    st.divider()
    st.subheader("2 · Review and export")
    successful_count = sum(bool(results[item["id"]].get("caption")) for item in visible_results)
    st.caption(f"{successful_count} of {len(visible_results)} photos captioned successfully.")

    for item in visible_results:
        result = results[item["id"]]
        image_column, detail_column = st.columns([1, 2], gap="large")
        with image_column:
            st.image(item["image"], caption=item["filename"], use_container_width=True)
        with detail_column:
            with st.container(border=True):
                if result.get("caption"):
                    new_name = caption_to_filename(result["caption"], item["image_format"])
                    st.markdown(f"**Suggested filename**  \n`{new_name}`")
                    st.markdown(f"**AI caption**  \n{result['caption']}")
                else:
                    st.warning(
                        f"Caption generation failed ({result.get('error', 'unknown error')})."
                    )

    if successful_count:
        archive = build_export_archive(items, results)
        st.download_button(
            "Download renamed photos + captions.csv",
            data=archive,
            file_name="photo-namer-results.zip",
            mime="application/zip",
            type="primary",
        )
    if st.button("Clear results and start over"):
        st.session_state.caption_results = {}
        st.rerun()

st.caption(
    "AI captions can be incomplete or incorrect. Review suggested names before "
    "using them in a business workflow."
)