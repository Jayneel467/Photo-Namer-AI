# Photo Namer AI

A batch photo-captioning and filename suggestion app based on the IBM/Coursera
image-captioning lab. It uses Salesforce BLIP locally through Hugging Face
Transformers, then turns each caption into a portable, descriptive filename.

## Features

- Batch upload of JPEG, PNG, and WEBP images with size, format, and pixel limits.
- Local BLIP inference on CPU, Apple Silicon MPS, or CUDA where available.
- Safe filename generation, duplicate-name handling, and a CSV manifest.
- Downloadable ZIP containing renamed copies and `captions.csv`.
- No API key; uploaded images are processed in memory and are not written to disk.
- Offline unit tests for validation and naming behavior.
- Docker image runs as a non-root user and includes a health check.

## Project structure

```text
photo-namer-ai/
├── app.py                    # Streamlit UI and upload workflow
├── photo_namer/              # Reusable application logic
│   ├── captioning.py         # Validation, BLIP inference, safe filenames
│   └── exporting.py          # In-memory ZIP and CSV generation
├── tests/                    # Offline unit tests
├── .streamlit/config.toml    # Streamlit upload and security settings
├── Dockerfile                # Container deployment
├── requirements.txt          # Isolated runtime dependencies
└── README.md                 # Setup, operation, and security notes
```

## Run locally

Use Python 3.10 or newer. From this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

On first use, Transformers downloads `Salesforce/blip-image-captioning-base`
from Hugging Face and caches it under the current user's Hugging Face cache.
The first caption can take longer while the model initializes. CPU inference is
supported but may be slow for batches. Set `PHOTO_NAMER_MODEL_ID` to a compatible
Hugging Face BLIP checkpoint to use a different model.

## Docker

Build from this directory and start the container with port 8501 published. The
model cache can be mounted at `/home/app/.cache/huggingface` to avoid downloading
the model for every new container. Keep the container behind your organization's
TLS-terminating ingress or reverse proxy; do not expose an unauthenticated
Streamlit port to the public internet.

```bash
docker build -t photo-namer-ai .
docker run --rm --publish 8501:8501 --volume photo-namer-model-cache:/home/app/.cache/huggingface photo-namer-ai
```

## Verify

Run the offline test suite from this directory:

```bash
python -m unittest discover -s tests -v
```

## Operational and security notes

- This is a production-minded starting point, not a security or compliance
  certification. Before a company rollout, add organization-managed
  authentication, authorization, network controls, monitoring, a model review,
  and a deployment-specific threat assessment.
- Inference is local, but the model weights are fetched from Hugging Face on the
  first run. Pin and review model artifacts, their license, and their provenance
  for your organization's use case; mirror approved weights internally for
  restricted environments.
- Photos remain in application memory during the active Streamlit session and
  are included in the ZIP only when a user requests the download. Avoid using
  sensitive or regulated photos unless the deployment has been approved for
  that data. Configure session lifetime, TLS, ingress limits, and access policy
  in the hosting environment.
- The caption model can misidentify objects or omit context. Require human
  review before using captions as authoritative metadata.
- Streamlit's in-process model cache and lock are suitable for a single
  application process. For multi-worker/high-throughput use, move inference to
  an authenticated model-serving service with explicit capacity management.

## Configuration

| Setting | Default | Purpose |
| --- | --- | --- |
| `PHOTO_NAMER_MODEL_ID` | `Salesforce/blip-image-captioning-base` | Hugging Face model repository ID |
| Image size | 10 MB | Maximum size for each uploaded image |
| Batch size | 20 images / 40 MB | Limits per submission |
| Pixel count | 40 megapixels | Decompression-bomb and resource guard |

## License

The application code follows the parent repository's MIT license. BLIP model
weights and third-party libraries have separate licenses and terms; review them
before redistribution or commercial use.