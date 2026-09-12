# ANPR Autogate System

An Automatic Number Plate Recognition (ANPR) autogate system built for a single-gate Indonesian residential complex. It uses **YOLOv8** for vehicle detection and **PaddleOCR** for license plate reading, with a **PyQt5** desktop GUI for security guard operations and administration.

## What It Does

- Ingests a live video feed (webcam or IP camera) and detects vehicles in real-time using a YOLO model
- Reads license plates via OCR, normalizes the plate text to Indonesian format, and resolves vehicle direction (inbound / outbound)
- Checks plate numbers against a local resident database and automatically opens the gate for authorized vehicles
- Surfaces unresolved events (low-confidence reads, OCR failures, unknown direction) to a Guard Dashboard for manual review
- Provides an Admin Dashboard for managing the resident whitelist and viewing event history
- Stores detection events and captured images locally (SQLite + disk) — fully offline, no cloud dependency

## Architecture Overview

```
Camera ─► Detection (YOLO) ─► OCR (PaddleOCR) ─► Normalizer ─► Direction Resolver
                                                                        │
                                                          Access Controller
                                                                        │
                                                        ┌───────────────┼───────────────┐
                                                        ▼               ▼               ▼
                                                   Gate Control    Event Log DB    Guard Dashboard
```

All concrete implementations are selected via configuration (no code changes needed to switch between local-test and field environments).

## Requirements

- **Python 3.10+**
- **Windows / Linux** (tested on Windows)
- A webcam or IP camera for live detection (optional for testing — simulation mode available)

## Installation

1. **Clone the repo:**
   ```bash
   git clone https://github.com/<your-username>/autogate_project.git
   cd autogate_project
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv

   # Windows
   .venv\Scripts\activate

   # Linux / macOS
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -e .
   ```
   This installs all runtime dependencies (pinned versions):
   | Package | Version | Purpose |
   |---------|---------|---------|
   | ultralytics | 8.3.0 | YOLOv8 vehicle detection |
   | lapx | 0.9.4 | Object tracking (required by ultralytics) |
   | paddleocr | 2.9.1 | License plate OCR |
   | paddlepaddle | 2.6.2 | PaddlePaddle inference engine |
   | opencv-python | 4.10.0.84 | Image/video processing |
   | PyQt5 | 5.15.11 | Desktop GUI |
   | PyYAML | 6.0.2 | Configuration loading |

4. **Install test dependencies (optional):**
   ```bash
   pip install -e ".[test]"
   ```

## YOLO Model Weights

The system expects a `best.pt` weights file in the `anpr/` directory. This file is not included in the repo due to its size. Ask the project owner for the trained weights, or train your own using the Ultralytics YOLOv8 framework.

## Running

```bash
# Using the console script (after pip install -e .)
anpr

# Or run directly
python -m anpr.main
```

## Run Streamlit App

```bash
# work while inside or outside the env
cd streamlit_app
streamlit run app.py
```

The application launches the Guard Dashboard (live feed + gate control) and the Admin Dashboard (resident management + event log).

## Run API 

```bash
docker compose up -d
pip install -r requirements.txt
uvicorn api.app.main:app --reload --port 8000
```

swaggernya bisa dilihat di  http://localhost:8000/docs

## Configuration

Configuration lives in `anpr/config/default_config.yaml`. Key settings:

| Section | Setting | Description |
|---------|---------|-------------|
| `camera.type` | `webcam` / `ip` | Video source type |
| `camera.device_index` | `0` | Webcam index |
| `camera.target_fps` | `15` | Frame ingestion rate |
| `gate.mode` | `simulation` / `hardware` | Gate control mode |
| `model.detection_threshold` | `0.5` | YOLO confidence threshold |
| `ocr.confidence_threshold` | `0.70` | OCR confidence threshold |
| `database.location` | `./anpr.db` | SQLite database path |

Environment variables override config values using the `ANPR_` prefix with double-underscore separators:
```bash
ANPR_CAMERA__TARGET_FPS=30
ANPR_GATE__MODE=hardware
```

## Running Tests

```bash
pytest
```

## Manual Verification (MySQL)

The automated test suite runs entirely against a fake database, so it never touches a
live MySQL server. Use these ordered steps to verify real MySQL behavior end-to-end with
the Docker-hosted database and phpMyAdmin.

1. **Start MySQL 8.0 + phpMyAdmin:**
   ```bash
   docker compose up -d
   ```
   On first startup (empty data directory), the schema is created automatically from
   `docker/mysql/init/01-schema.sql` — the `residents`, `event_log`, and `images` tables.
   The application never creates or alters schema; it expects these tables to already
   exist.

2. **Seed a resident row in phpMyAdmin:**
   - Open phpMyAdmin at [http://localhost:8080](http://localhost:8080) and log in with
     user `anpr` / password `anprpassword` (or root `rootpassword`).
   - Select the `anpr` database and open the `residents` table.
   - Insert one row, setting `normalized_plate` to a plate you can present to the camera
     (in normalized Indonesian format, e.g. `B1234ABC`). Provide unique `id`, `created_at`,
     and `updated_at` values.

3. **Run the app against the seeded database:**
   ```bash
   anpr
   ```
   The default config connects to MySQL at `127.0.0.1:3306`, database `anpr`, user `anpr`,
   password `anprpassword`. Override the password without editing config via
   `ANPR_DATABASE__PASSWORD`. Startup refuses to continue if the database is unreachable or
   any required table is missing.

4. **Confirm the resulting `event_log` rows in phpMyAdmin:**
   Present plates to the camera and inspect the `event_log` table. For every handled
   detection the app appends exactly one flat row:

   | Detection | `classification` | `grant_method` | `direction` | `event_kind` | `entry_state` |
   |-----------|------------------|----------------|-------------|--------------|---------------|
   | Resident (gate opens) | `RESIDENT` | `AUTOMATIC` | NULL | NULL | `N/A` |
   | Guest (no match) | `GUEST` | `NONE` | NULL | NULL | `N/A` |

   In both cases `direction` and `event_kind` are `NULL` and `entry_state` is the `N/A`
   sentinel, confirming the direction-free match-and-log flow.

Stop the database with `docker compose down` (data persists in the named volume) or
`docker compose down -v` to delete all data.

## Project Structure

```
anpr/
├── config/          # Configuration loading and validation
├── core/            # Domain models, interfaces, access controller, normalizer
├── detection/       # YOLO vehicle detector wrapper
├── direction/       # Direction resolver (inbound/outbound)
├── gate/            # Gate controller (simulation + hardware)
├── imaging/         # Image capture and storage
├── ocr/             # PaddleOCR engine wrapper
├── persistence/     # SQLite database, resident and event log repositories
├── pipeline/        # Video source + detection pipeline orchestration
├── ui/              # PyQt5 Guard and Admin dashboards
└── main.py          # Composition root (entry point)
```

## License

Private project. All rights reserved.
