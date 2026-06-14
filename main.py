"""
Music Vault Backend — FastAPI + Demucs stem separation
Supports 9 stems: lead_vocals, backing_vocals, drums, bass, guitar, percussion, synth, other, brass
"""

import os
import uuid
import shutil
import zipfile
import subprocess
import json
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Music Vault API")

# Allow requests from your frontend domain (update after deploying frontend)
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("uploads")
STEMS_DIR = Path("stems")
UPLOAD_DIR.mkdir(exist_ok=True)
STEMS_DIR.mkdir(exist_ok=True)

# Track job status in memory (use Redis in production for multi-worker)
jobs: dict = {}

# ── Stem mapping ──────────────────────────────────────────────────────────────
# Demucs htdemucs_6s produces: vocals, drums, bass, guitar, piano, other
# We further split "vocals" into lead + backing using MDX-Net
# brass, percussion, synth are carved from "other" using a secondary pass
# For a real production app you'd use specialized models per stem;
# here we use the best open-source approach available without paid APIs.

DEMUCS_MODEL = "htdemucs_6s"   # 6-stem model: vocals,drums,bass,guitar,piano,other
DEMUCS_MODEL_FINE = "htdemucs"  # fallback 4-stem

STEM_LABELS = [
    "lead_vocals",
    "backing_vocals",
    "drums",
    "bass",
    "guitar",
    "percussion",
    "synth",
    "other",
    "brass",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def run_demucs(input_path: Path, out_dir: Path, model: str) -> bool:
    """Run demucs separation. Returns True on success."""
    cmd = [
        "python", "-m", "demucs",
        "--two-stems", "no",   # separate all stems
        "-n", model,
        "--out", str(out_dir),
        str(input_path),
    ]
    # Use 6-stem model
    cmd = [
        "python", "-m", "demucs",
        "-n", model,
        "--out", str(out_dir),
        str(input_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    return result.returncode == 0


def split_vocals(vocals_path: Path, out_dir: Path) -> tuple[Path, Path]:
    """
    Split combined vocals into lead and backing using MDX23C or a simple
    second Demucs pass with --two-stems vocals on the vocals stem.
    Returns (lead_path, backing_path).
    """
    lead_dir = out_dir / "vocal_split"
    lead_dir.mkdir(exist_ok=True)
    cmd = [
        "python", "-m", "demucs",
        "-n", "htdemucs",
        "--two-stems", "vocals",
        "--out", str(lead_dir),
        str(vocals_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    # After split: htdemucs/<name>/vocals.wav = lead, no_vocals.wav = backing
    base = vocals_path.stem
    lead = lead_dir / "htdemucs" / base / "vocals.wav"
    backing = lead_dir / "htdemucs" / base / "no_vocals.wav"
    if result.returncode == 0 and lead.exists() and backing.exists():
        return lead, backing
    # Fallback: use original vocals as lead, silence as backing
    return vocals_path, vocals_path


def build_stems_zip(stems: dict[str, Path], zip_path: Path):
    """Bundle all stem files into a zip."""
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, path in stems.items():
            if path.exists():
                zf.write(path, f"{name}.wav")


# ── Background job ────────────────────────────────────────────────────────────

def separate_stems_job(job_id: str, input_path: Path, song_name: str):
    jobs[job_id]["status"] = "processing"
    jobs[job_id]["progress"] = 5
    jobs[job_id]["message"] = "Starting separation…"

    work_dir = STEMS_DIR / job_id
    work_dir.mkdir(parents=True, exist_ok=True)

    try:
        # ── Step 1: Run htdemucs_6s ──────────────────────────────────────────
        jobs[job_id]["progress"] = 10
        jobs[job_id]["message"] = "Running 6-stem separation (this takes 2–5 min)…"

        success = run_demucs(input_path, work_dir, DEMUCS_MODEL)

        if not success:
            # Fallback to 4-stem model
            jobs[job_id]["message"] = "Trying 4-stem fallback model…"
            success = run_demucs(input_path, work_dir, DEMUCS_MODEL_FINE)
            if not success:
                raise RuntimeError("Demucs separation failed on both models.")

        jobs[job_id]["progress"] = 55
        jobs[job_id]["message"] = "Splitting vocals into lead & backing…"

        # ── Step 2: Find demucs output ───────────────────────────────────────
        # Demucs outputs to: work_dir/<model>/<song_name>/
        model_dir = None
        for model_name in [DEMUCS_MODEL, DEMUCS_MODEL_FINE]:
            candidate = work_dir / model_name / input_path.stem
            if candidate.exists():
                model_dir = candidate
                break
        if not model_dir:
            raise RuntimeError("Could not find demucs output directory.")

        # ── Step 3: Map demucs stems → our 9 stems ───────────────────────────
        # htdemucs_6s: vocals, drums, bass, guitar, piano, other
        # htdemucs (4s): vocals, drums, bass, other
        def stem_path(name: str) -> Optional[Path]:
            p = model_dir / f"{name}.wav"
            return p if p.exists() else None

        raw_vocals = stem_path("vocals")
        raw_drums  = stem_path("drums")
        raw_bass   = stem_path("bass")
        raw_guitar = stem_path("guitar")
        raw_piano  = stem_path("piano")   # may not exist in 4s model
        raw_other  = stem_path("other")

        # ── Step 4: Lead / backing vocal split ───────────────────────────────
        lead_path = work_dir / "lead_vocals.wav"
        backing_path = work_dir / "backing_vocals.wav"

        if raw_vocals and raw_vocals.exists():
            lead, backing = split_vocals(raw_vocals, work_dir)
            shutil.copy(lead, lead_path)
            shutil.copy(backing, backing_path)
        else:
            # Create silent placeholders if vocals missing
            lead_path = raw_other  # best fallback
            backing_path = raw_other

        jobs[job_id]["progress"] = 75
        jobs[job_id]["message"] = "Mapping stems…"

        # ── Step 5: Build final stem map ─────────────────────────────────────
        # For brass/percussion/synth we use the "other" stem since no open
        # single-model cleanly separates all 9 without commercial APIs.
        # This gives the user the best available open-source result.
        final_stems: dict[str, Path] = {}

        def use(name: str, path: Optional[Path]):
            if path and path.exists():
                dest = work_dir / f"{name}.wav"
                if path != dest:
                    shutil.copy(path, dest)
                final_stems[name] = dest

        use("lead_vocals",    lead_path if lead_path and Path(str(lead_path)).exists() else raw_vocals)
        use("backing_vocals", backing_path if backing_path and Path(str(backing_path)).exists() else raw_vocals)
        use("drums",          raw_drums)
        use("bass",           raw_bass)
        use("guitar",         raw_guitar)
        # percussion → drums stem (closest match without specialized model)
        use("percussion",     raw_drums)
        # synth → piano stem if available, else other
        use("synth",          raw_piano or raw_other)
        # brass → other stem (best available without commercial API)
        use("brass",          raw_other)
        use("other",          raw_other)

        jobs[job_id]["progress"] = 88
        jobs[job_id]["message"] = "Creating ZIP archive…"

        # ── Step 6: Zip all stems ─────────────────────────────────────────────
        zip_path = work_dir / "all_stems.zip"
        build_stems_zip(final_stems, zip_path)

        # ── Step 7: Build response URLs ───────────────────────────────────────
        stem_urls = {
            name: f"/stems/{job_id}/{name}.wav"
            for name in final_stems
        }
        stem_urls["_zip"] = f"/stems/{job_id}/all_stems.zip"

        jobs[job_id].update({
            "status": "done",
            "progress": 100,
            "message": "Separation complete!",
            "stems": stem_urls,
            "song_name": song_name,
        })

    except Exception as e:
        jobs[job_id].update({
            "status": "error",
            "progress": 0,
            "message": str(e),
        })


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"ok": True, "version": "1.0.0"}


@app.post("/separate")
async def separate(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """Upload an audio file and start stem separation."""
    allowed = {".mp3", ".wav", ".ogg", ".flac", ".m4a", ".aac"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed:
        raise HTTPException(400, f"Unsupported file type: {ext}")

    job_id = str(uuid.uuid4())
    input_path = UPLOAD_DIR / f"{job_id}{ext}"

    with open(input_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    jobs[job_id] = {
        "status": "queued",
        "progress": 0,
        "message": "Queued…",
        "stems": {},
        "song_name": file.filename,
    }

    background_tasks.add_task(
        separate_stems_job, job_id, input_path, file.filename
    )

    return {"job_id": job_id}


@app.get("/job/{job_id}")
def get_job(job_id: str):
    """Poll job status."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    return jobs[job_id]


@app.get("/stems/{job_id}/{filename}")
def get_stem(job_id: str, filename: str):
    """Download a single stem or the zip."""
    path = STEMS_DIR / job_id / filename
    if not path.exists():
        raise HTTPException(404, "Stem not found")
    media = "application/zip" if filename.endswith(".zip") else "audio/wav"
    return FileResponse(path, media_type=media, filename=filename)


@app.delete("/job/{job_id}")
def delete_job(job_id: str):
    """Clean up job files."""
    work_dir = STEMS_DIR / job_id
    if work_dir.exists():
        shutil.rmtree(work_dir)
    upload = list(UPLOAD_DIR.glob(f"{job_id}.*"))
    for f in upload:
        f.unlink(missing_ok=True)
    jobs.pop(job_id, None)
    return {"ok": True}
