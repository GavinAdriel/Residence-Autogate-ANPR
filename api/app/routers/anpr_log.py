"""Endpoint read-only untuk ANPR_Log dan Camera.

Baris ANPR_Log ditulis oleh aplikasi ANPR (paket `anpr`), bukan oleh API ini,
jadi router ini sengaja hanya menyediakan operasi baca: tidak ada POST/PUT/DELETE
untuk log. Dashboard Streamlit memakai endpoint ini untuk tab Monitoring dan
ANPR Logs, menggantikan data mockup yang sebelumnya di-hardcode.
"""

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from .. import models, schemas
from ..database import get_db

router = APIRouter(tags=["Monitoring"])


def _to_out(row) -> schemas.AnprLogOut:
    """Petakan satu baris hasil join menjadi ``AnprLogOut``.

    ``row`` adalah tuple ``(AnprLog, Camera_Name, Resident_Name)`` hasil outer
    join: nama kamera dan nama penghuni bisa ``None`` bila kendaraannya tidak
    cocok dengan whitelist (kasus guest).
    """
    log, camera_name, resident_name = row
    out = schemas.AnprLogOut.model_validate(log)
    out.Camera_Name = camera_name
    out.Resident_Name = resident_name
    return out


def _base_query(db: Session):
    """Query dasar: ANPR_Log + nama kamera + nama penghuni (via Vehicle).

    Memakai OUTER JOIN supaya event guest (Vehicle_ID NULL) tetap ikut terbawa;
    INNER JOIN akan menghilangkan justru baris yang paling perlu ditinjau guard.
    """
    return (
        db.query(
            models.AnprLog,
            models.Camera.Camera_Name,
            models.Resident.Resident_Name,
        )
        .outerjoin(models.Camera, models.Camera.Camera_ID == models.AnprLog.Camera_ID)
        .outerjoin(models.Vehicle, models.Vehicle.Vehicle_ID == models.AnprLog.Vehicle_ID)
        .outerjoin(
            models.Resident,
            models.Resident.Resident_ID == models.Vehicle.Resident_ID,
        )
    )


@router.get("/cameras", response_model=List[schemas.CameraOut])
def list_cameras(db: Session = Depends(get_db)):
    """Daftar kamera terdaftar, untuk mengisi dropdown filter kamera."""
    return db.query(models.Camera).order_by(models.Camera.Camera_ID).all()


@router.get("/anpr-logs", response_model=List[schemas.AnprLogOut])
def list_anpr_logs(
    plate: Optional[str] = Query(
        default=None, description="Cocokkan sebagian nomor plat (ternormalisasi)."
    ),
    classification: Optional[str] = Query(
        default=None, description="Filter tepat, mis. RESIDENT atau GUEST."
    ),
    camera_id: Optional[int] = Query(default=None, description="Filter per kamera."),
    limit: int = Query(default=100, ge=1, le=1000),
    skip: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
):
    """Log ANPR terbaru lebih dulu, dengan filter opsional."""
    query = _base_query(db)

    if plate:
        # Dinaikkan ke huruf besar agar cocok dengan Normalized_Plate.
        query = query.filter(
            models.AnprLog.Normalized_Plate.like(f"%{plate.upper()}%")
        )
    if classification:
        query = query.filter(models.AnprLog.Classification == classification)
    if camera_id is not None:
        query = query.filter(models.AnprLog.Camera_ID == camera_id)

    rows = (
        query.order_by(models.AnprLog.Log_ID.desc()).offset(skip).limit(limit).all()
    )
    return [_to_out(row) for row in rows]


@router.get("/anpr-logs/latest", response_model=Optional[schemas.AnprLogOut])
def latest_anpr_log(db: Session = Depends(get_db)):
    """Event terakhir, untuk panel "Last Detection" di tab Monitoring.

    Mengembalikan ``null`` (bukan 404) ketika belum ada event sama sekali, supaya
    dashboard bisa menampilkan status kosong tanpa memperlakukannya sebagai error.
    """
    row = _base_query(db).order_by(models.AnprLog.Log_ID.desc()).first()
    if row is None:
        return None
    return _to_out(row)


@router.get("/anpr-logs/stats")
def anpr_log_stats(db: Session = Depends(get_db)):
    """Ringkasan jumlah event, untuk kartu metrik di tab Dashboard."""
    total = db.query(models.AnprLog).count()
    residents = (
        db.query(models.AnprLog)
        .filter(models.AnprLog.Classification == "RESIDENT")
        .count()
    )
    guests = (
        db.query(models.AnprLog)
        .filter(models.AnprLog.Classification == "GUEST")
        .count()
    )
    automatic = (
        db.query(models.AnprLog)
        .filter(models.AnprLog.Grant_Method == "AUTOMATIC")
        .count()
    )
    manual = (
        db.query(models.AnprLog)
        .filter(models.AnprLog.Grant_Method == "MANUAL")
        .count()
    )
    return {
        "total_events": total,
        "resident_events": residents,
        "guest_events": guests,
        "automatic_grants": automatic,
        "manual_grants": manual,
        "registered_vehicles": db.query(models.Vehicle).count(),
        "registered_residents": db.query(models.Resident).count(),
    }


@router.get("/anpr-logs/{log_id}", response_model=schemas.AnprLogOut)
def get_anpr_log(log_id: int, db: Session = Depends(get_db)):
    """Satu baris log berdasarkan ``Log_ID``."""
    row = _base_query(db).filter(models.AnprLog.Log_ID == log_id).first()
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Log tidak ditemukan."
        )
    return _to_out(row)
