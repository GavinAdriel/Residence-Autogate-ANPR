from datetime import date, datetime, timedelta
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from .. import models, schemas
from ..database import get_db

router = APIRouter(prefix="/anpr-logs", tags=["ANPR Log"])


@router.get("", response_model=List[schemas.ANPRLogOut])
def list_anpr_logs(
    start_date: Optional[date] = Query(
        default=None,
        description="Tanggal mulai (YYYY-MM-DD), inklusif. Kosongkan untuk tidak membatasi dari awal.",
    ),
    end_date: Optional[date] = Query(
        default=None,
        description="Tanggal akhir (YYYY-MM-DD), inklusif (mencakup seluruh hari itu). Kosongkan untuk tidak membatasi sampai akhir.",
    ),
    skip: int = 0,
    limit: int = 500,
    db: Session = Depends(get_db),
):
    """Ambil log ANPR, opsional difilter berdasarkan rentang tanggal `Inserted_Time`.

    Contoh: `/anpr-logs?start_date=2026-09-01&end_date=2026-09-19`
    """
    query = db.query(models.ANPRLog)

    if start_date is not None:
        query = query.filter(models.ANPRLog.Inserted_Time >= datetime.combine(start_date, datetime.min.time()))

    if end_date is not None:
        # +1 hari biar tanggal akhir ikut tercakup penuh (inklusif), bukan terpotong jam 00:00.
        end_exclusive = datetime.combine(end_date, datetime.min.time()) + timedelta(days=1)
        query = query.filter(models.ANPRLog.Inserted_Time < end_exclusive)

    return (
        query.order_by(models.ANPRLog.Inserted_Time.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )


@router.get("/{log_id}", response_model=schemas.ANPRLogOut)
def get_anpr_log(log_id: int, db: Session = Depends(get_db)):
    log = db.query(models.ANPRLog).filter(models.ANPRLog.Log_ID == log_id).first()
    if not log:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Log tidak ditemukan.")
    return log