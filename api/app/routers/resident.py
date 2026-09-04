from typing import List
 
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
 
from .. import models, schemas
from ..database import get_db
 
router = APIRouter(prefix="/residents", tags=["Resident"])
 
 
@router.post("", response_model=schemas.ResidentOut, status_code=status.HTTP_201_CREATED)
def create_resident(payload: schemas.ResidentCreate, db: Session = Depends(get_db)):
    resident = models.Resident(**payload.model_dump())
    db.add(resident)
    try:
        db.commit()
    except IntegrityError as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Gagal menyimpan, konflik di database: {e.orig}",
        )
    db.refresh(resident)
    return resident
 
 
@router.get("", response_model=List[schemas.ResidentOut])
def list_residents(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    return db.query(models.Resident).offset(skip).limit(limit).all()
 
 
@router.get("/{resident_id}", response_model=schemas.ResidentOut)
def get_resident(resident_id: int, db: Session = Depends(get_db)):
    resident = db.query(models.Resident).filter(models.Resident.Resident_ID == resident_id).first()
    if not resident:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Resident tidak ditemukan.")
    return resident
 
 
@router.put("/{resident_id}", response_model=schemas.ResidentOut)
def update_resident(resident_id: int, payload: schemas.ResidentUpdate, db: Session = Depends(get_db)):
    resident = db.query(models.Resident).filter(models.Resident.Resident_ID == resident_id).first()
    if not resident:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Resident tidak ditemukan.")
 
    updates = payload.model_dump(exclude_unset=True)
    for field, value in updates.items():
        setattr(resident, field, value)
 
    try:
        db.commit()
    except IntegrityError as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Gagal menyimpan, konflik di database: {e.orig}",
        )
    db.refresh(resident)
    return resident
 
 
@router.delete("/{resident_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_resident(resident_id: int, db: Session = Depends(get_db)):
    resident = db.query(models.Resident).filter(models.Resident.Resident_ID == resident_id).first()
    if not resident:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Resident tidak ditemukan.")
 
    try:
        db.delete(resident)
        db.commit()
    except IntegrityError as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Tidak bisa dihapus, masih dipakai data lain: {e.orig}",
        )
    return None
 