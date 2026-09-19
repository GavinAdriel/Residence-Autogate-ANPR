from sqlalchemy import Boolean, Column, DateTime, Integer, Numeric, String
from sqlalchemy.sql import func
 
from .database import Base
class Resident(Base):
    """Mapping ke tabel `Resident` yang sudah ada di database anpr_system."""
 
    __tablename__ = "Resident"
 
    Resident_ID = Column(Integer, primary_key=True, autoincrement=True)
    Resident_Name = Column(String(100), nullable=False)
    Resident_Address = Column(String(255), nullable=True)
    Resident_Phone_Number = Column(String(20), nullable=True)
    Created_At = Column(DateTime, server_default=func.now())
    Updated_At = Column(DateTime, server_default=func.now(), onupdate=func.now())
 
 
class Vehicle(Base):
    """Mapping ke tabel `Vehicle` yang sudah ada di database anpr_system.
    Kolom disesuaikan persis dengan struktur tabel di phpMyAdmin."""
 
    __tablename__ = "Vehicle"
 
    Vehicle_ID = Column(Integer, primary_key=True, autoincrement=True)
    License_Plate_Number = Column(String(20), nullable=False)
    Normalized_Plate = Column(String(20), nullable=False, unique=True)
    Resident_ID = Column(Integer, nullable=False)
    Vehicle_Type = Column(String(50), nullable=True)
    Created_At = Column(DateTime, server_default=func.now())
    Updated_At = Column(DateTime, server_default=func.now(), onupdate=func.now())


class Camera(Base):
    """Mapping ke tabel `Camera` di database anpr_system.

    Dipakai untuk mengisi filter kamera di dashboard (menggantikan daftar
    'Gate 01/Gate 02' yang sebelumnya di-hardcode), dan menjadi target foreign
    key `ANPR_Log.Camera_ID` yang NOT NULL.
    """

    __tablename__ = "Camera"

    Camera_ID = Column(Integer, primary_key=True, autoincrement=True)
    Camera_Name = Column(String(100), nullable=False)
    Type = Column(String(50), nullable=True)
    Location = Column(String(100), nullable=True)
    IP_Address = Column(String(50), nullable=True)
    Is_Active = Column(Boolean, nullable=False, default=True)
    Created_At = Column(DateTime, server_default=func.now())


class AnprLog(Base):
    """Mapping ke tabel `ANPR_Log` di database anpr_system.

    Read-only dari sisi API: baris ditulis oleh aplikasi ANPR (paket `anpr`),
    sedangkan dashboard hanya membacanya. Kolom metrik bertipe DECIMAL dan
    bernilai NULL bila metriknya tidak tersedia (mis. OCR timeout).
    """

    __tablename__ = "ANPR_Log"

    Log_ID = Column(Integer, primary_key=True, autoincrement=True)
    Inserted_Time = Column(DateTime, server_default=func.now())
    Camera_ID = Column(Integer, nullable=False)
    License_Plate_Number = Column(String(20), nullable=False)
    Normalized_Plate = Column(String(20), nullable=False)
    Guard_Plate = Column(String(20), nullable=True)
    Vehicle_ID = Column(Integer, nullable=True)
    Classification = Column(String(32), nullable=True)
    Direction = Column(String(20), nullable=True)
    Event_Kind = Column(String(32), nullable=True)
    Grant_Method = Column(String(32), nullable=True)
    Entry_State = Column(String(32), nullable=False, default="OPEN")
    Detection_Confidence = Column(Numeric(5, 4), nullable=True)
    OCR_Confidence = Column(Numeric(5, 4), nullable=True)
    Processing_Time_MS = Column(Numeric(10, 2), nullable=True)
    Environment_Label = Column(String(32), nullable=True)
    Image_Ref = Column(String(512), nullable=True)
    Closed_By_Log_ID = Column(Integer, nullable=True)
