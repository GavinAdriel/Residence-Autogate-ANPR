from sqlalchemy import Column, Integer, String, DateTime, Numeric
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
    
 
class ANPRLog(Base):
    """Mapping ke tabel `ANPR_Log` yang sudah ada di database anpr_system."""
 
    __tablename__ = "ANPR_Log"
 
    Log_ID = Column(Integer, primary_key=True, autoincrement=True)
    Inserted_Time = Column(DateTime, server_default=func.now(), nullable=False)
    Camera_ID = Column(Integer, nullable=False)
    License_Plate_Number = Column(String(20), nullable=False)
    Normalized_Plate = Column(String(20), nullable=False)
    Guard_Plate = Column(String(20), nullable=True)
    Vehicle_ID = Column(Integer, nullable=True)
    Classification = Column(String(32), nullable=True)
    Direction = Column(String(20), nullable=True)
    Event_Kind = Column(String(32), nullable=True)
    Grant_Method = Column(String(32), nullable=True)
    Entry_State = Column(String(32), nullable=False, server_default="OPEN")
    Detection_Confidence = Column(Numeric(5, 4), nullable=True)
    OCR_Confidence = Column(Numeric(5, 4), nullable=True)
    Processing_Time_MS = Column(Numeric(10, 2), nullable=True)
    Environment_Label = Column(String(32), nullable=True)
    Image_Ref = Column(String(512), nullable=True)
    Closed_By_Log_ID = Column(Integer, nullable=True)