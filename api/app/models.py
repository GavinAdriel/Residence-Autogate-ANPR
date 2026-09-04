from sqlalchemy import Column, Integer, String, DateTime
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