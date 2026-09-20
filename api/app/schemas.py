from datetime import datetime
from typing import Optional
from decimal import Decimal
from pydantic import BaseModel, ConfigDict, Field
 
 
class VehicleBase(BaseModel):
    License_Plate_Number: str = Field(..., max_length=20, examples=["B1234ABC"])
    Normalized_Plate: str = Field(..., max_length=20, examples=["B1234ABC"])
    Resident_ID: int
    Vehicle_Type: Optional[str] = Field(default=None, max_length=50, examples=["Motor"])
 
 
class VehicleCreate(VehicleBase):
    """Body untuk POST (create)."""
    pass
 
 
class VehicleUpdate(BaseModel):
    """Body untuk PUT/PATCH (update) — semua field opsional."""
    License_Plate_Number: Optional[str] = Field(default=None, max_length=20)
    Normalized_Plate: Optional[str] = Field(default=None, max_length=20)
    Resident_ID: Optional[int] = None
    Vehicle_Type: Optional[str] = Field(default=None, max_length=50)
 
 
class VehicleOut(VehicleBase):
    """Response yang dikembalikan ke client, termasuk field yang di-generate DB."""
    model_config = ConfigDict(from_attributes=True)
 
    Vehicle_ID: int
    Created_At: datetime
    Updated_At: datetime
    
    
class ResidentBase(BaseModel):
    Resident_Name: str = Field(..., max_length=100, examples=["Budi Santoso"])
    Resident_Address: Optional[str] = Field(default=None, max_length=255, examples=["Jl. Merpati No. 12"])
    Resident_Phone_Number: Optional[str] = Field(default=None, max_length=20, examples=["081234567890"])
 
 
class ResidentCreate(ResidentBase):
    """Body untuk POST (create)."""
    pass
 
 
class ResidentUpdate(BaseModel):
    """Body untuk PUT/PATCH (update) — semua field opsional."""
    Resident_Name: Optional[str] = Field(default=None, max_length=100)
    Resident_Address: Optional[str] = Field(default=None, max_length=255)
    Resident_Phone_Number: Optional[str] = Field(default=None, max_length=20)
 
 
class ResidentOut(ResidentBase):
    """Response yang dikembalikan ke client, termasuk field yang di-generate DB."""
    model_config = ConfigDict(from_attributes=True)
 
    Resident_ID: int
    Created_At: datetime
    Updated_At: datetime


class CameraOut(BaseModel):
    """Kamera yang terdaftar; dipakai untuk mengisi filter kamera di dashboard."""
    model_config = ConfigDict(from_attributes=True)

    Camera_ID: int
    Camera_Name: str
    Type: Optional[str] = None
    Location: Optional[str] = None
    Is_Active: bool


class AnprLogOut(BaseModel):
    """Satu baris ANPR_Log untuk ditampilkan di Monitoring / ANPR Logs.

    Metrik bertipe Optional karena kolomnya DECIMAL yang NULL-able: metrik yang
    tidak tersedia disimpan sebagai NULL (bukan string "N/A"), dan sisi tampilan
    yang merender NULL menjadi "N/A".
    """
    model_config = ConfigDict(from_attributes=True)

    Log_ID: int
    Inserted_Time: Optional[datetime] = None
    Camera_ID: int
    License_Plate_Number: str
    Normalized_Plate: str
    Guard_Plate: Optional[str] = None
    Vehicle_ID: Optional[int] = None
    Classification: Optional[str] = None
    Direction: Optional[str] = None
    Event_Kind: Optional[str] = None
    Grant_Method: Optional[str] = None
    Entry_State: Optional[str] = None
    Detection_Confidence: Optional[float] = None
    OCR_Confidence: Optional[float] = None
    Processing_Time_MS: Optional[float] = None
    Environment_Label: Optional[str] = None
    Image_Ref: Optional[str] = None

    # Kolom hasil join, diisi router bila kendaraannya cocok dengan whitelist.
    Camera_Name: Optional[str] = None
    Resident_Name: Optional[str] = None
    Closed_By_Log_ID: Optional[int] = None


