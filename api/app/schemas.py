from datetime import datetime
from typing import Optional
 
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