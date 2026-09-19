"""
Client tipis untuk memanggil ANPR API dari Streamlit.
Taruh file ini di dalam folder streamlit_app/, lalu import fungsinya
dari halaman mana pun yang butuh data Resident/Vehicle.
"""
 
import requests
 
API_URL = "http://localhost:8000"
 
TIMEOUT = 10  # detik — WAJIB, biar gak buffering selamanya kalau API hang
# ---------- Resident ----------
 
def get_residents():
    res = requests.get(f"{API_URL}/residents")
    res.raise_for_status()
    return res.json()

##========COUNT RESIDENT======
def get_total_residents_count():
    res = requests.get(f"{API_URL}/residents/count")
    res.raise_for_status()
    return res.json() 
 
def create_resident(data: dict):
    res = requests.post(f"{API_URL}/residents", json=data)
    res.raise_for_status()
    return res.json()
 
 
def update_resident(resident_id: int, data: dict):
    res = requests.put(f"{API_URL}/residents/{resident_id}", json=data)
    res.raise_for_status()
    return res.json()
 
 
def delete_resident(resident_id: int):
    res = requests.delete(f"{API_URL}/residents/{resident_id}")
    res.raise_for_status()
 
 
# ---------- Vehicle ----------
 
def get_vehicles():
    res = requests.get(f"{API_URL}/vehicles")
    res.raise_for_status()
    return res.json()


def get_total_vehicles_count():
    res = requests.get(f"{API_URL}/vehicles/count")
    res.raise_for_status()
    return res.json()
 
 
def create_vehicle(data: dict):
    res = requests.post(f"{API_URL}/vehicles", json=data)
    res.raise_for_status()
    return res.json()
 
 
def update_vehicle(vehicle_id: int, data: dict):
    res = requests.put(f"{API_URL}/vehicles/{vehicle_id}", json=data)
    res.raise_for_status()
    return res.json()
 
 
def delete_vehicle(vehicle_id: int):
    res = requests.delete(f"{API_URL}/vehicles/{vehicle_id}")
    res.raise_for_status()
    
def get_anpr_logs(start_date=None, end_date=None):

    params = {}
    if start_date:
        params["start_date"] = start_date.isoformat()
    if end_date:
        params["end_date"] = end_date.isoformat()
 
    res = requests.get(f"{API_URL}/anpr-logs", params=params)
    res.raise_for_status()
    return res.json()
    


# ---------- Camera ----------

def get_cameras():
    """Kamera terdaftar, untuk mengisi dropdown filter."""
    res = requests.get(f"{API_URL}/cameras", timeout=TIMEOUT)
    res.raise_for_status()
    return res.json()


# ---------- ANPR Log (read-only) ----------

def get_anpr_logs(
    plate: str | None = None,
    classification: str | None = None,
    camera_id: int | None = None,
    limit: int = 100,
):
    """Log ANPR terbaru lebih dulu, dengan filter opsional.

    Parameter yang bernilai None tidak dikirim, supaya API memakai defaultnya
    dan tidak memfilter kolom tersebut.
    """
    params: dict = {"limit": limit}
    if plate:
        params["plate"] = plate
    if classification:
        params["classification"] = classification
    if camera_id is not None:
        params["camera_id"] = camera_id

    res = requests.get(f"{API_URL}/anpr-logs", params=params, timeout=TIMEOUT)
    res.raise_for_status()
    return res.json()


def get_latest_anpr_log():
    """Event terakhir, atau None kalau belum ada event sama sekali."""
    res = requests.get(f"{API_URL}/anpr-logs/latest", timeout=TIMEOUT)
    res.raise_for_status()
    return res.json()


def get_anpr_stats():
    """Ringkasan jumlah event untuk kartu metrik."""
    res = requests.get(f"{API_URL}/anpr-logs/stats", timeout=TIMEOUT)
    res.raise_for_status()
    return res.json()
