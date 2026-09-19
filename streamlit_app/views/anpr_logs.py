import streamlit as st

from api_client import get_anpr_logs, get_cameras

# Nilai Classification yang benar-benar ditulis aplikasi ANPR. Enum di
# anpr/core/models.py hanya punya RESIDENT dan GUEST -- bukan
# AUTHORIZED/DENIED -- dan bisa NULL ketika pencarian resident gagal,
# sehingga event itu diserahkan ke guard.
_CLASSIFICATIONS = ["Semua", "RESIDENT", "GUEST"]


def _fmt_pct(value) -> str:
    """Confidence 0..1 -> persen; NULL -> 'N/A'."""
    if value is None:
        return "N/A"
    return f"{float(value) * 100:.1f}%"


def _fmt_ms(value) -> str:
    """Waktu proses -> ms; NULL -> 'N/A'."""
    if value is None:
        return "N/A"
    return f"{float(value):.0f}"


def show_anpr_logs():

    st.subheader("ANPR Logs")

    # Daftar kamera diambil dari tabel Camera, bukan di-hardcode, supaya filter
    # selalu mencerminkan kamera yang benar-benar terdaftar.
    try:
        cameras = get_cameras()
    except Exception as e:
        st.warning(f"Gagal ambil daftar kamera: {e}")
        cameras = []

    camera_labels = {0: "Semua"}
    for cam in cameras:
        camera_labels[cam["Camera_ID"]] = cam.get("Camera_Name") or f"Camera {cam['Camera_ID']}"

    col1, col2, col3 = st.columns(3)

    with col1:
        plate = st.text_input("License Plate", placeholder="B1234ABC")

    with col2:
        classification = st.selectbox("Classification", _CLASSIFICATIONS)

    with col3:
        camera_id = st.selectbox(
            "Camera",
            options=list(camera_labels.keys()),
            format_func=lambda cid: camera_labels[cid],
        )

    limit = st.slider("Jumlah baris", min_value=10, max_value=500, value=100, step=10)

    st.divider()

    try:
        logs = get_anpr_logs(
            plate=plate or None,
            classification=None if classification == "Semua" else classification,
            camera_id=None if camera_id == 0 else camera_id,
            limit=limit,
        )
    except Exception as e:
        st.error(f"Gagal ambil ANPR logs: {e}")
        st.caption("Pastikan API berjalan di http://localhost:8000.")
        return

    if not logs:
        st.info("Belum ada event yang cocok dengan filter.")
        return

    # Diratakan menjadi baris tampilan supaya NULL dirender sebagai "N/A"
    # dan kolomnya berlabel manusiawi.
    table = [
        {
            "Log ID": log["Log_ID"],
            "Waktu": log.get("Inserted_Time") or "-",
            "Plat": log.get("Normalized_Plate") or "-",
            "Plat OCR": log.get("License_Plate_Number") or "-",
            "Plat Guard": log.get("Guard_Plate") or "-",
            "Kamera": log.get("Camera_Name") or log.get("Camera_ID"),
            "Resident": log.get("Resident_Name") or "-",
            "Klasifikasi": log.get("Classification") or "UNRESOLVED",
            "Akses": log.get("Grant_Method") or "NONE",
            "Deteksi": _fmt_pct(log.get("Detection_Confidence")),
            "OCR": _fmt_pct(log.get("OCR_Confidence")),
            "Proses (ms)": _fmt_ms(log.get("Processing_Time_MS")),
            "Environment": log.get("Environment_Label") or "-",
        }
        for log in logs
    ]

    st.caption(f"Menampilkan {len(table)} event (terbaru lebih dulu).")
    st.dataframe(table, use_container_width=True, hide_index=True)
