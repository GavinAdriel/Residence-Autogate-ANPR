import streamlit as st

from api_client import get_latest_anpr_log


def _pct(value) -> str:
    """Render confidence 0..1 sebagai persen, atau 'N/A' bila NULL.

    Metrik yang tidak tersedia disimpan sebagai NULL di kolom DECIMAL
    ANPR_Log, jadi None di sini berarti metriknya memang tidak ada
    (mis. OCR timeout) — bukan nol.
    """
    if value is None:
        return "N/A"
    return f"{float(value) * 100:.1f}%"


def _ms(value) -> str:
    """Render waktu proses dalam ms, atau 'N/A' bila NULL."""
    if value is None:
        return "N/A"
    return f"{float(value):.0f} ms"


def _status_label(log: dict) -> str:
    """Turunkan label status dari Classification + Grant_Method.

    Tidak ada kolom "status" di ANPR_Log; yang ada adalah klasifikasi
    (RESIDENT/GUEST) dan cara pemberian akses (AUTOMATIC/MANUAL/NONE),
    jadi labelnya dirangkai dari keduanya.
    """
    classification = log.get("Classification")
    grant = log.get("Grant_Method")

    if classification == "RESIDENT" and grant == "AUTOMATIC":
        return "AUTHORIZED (otomatis)"
    if grant == "MANUAL":
        return "MANUAL (diberi guard)"
    if classification == "GUEST":
        return "GUEST (perlu tinjauan)"
    if classification is None:
        return "UNRESOLVED (perlu tinjauan)"
    return f"{classification} / {grant or 'NONE'}"


def show_monitoring():

    st.subheader("Monitoring Management")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.info("LIVE CAMERA FEED")
        st.caption(
            "Live feed masih ditangani aplikasi desktop (Guard Dashboard); "
            "belum dialirkan ke web."
        )

    with col2:
        st.subheader("Last Detection")

        try:
            log = get_latest_anpr_log()
        except Exception as e:
            st.error(f"Gagal ambil data event terakhir: {e}")
            log = None
            st.caption("Pastikan API berjalan di http://localhost:8000.")

        if log is None:
            st.info("Belum ada event ANPR yang tercatat.")
        else:
            st.markdown(
                f'<div class="plate">{log.get("Normalized_Plate") or "-"}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(f"**Status:** {_status_label(log)}")

            resident_name = log.get("Resident_Name")
            st.markdown(f"**Resident:** {resident_name or 'Tidak terdaftar (guest)'}")

            st.markdown(f"**Kamera:** {log.get('Camera_Name') or log.get('Camera_ID')}")
            st.markdown(f"**OCR Confidence:** {_pct(log.get('OCR_Confidence'))}")
            st.markdown(
                f"**Detection Confidence:** {_pct(log.get('Detection_Confidence'))}"
            )
            st.markdown(f"**Processing:** {_ms(log.get('Processing_Time_MS'))}")
            st.caption(f"Waktu: {log.get('Inserted_Time') or '-'}")

    st.divider()

    if st.button("🔄 Refresh", key="monitoring_refresh"):
        st.rerun()
