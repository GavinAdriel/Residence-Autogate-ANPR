"""
Halaman untuk melihat log ANPR — bisa filter berdasarkan rentang tanggal
dan/atau plat nomor sekaligus.
Taruh file ini di dalam folder streamlit_app/, sejajar dengan api_client.py.
Jalankan sendiri untuk testing: streamlit run anpr_log_view.py
(atau panggil show_anpr_logs() dari file tab utama kamu)
"""

from datetime import date, timedelta

import streamlit as st
from api_client import get_anpr_logs


def show_anpr_logs():
    st.title("📋 Log ANPR")

    plate = st.text_input("Cari berdasarkan Nomor Plat (opsional)", placeholder="B1234ABC")

    filter_tanggal = st.checkbox("Batasi berdasarkan rentang tanggal", value=False)

    start_date = end_date = None
    if filter_tanggal:
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Dari tanggal", value=date.today() - timedelta(days=7))
        with col2:
            end_date = st.date_input("Sampai tanggal", value=date.today())

        if start_date > end_date:
            st.warning("Tanggal 'Dari' tidak boleh lebih besar dari tanggal 'Sampai'.")
            return

    cari = st.button("🔍 Cari")

    if not cari:
        return

    try:
        logs = get_anpr_logs(start_date=start_date, end_date=end_date, plate=plate or None)
    except Exception as e:
        st.error(f"Gagal ambil data log: {e}")
        return

    filter_desc = []
    if plate:
        filter_desc.append(f"plat mengandung '{plate.upper()}'")
    if filter_tanggal:
        filter_desc.append(f"tanggal {start_date} s/d {end_date}")
    desc = " dan ".join(filter_desc) if filter_desc else "semua data"
    st.caption(f"Menampilkan {len(logs)} log — filter: {desc}")

    if not logs:
        st.info("Tidak ada log yang cocok.")
        return

    st.dataframe(logs, use_container_width=True)

    # Ringkasan cepat, enak buat testing/verifikasi
    with st.expander("Ringkasan"):
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Total Log", len(logs))
        col_b.metric("Plat Unik", len({log["Normalized_Plate"] for log in logs}))
        confidences = [float(log["OCR_Confidence"]) for log in logs if log.get("OCR_Confidence") is not None]
        avg_conf = f"{sum(confidences) / len(confidences):.2%}" if confidences else "N/A"
        col_c.metric("Rata-rata OCR Confidence", avg_conf)


if __name__ == "__main__":
    show_anpr_logs()