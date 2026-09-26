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

    # Semua widget filter dibungkus st.form — mengubah nilainya TIDAK memicu
    # rerun sama sekali. Script cuma rerun sekali saat tombol submit diklik.
    with st.form("anpr_log_filter_form"):
        plate = st.text_input("Cari berdasarkan Nomor Plat (opsional)", placeholder="B1234ABC")

        resident_option = st.selectbox(
            "Status Kendaraan",
            options=["Semua", "Resident", "Bukan Resident"],
        )
        resident_filter = {"Semua": None, "Resident": True, "Bukan Resident": False}[resident_option]

        filter_tanggal = st.checkbox("Batasi berdasarkan rentang tanggal", value=False)

        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Dari tanggal", value=date.today() - timedelta(days=7))
        with col2:
            end_date = st.date_input("Sampai tanggal", value=date.today())

        cari = st.form_submit_button("🔍 Cari")

    # Validasi & pemanggilan API baru terjadi setelah form di-submit —
    # bagian ini jalan sekali per klik tombol, bukan tiap widget diubah.
    if cari:
        if filter_tanggal and start_date > end_date:
            st.warning("Tanggal 'Dari' tidak boleh lebih besar dari tanggal 'Sampai'.")
            return

        try:
            logs = get_anpr_logs(
                start_date=start_date if filter_tanggal else None,
                end_date=end_date if filter_tanggal else None,
                plate=plate or None,
                resident=resident_filter,
            )
        except Exception as e:
            st.error(f"Gagal ambil data log: {e}")
            return

        st.session_state["anpr_logs_result"] = logs
        st.session_state["anpr_logs_filter_desc"] = {
            "plate": plate,
            "resident_filter": resident_filter,
            "filter_tanggal": filter_tanggal,
            "start_date": start_date,
            "end_date": end_date,
        }

    if "anpr_logs_result" not in st.session_state:
        return

    logs = st.session_state["anpr_logs_result"]
    f = st.session_state["anpr_logs_filter_desc"]

    filter_desc = []
    if f["plate"]:
        filter_desc.append(f"plat mengandung '{f['plate'].upper()}'")
    if f["resident_filter"] is not None:
        filter_desc.append("resident" if f["resident_filter"] else "bukan resident")
    if f["filter_tanggal"]:
        filter_desc.append(f"tanggal {f['start_date']} s/d {f['end_date']}")
    desc = " dan ".join(filter_desc) if filter_desc else "semua data"
    st.caption(f"Menampilkan {len(logs)} log — filter: {desc}")

    if not logs:
        st.info("Tidak ada log yang cocok.")
        return

    st.dataframe(logs, use_container_width=True)

    with st.expander("Ringkasan"):
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Total Log", len(logs))
        col_b.metric("Plat Unik", len({log["Normalized_Plate"] for log in logs}))
        confidences = [float(log["OCR_Confidence"]) for log in logs if log.get("OCR_Confidence") is not None]
        avg_conf = f"{sum(confidences) / len(confidences):.2%}" if confidences else "N/A"
        col_c.metric("Rata-rata OCR Confidence", avg_conf)


if __name__ == "__main__":
    show_anpr_logs()