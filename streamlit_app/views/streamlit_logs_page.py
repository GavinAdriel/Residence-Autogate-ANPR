

from datetime import date, timedelta

import streamlit as st
from api_client import get_anpr_logs


def show_anpr_logs():
    st.title("📋 Log ANPR")

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        start_date = st.date_input("Dari tanggal", value=date.today() - timedelta(days=7))
    with col2:
        end_date = st.date_input("Sampai tanggal", value=date.today())
    with col3:
        st.write("")  # spacer biar tombol sejajar input
        st.write("")
        cari = st.button("🔍 Cari", use_container_width=True)

    if start_date > end_date:
        st.warning("Tanggal 'Dari' tidak boleh lebih besar dari tanggal 'Sampai'.")
        return

    try:
        logs = get_anpr_logs(start_date=start_date, end_date=end_date)
    except Exception as e:
        st.error(f"Gagal ambil data log: {e}")
        return

    st.caption(f"Menampilkan {len(logs)} log dari {start_date} sampai {end_date}")

    if not logs:
        st.info("Tidak ada log di rentang tanggal ini.")
        return

    st.dataframe(logs, use_container_width=True)

    # Ringkasan cepat, enak buat testing/verifikasi
    with st.expander("Ringkasan"):
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Total Log", len(logs))
        col_b.metric("Plat Unik", len({log["Normalized_Plate"] for log in logs}))
        col_c.metric(
            "Rata-rata OCR Confidence",
            f"{sum(float(log['OCR_Confidence'] or 0) for log in logs) / len(logs):.2%}",
        )


if __name__ == "__main__":
    show_anpr_logs()