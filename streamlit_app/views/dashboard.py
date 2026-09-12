import streamlit as st
import pandas as pd

from api_client import (
    get_residents,
    get_total_residents_count,
    get_vehicles,
    get_total_vehicles_count
)

def apply_table_styles():
    st.markdown("""
    <style>
        /* Card Container untuk Tabel */
        .table-card {
            background-color: white;
            padding: 24px;
            border-radius: 12px;
            border: 1px solid #e5e7eb;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.05);
            margin-bottom: 24px;
        }

        /* Subheader di dalam Card */
        .table-title {
            font-size: 18px;
            font-weight: 700;
            color: #111827;
            margin-bottom: 16px;
        }

        /* Mengubah border dan radius dataframe Streamlit */
        div[data-testid="stDataFrame"] {
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            overflow: hidden;
        }
    </style>
    """, unsafe_allow_html=True)

def show_dashboard():
    apply_table_styles()

    st.subheader("System Dashboard")

    total_resident = get_total_residents_count()
    total_vehicle = get_total_vehicles_count()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Residents", total_resident)
    with col2:
        st.metric("Vehicles", total_vehicle)
    with col3:
        st.metric("Active Cameras", "4")
    with col4:
        st.metric("Today's ANPR Logs", "1,248")

    st.divider()

    residents = get_residents()
    vehicles = get_vehicles()

    # --- TABEL RESIDENT ---
    st.markdown('<div class="table-title">Daftar Resident</div>', unsafe_allow_html=True)
    
    if residents:
        df_residents = pd.DataFrame(residents)
        st.dataframe(
            df_residents,
            use_container_width=True,
            hide_index=True,
            column_config={
                "id": st.column_config.NumberColumn("ID", format="%d", width="small"),
                "name": st.column_config.TextColumn("Nama Lengkap", help="Nama terdaftar resident"),
                "phone": st.column_config.TextColumn("No. Telepon"),
                "unit": st.column_config.TextColumn("Unit / Blok"),
                "status": st.column_config.SelectboxColumn(
                    "Status",
                    options=["Active", "Inactive"],
                    required=True,
                )
            }
        )
    else:
        st.info("Belum ada data resident.")
    st.markdown('</div>', unsafe_allow_html=True)

    # --- TABEL VEHICLE ---
    st.markdown('<div class="table-title">Daftar Vehicle</div>', unsafe_allow_html=True)
    
    if vehicles:
        df_vehicles = pd.DataFrame(vehicles)
        st.dataframe(
            df_vehicles,
            use_container_width=True,
            hide_index=True,
            column_config={
                "id": st.column_config.NumberColumn("ID", format="%d", width="small"),
                "plate_number": st.column_config.TextColumn(
                    "Plat Nomor",
                    help="Nomor kendaraan terdaftar"
                ),
                "type": st.column_config.TextColumn("Jenis Kendaraan"),
                "owner": st.column_config.TextColumn("Pemilik"),
            }
        )
    else:
        st.info("Belum ada data vehicle.")
    st.markdown('</div>', unsafe_allow_html=True)