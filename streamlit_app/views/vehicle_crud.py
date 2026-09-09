import streamlit as st

from api_client import (
    get_residents,
    get_vehicles,
    create_vehicle,
    update_vehicle,
    delete_vehicle,
)

def show_vehicles_API():
    st.title("🚗 CRUD Vehicle")

    # ---------- Load resident ----------
    try:
        residents = get_residents()
    except Exception as e:
        st.error(f"Gagal ambil data resident: {e}")
        residents = []

    if not residents:
        st.warning("Belum ada data Resident. Tambahkan Resident dulu sebelum bisa tambah Vehicle.")

    resident_options = {
        r["Resident_ID"]: f"{r['Resident_ID']} - {r['Resident_Name']}" for r in residents
    }

    # ---------- Load vehicle (sekali di awal) ----------
    try:
        vehicles = get_vehicles()
    except Exception as e:
        st.error(f"Gagal ambil data vehicle: {e}")
        vehicles = []

    # ================== TAMBAH ==================
    st.subheader("➕ Tambah Vehicle")

    with st.form("form_add_vehicle", clear_on_submit=True):
        plate = st.text_input("Nomor Plat (License_Plate_Number)", placeholder="B1234ABC")
        normalized = st.text_input(
            "Normalized Plate",
            placeholder="B1234ABC",
            help="Diformat/dibersihkan (tanpa spasi, huruf besar semua).",
        )
        resident_id = st.selectbox(
            "Resident",
            options=list(resident_options.keys()) if resident_options else [],
            format_func=lambda x: resident_options.get(x, x),
            disabled=not resident_options,
        )
        vehicle_type = st.selectbox("Jenis Kendaraan", ["Motor", "Mobil"])
        submitted = st.form_submit_button("Tambah", disabled=not resident_options)

    if submitted:
        if not plate or not normalized:
            st.warning("Nomor plat dan normalized plate wajib diisi.")
        else:
            try:
                create_vehicle({
                    "License_Plate_Number": plate,
                    "Normalized_Plate": normalized,
                    "Resident_ID": resident_id,
                    "Vehicle_Type": vehicle_type,
                })
                st.success(f"Vehicle {plate} berhasil ditambahkan!")
                vehicles = get_vehicles()   # <-- fetch ulang via API
            except Exception as e:
                st.error(f"Gagal tambah vehicle: {e}")

    st.divider()

    # ================== EDIT / HAPUS ==================
    st.subheader("✏️ Edit / Hapus Vehicle")

    if vehicles:
        vehicle_options = {
            v["Vehicle_ID"]: f"{v['Vehicle_ID']} - {v['License_Plate_Number']}" for v in vehicles
        }
        selected_id = st.selectbox(
            "Pilih Vehicle",
            options=list(vehicle_options.keys()),
            format_func=lambda x: vehicle_options[x],
        )
        selected_vehicle = next(v for v in vehicles if v["Vehicle_ID"] == selected_id)

        with st.form("form_edit_vehicle"):
            edit_plate = st.text_input("Nomor Plat", value=selected_vehicle["License_Plate_Number"])
            edit_normalized = st.text_input("Normalized Plate", value=selected_vehicle["Normalized_Plate"])
            edit_resident_id = st.selectbox(
                "Resident",
                options=list(resident_options.keys()) if resident_options else [],
                format_func=lambda x: resident_options.get(x, x),
                index=list(resident_options.keys()).index(selected_vehicle["Resident_ID"])
                if selected_vehicle["Resident_ID"] in resident_options else 0,
            )
            edit_type = st.selectbox(
                "Jenis Kendaraan",
                ["Motor", "Mobil"],
                index=["Motor", "Mobil"].index(selected_vehicle["Vehicle_Type"])
                if selected_vehicle["Vehicle_Type"] in ["Motor", "Mobil"] else 0,
            )
            col_update, col_delete = st.columns(2)
            with col_update:
                update_clicked = st.form_submit_button("💾 Update", use_container_width=True)
            with col_delete:
                delete_clicked = st.form_submit_button("🗑️ Hapus", use_container_width=True, type="primary")

        if update_clicked:
            try:
                update_vehicle(selected_id, {
                    "License_Plate_Number": edit_plate,
                    "Normalized_Plate": edit_normalized,
                    "Resident_ID": edit_resident_id,
                    "Vehicle_Type": edit_type,
                })
                st.success("Vehicle berhasil diupdate!")
                vehicles = get_vehicles()   # <-- fetch ulang via API
            except Exception as e:
                st.error(f"Gagal update: {e}")

        if delete_clicked:
            try:
                delete_vehicle(selected_id)
                st.success("Vehicle berhasil dihapus!")
                vehicles = get_vehicles()   # <-- fetch ulang via API
            except Exception as e:
                st.error(f"Gagal hapus: {e}")
    else:
        st.info("Belum ada vehicle untuk diedit.")

    st.divider()

    # ================== TABEL (DIGAMBAR PALING AKHIR) ==================
    st.subheader("Daftar Vehicle")
    if vehicles:
        st.dataframe(vehicles, use_container_width=True)
    else:
        st.info("Belum ada data vehicle.")

if __name__ == "__main__":
    show_vehicles_API()
