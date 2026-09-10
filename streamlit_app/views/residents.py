import streamlit as st

from api_client import (
    get_residents,
    create_resident,
    update_resident,
    delete_resident,
)

def show_residentsCRUD():
    st.title("👤 Residents")

    # ---------- Load resident (sekali di awal) ----------
    try:
        residents = get_residents()
    except Exception as e:
        st.error(f"Gagal ambil data resident: {e}")
        residents = []

    if not residents:
        st.warning("Belum ada data Resident. Tambahkan Resident dulu sebelum bisa tambah Vehicle.")

    # ================== TAMBAH ==================
    st.subheader("➕ Tambah Resident")

    with st.form("form_add_resident", clear_on_submit=True):
        name = st.text_input("Resident Name *", placeholder="Enter resident name")
        address = st.text_area("Resident Address", placeholder="Enter resident address")
        phone = st.text_input("Phone Number", placeholder="Enter phone number")

        submitted = st.form_submit_button("Tambah", use_container_width=True)

    if submitted:
        errors = []
        if not name.strip():
            errors.append("Resident Name wajib diisi.")
        if not address.strip():
            errors.append("Resident Address wajib diisi.")
        if not phone.strip():
            errors.append("Resident Phone wajib diisi.")

        if errors:
            for err in errors:
                st.error(err)
        else:
            try:
                create_resident({
                    "Resident_Name": name.strip(),
                    "Resident_Address": address.strip(),
                    "Resident_Phone_Number": phone.strip()
                })
                st.success("Resident berhasil disimpan.")
                residents = get_residents()   # <-- fetch ulang via API
            except Exception as e:
                st.error(f"Gagal menyimpan resident: {e}")

    st.divider()

    # ================== EDIT / HAPUS ==================
    st.subheader("✏️ Edit / Hapus Resident")

    if residents:
        id_key_candidates = ["Resident_ID", "id", "ID"]
        id_key = next((k for k in id_key_candidates if k in residents[0]), None)

        if id_key is None:
            st.error("Tidak dapat menemukan field ID pada data resident. Cek struktur API.")
        else:
            resident_options = {
                r[id_key]: f"{r[id_key]} - {r.get('Resident_Name', '-')}" for r in residents
            }
            selected_id = st.selectbox(
                "Pilih Resident",
                options=list(resident_options.keys()),
                format_func=lambda x: resident_options[x],
            )
            selected_resident = next(r for r in residents if r[id_key] == selected_id)

            with st.form("form_edit_resident"):
                edit_name = st.text_input("Resident Name", value=selected_resident.get("Resident_Name", ""))
                edit_address = st.text_area("Resident Address", value=selected_resident.get("Resident_Address", ""))
                edit_phone = st.text_input("Phone Number", value=selected_resident.get("Resident_Phone_Number", ""))

                col_update, col_delete = st.columns(2)
                with col_update:
                    update_clicked = st.form_submit_button("💾 Update", use_container_width=True)
                with col_delete:
                    delete_clicked = st.form_submit_button("🗑️ Hapus", use_container_width=True, type="primary")

            if update_clicked:
                if not edit_name.strip() or not edit_address.strip() or not edit_phone.strip():
                    st.warning("Semua field wajib diisi.")
                else:
                    try:
                        update_resident(selected_id, {
                            "Resident_Name": edit_name.strip(),
                            "Resident_Address": edit_address.strip(),
                            "Resident_Phone_Number": edit_phone.strip(),
                        })
                        st.success("Resident berhasil diupdate!")
                        residents = get_residents()   # <-- fetch ulang via API
                    except Exception as e:
                        st.error(f"Gagal update: {e}")

            if delete_clicked:
                try:
                    delete_resident(selected_id)
                    st.success("Resident berhasil dihapus!")
                    residents = get_residents()   # <-- fetch ulang via API
                except Exception as e:
                    st.error(f"Gagal hapus: {e}")
    else:
        st.info("Belum ada resident untuk diedit.")

    st.divider()

    # ================== TABEL (DIGAMBAR PALING AKHIR) ==================
    st.subheader("Daftar Resident")
    if residents:
        st.dataframe(residents, use_container_width=True, hide_index=True)
    else:
        st.info("Belum ada data resident.")


if __name__ == "__main__":
    show_residentsCRUD()