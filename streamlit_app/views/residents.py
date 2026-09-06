import streamlit as st
from datetime import datetime


def show_residents():

    st.subheader("Resident Management")

    # ==========================================================
    # DUMMY DATA - UI ONLY
    # ==========================================================
    residents = [
        {
            "Resident_ID": 1,
            "Resident_Name": "Budi Santoso",
            "Resident_Address": "Jl. Melati No. 10",
            "Resident_Phone_Number": "081234567890",
            "Created_At": "2026-08-20 10:15:00",
            "Updated_At": "2026-08-20 10:15:00",
        },
        {
            "Resident_ID": 2,
            "Resident_Name": "Andi Wijaya",
            "Resident_Address": "Jl. Mawar No. 25",
            "Resident_Phone_Number": "081298765432",
            "Created_At": "2026-08-21 09:30:00",
            "Updated_At": "2026-08-22 14:20:00",
        },
        {
            "Resident_ID": 3,
            "Resident_Name": "Siti Rahma",
            "Resident_Address": "Jl. Kenanga No. 5",
            "Resident_Phone_Number": "085712345678",
            "Created_At": "2026-08-22 11:45:00",
            "Updated_At": "2026-08-22 11:45:00",
        },
    ]

    # ==========================================================
    # HEADER
    # ==========================================================
    col1, col2 = st.columns([4, 1])

    with col1:
        search = st.text_input(
            "Search Resident",
            placeholder="Enter resident name..."
        )

    with col2:
        st.write("")
        add_button = st.button(
            "＋ Add Resident",
            use_container_width=True
        )

    # ==========================================================
    # ADD RESIDENT FORM
    # ==========================================================
    if add_button:
        st.session_state["show_add_resident"] = True

    if "show_add_resident" not in st.session_state:
        st.session_state["show_add_resident"] = False

    if st.session_state["show_add_resident"]:

        st.divider()

        st.markdown("### Add New Resident")

        with st.form("add_resident_form"):

            name = st.text_input(
                "Resident Name *",
                placeholder="Enter resident name"
            )

            address = st.text_area(
                "Resident Address",
                placeholder="Enter resident address"
            )

            phone = st.text_input(
                "Phone Number",
                placeholder="Enter phone number"
            )

            col_save, col_cancel = st.columns(2)

            with col_save:
                save = st.form_submit_button(
                    "Save Resident",
                    use_container_width=True
                )

            with col_cancel:
                cancel = st.form_submit_button(
                    "Cancel",
                    use_container_width=True
                )

            if save:
                if not name.strip():
                    st.error("Resident Name is required.")
                else:
                    st.success(
                        "Resident saved successfully. "
                        "(UI only - backend belum terhubung)"
                    )

            if cancel:
                st.session_state["show_add_resident"] = False
                st.rerun()

    st.divider()

    # ==========================================================
    # SEARCH FILTER - UI ONLY
    # ==========================================================
    if search:
        filtered_residents = [
            resident
            for resident in residents
            if search.lower() in resident["Resident_Name"].lower()
        ]
    else:
        filtered_residents = residents

    # ==========================================================
    # RESIDENT TABLE
    # ==========================================================
    st.markdown("### Resident List")

    if not filtered_residents:
        st.info("No resident found.")
        return

    # Header
    header = st.columns([0.7, 2, 3, 1.8, 1.5, 1.5, 1.5])

    header[0].markdown("**ID**")
    header[1].markdown("**Name**")
    header[2].markdown("**Address**")
    header[3].markdown("**Phone**")
    header[4].markdown("**Created At**")
    header[5].markdown("**Updated At**")
    header[6].markdown("**Action**")

    st.divider()

    # Rows
    for resident in filtered_residents:

        row = st.columns([0.7, 2, 3, 1.8, 1.5, 1.5, 1.5])

        row[0].write(resident["Resident_ID"])
        row[1].write(resident["Resident_Name"])
        row[2].write(resident["Resident_Address"])
        row[3].write(resident["Resident_Phone_Number"])
        row[4].write(resident["Created_At"])
        row[5].write(resident["Updated_At"])

        with row[6]:

            edit_col, delete_col = st.columns(2)

            with edit_col:
                if st.button(
                    "✏️",
                    key=f"edit_{resident['Resident_ID']}",
                    help="Edit Resident"
                ):
                    st.session_state["edit_resident_id"] = (
                        resident["Resident_ID"]
                    )

            with delete_col:
                if st.button(
                    "🗑️",
                    key=f"delete_{resident['Resident_ID']}",
                    help="Delete Resident"
                ):
                    st.session_state["delete_resident_id"] = (
                        resident["Resident_ID"]
                    )

    # ==========================================================
    # EDIT FORM
    # ==========================================================
    if "edit_resident_id" in st.session_state:

        resident_id = st.session_state["edit_resident_id"]

        resident = next(
            (
                r for r in residents
                if r["Resident_ID"] == resident_id
            ),
            None
        )

        if resident:

            st.divider()
            st.markdown("### Edit Resident")

            with st.form(f"edit_resident_{resident_id}"):

                st.text_input(
                    "Resident ID",
                    value=str(resident["Resident_ID"]),
                    disabled=True
                )

                name = st.text_input(
                    "Resident Name *",
                    value=resident["Resident_Name"]
                )

                address = st.text_area(
                    "Resident Address",
                    value=resident["Resident_Address"]
                )

                phone = st.text_input(
                    "Phone Number",
                    value=resident["Resident_Phone_Number"]
                )

                col_update, col_cancel = st.columns(2)

                with col_update:
                    update = st.form_submit_button(
                        "Update Resident",
                        use_container_width=True
                    )

                with col_cancel:
                    cancel = st.form_submit_button(
                        "Cancel",
                        use_container_width=True
                    )

                if update:
                    if not name.strip():
                        st.error("Resident Name is required.")
                    else:
                        st.success(
                            "Resident updated successfully. "
                            "(UI only - backend belum terhubung)"
                        )

                if cancel:
                    del st.session_state["edit_resident_id"]
                    st.rerun()

    # ==========================================================
    # DELETE CONFIRMATION
    # ==========================================================
    if "delete_resident_id" in st.session_state:

        resident_id = st.session_state["delete_resident_id"]

        resident = next(
            (
                r for r in residents
                if r["Resident_ID"] == resident_id
            ),
            None
        )

        if resident:

            st.divider()

            st.warning(
                f"Are you sure you want to delete "
                f"**{resident['Resident_Name']}**?"
            )

            col_delete, col_cancel = st.columns(2)

            with col_delete:
                if st.button(
                    "Delete Resident",
                    type="primary",
                    use_container_width=True
                ):
                    st.success(
                        "Resident deleted successfully. "
                        "(UI only - backend belum terhubung)"
                    )

                    del st.session_state["delete_resident_id"]
                    st.rerun()

            with col_cancel:
                if st.button(
                    "Cancel Delete",
                    use_container_width=True
                ):
                    del st.session_state["delete_resident_id"]
                    st.rerun()