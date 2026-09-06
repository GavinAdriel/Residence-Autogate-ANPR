
import streamlit as st


def show_vehicles():

    st.subheader("Vehicle Management")

    # ==========================================================
    # DUMMY DATA - UI ONLY
    # ==========================================================
    vehicles = [
        {
            "Vehicle_ID": 1,
            "License_Plate_Number": "B 1234 ABC",
            "Normalized_Plate": "B1234ABC",
            "Resident_ID": 1,
            "Resident_Name": "Budi Santoso",
            "Vehicle_Type": "Car",
            "Created_At": "2026-08-20 10:15:00",
            "Updated_At": "2026-08-20 10:15:00",
        },
        {
            "Vehicle_ID": 2,
            "License_Plate_Number": "B 5678 XYZ",
            "Normalized_Plate": "B5678XYZ",
            "Resident_ID": 2,
            "Resident_Name": "Andi Wijaya",
            "Vehicle_Type": "Motorcycle",
            "Created_At": "2026-08-21 09:30:00",
            "Updated_At": "2026-08-22 14:20:00",
        },
        {
            "Vehicle_ID": 3,
            "License_Plate_Number": "B 9012 DEF",
            "Normalized_Plate": "B9012DEF",
            "Resident_ID": 3,
            "Resident_Name": "Siti Rahma",
            "Vehicle_Type": "Car",
            "Created_At": "2026-08-22 11:45:00",
            "Updated_At": "2026-08-22 11:45:00",
        },
    ]

    # Dummy resident untuk dropdown
    residents = [
        {
            "Resident_ID": 1,
            "Resident_Name": "Budi Santoso"
        },
        {
            "Resident_ID": 2,
            "Resident_Name": "Andi Wijaya"
        },
        {
            "Resident_ID": 3,
            "Resident_Name": "Siti Rahma"
        },
    ]

    # ==========================================================
    # HEADER
    # ==========================================================
    col1, col2 = st.columns([4, 1])

    with col1:
        search = st.text_input(
            "Search Vehicle",
            placeholder="Search license plate or resident name..."
        )

    with col2:
        st.write("")
        add_button = st.button(
            "＋ Add Vehicle",
            use_container_width=True
        )

    # ==========================================================
    # ADD VEHICLE FORM
    # ==========================================================
    if add_button:
        st.session_state["show_add_vehicle"] = True

    if "show_add_vehicle" not in st.session_state:
        st.session_state["show_add_vehicle"] = False

    if st.session_state["show_add_vehicle"]:

        st.divider()

        st.markdown("### Add New Vehicle")

        with st.form("add_vehicle_form"):

            license_plate = st.text_input(
                "License Plate Number *",
                placeholder="Example: B 1234 ABC"
            )

            normalized_plate = st.text_input(
                "Normalized Plate *",
                placeholder="Example: B1234ABC"
            )

            resident_options = {
                f"{r['Resident_Name']} (ID: {r['Resident_ID']})":
                r["Resident_ID"]
                for r in residents
            }

            selected_resident = st.selectbox(
                "Resident *",
                options=list(resident_options.keys())
            )

            vehicle_type = st.selectbox(
                "Vehicle Type",
                options=[
                    "Car",
                    "Motorcycle",
                    "Truck",
                    "Van",
                    "Other"
                ]
            )

            col_save, col_cancel = st.columns(2)

            with col_save:
                save = st.form_submit_button(
                    "Save Vehicle",
                    use_container_width=True
                )

            with col_cancel:
                cancel = st.form_submit_button(
                    "Cancel",
                    use_container_width=True
                )

            if save:

                if not license_plate.strip():
                    st.error("License Plate Number is required.")

                elif not normalized_plate.strip():
                    st.error("Normalized Plate is required.")

                else:
                    st.success(
                        "Vehicle saved successfully. "
                        "(UI only - backend belum terhubung)"
                    )

            if cancel:
                st.session_state["show_add_vehicle"] = False
                st.rerun()

    st.divider()

    # ==========================================================
    # SEARCH FILTER
    # ==========================================================
    if search:

        search_lower = search.lower()

        filtered_vehicles = [
            vehicle
            for vehicle in vehicles
            if (
                search_lower in vehicle["License_Plate_Number"].lower()
                or search_lower in vehicle["Normalized_Plate"].lower()
                or search_lower in vehicle["Resident_Name"].lower()
            )
        ]

    else:
        filtered_vehicles = vehicles

    # ==========================================================
    # VEHICLE LIST
    # ==========================================================
    st.markdown("### Vehicle List")

    if not filtered_vehicles:
        st.info("No vehicle found.")
        return

    # Header
    header = st.columns(
        [0.6, 1.5, 1.5, 1.8, 1.2, 1.5, 1.5, 1.0]
    )

    header[0].markdown("**ID**")
    header[1].markdown("**License Plate**")
    header[2].markdown("**Normalized**")
    header[3].markdown("**Resident**")
    header[4].markdown("**Type**")
    header[5].markdown("**Created At**")
    header[6].markdown("**Updated At**")
    header[7].markdown("**Action**")

    st.divider()

    # Rows
    for vehicle in filtered_vehicles:

        row = st.columns(
            [0.6, 1.5, 1.5, 1.8, 1.2, 1.5, 1.5, 1.0]
        )

        row[0].write(vehicle["Vehicle_ID"])

        row[1].write(
            vehicle["License_Plate_Number"]
        )

        row[2].write(
            vehicle["Normalized_Plate"]
        )

        row[3].write(
            vehicle["Resident_Name"]
        )

        row[4].write(
            vehicle["Vehicle_Type"]
        )

        row[5].write(
            vehicle["Created_At"]
        )

        row[6].write(
            vehicle["Updated_At"]
        )

        with row[7]:

            edit_col, delete_col = st.columns(2)

            with edit_col:

                if st.button(
                    "✏️",
                    key=f"edit_vehicle_{vehicle['Vehicle_ID']}",
                    help="Edit Vehicle"
                ):
                    st.session_state["edit_vehicle_id"] = (
                        vehicle["Vehicle_ID"]
                    )

            with delete_col:

                if st.button(
                    "🗑️",
                    key=f"delete_vehicle_{vehicle['Vehicle_ID']}",
                    help="Delete Vehicle"
                ):
                    st.session_state["delete_vehicle_id"] = (
                        vehicle["Vehicle_ID"]
                    )

    # ==========================================================
    # EDIT VEHICLE FORM
    # ==========================================================
    if "edit_vehicle_id" in st.session_state:

        vehicle_id = st.session_state["edit_vehicle_id"]

        vehicle = next(
            (
                v for v in vehicles
                if v["Vehicle_ID"] == vehicle_id
            ),
            None
        )

        if vehicle:

            st.divider()

            st.markdown("### Edit Vehicle")

            with st.form(
                f"edit_vehicle_form_{vehicle_id}"
            ):

                st.text_input(
                    "Vehicle ID",
                    value=str(vehicle["Vehicle_ID"]),
                    disabled=True
                )

                license_plate = st.text_input(
                    "License Plate Number *",
                    value=vehicle["License_Plate_Number"]
                )

                normalized_plate = st.text_input(
                    "Normalized Plate *",
                    value=vehicle["Normalized_Plate"]
                )

                resident_options = {
                    f"{r['Resident_Name']} (ID: {r['Resident_ID']})":
                    r["Resident_ID"]
                    for r in residents
                }

                current_resident = (
                    f"{vehicle['Resident_Name']} "
                    f"(ID: {vehicle['Resident_ID']})"
                )

                resident_list = list(
                    resident_options.keys()
                )

                current_index = (
                    resident_list.index(current_resident)
                    if current_resident in resident_list
                    else 0
                )

                selected_resident = st.selectbox(
                    "Resident *",
                    options=resident_list,
                    index=current_index
                )

                vehicle_types = [
                    "Car",
                    "Motorcycle",
                    "Truck",
                    "Van",
                    "Other"
                ]

                current_type = vehicle["Vehicle_Type"]

                type_index = (
                    vehicle_types.index(current_type)
                    if current_type in vehicle_types
                    else 0
                )

                vehicle_type = st.selectbox(
                    "Vehicle Type",
                    options=vehicle_types,
                    index=type_index
                )

                col_update, col_cancel = st.columns(2)

                with col_update:

                    update = st.form_submit_button(
                        "Update Vehicle",
                        use_container_width=True
                    )

                with col_cancel:

                    cancel = st.form_submit_button(
                        "Cancel",
                        use_container_width=True
                    )

                if update:

                    if not license_plate.strip():

                        st.error(
                            "License Plate Number is required."
                        )

                    elif not normalized_plate.strip():

                        st.error(
                            "Normalized Plate is required."
                        )

                    else:

                        st.success(
                            "Vehicle updated successfully. "
                            "(UI only - backend belum terhubung)"
                        )

                if cancel:

                    del st.session_state[
                        "edit_vehicle_id"
                    ]

                    st.rerun()

    # ==========================================================
    # DELETE CONFIRMATION
    # ==========================================================
    if "delete_vehicle_id" in st.session_state:

        vehicle_id = st.session_state["delete_vehicle_id"]

        vehicle = next(
            (
                v for v in vehicles
                if v["Vehicle_ID"] == vehicle_id
            ),
            None
        )

        if vehicle:

            st.divider()

            st.warning(
                f"Are you sure you want to delete "
                f"**{vehicle['License_Plate_Number']}** "
                f"owned by **{vehicle['Resident_Name']}**?"
            )

            col_delete, col_cancel = st.columns(2)

            with col_delete:

                if st.button(
                    "Delete Vehicle",
                    type="primary",
                    use_container_width=True
                ):

                    st.success(
                        "Vehicle deleted successfully. "
                        "(UI only - backend belum terhubung)"
                    )

                    del st.session_state[
                        "delete_vehicle_id"
                    ]

                    st.rerun()

            with col_cancel:

                if st.button(
                    "Cancel Delete",
                    use_container_width=True
                ):

                    del st.session_state[
                        "delete_vehicle_id"
                    ]

                    st.rerun()

