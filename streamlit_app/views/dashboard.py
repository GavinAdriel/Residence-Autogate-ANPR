import streamlit as st


def show_dashboard():

    st.subheader("System Dashboard")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Residents",
            "128"
        )

    with col2:
        st.metric(
            "Vehicles",
            "94"
        )

    with col3:
        st.metric(
            "Active Cameras",
            "4"
        )

    with col4:
        st.metric(
            "Today's ANPR Logs",
            "1,248"
        )

    st.divider()

    st.write(
        "Dashboard statistics akan kita tambahkan "
        "setelah CRUD selesai."
    )