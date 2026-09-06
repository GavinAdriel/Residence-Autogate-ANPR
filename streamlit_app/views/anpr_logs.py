import streamlit as st


def show_anpr_logs():

    st.subheader("ANPR Logs")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.text_input(
            "License Plate"
        )

    with col2:
        st.selectbox(
            "Classification",
            [
                "All",
                "AUTHORIZED",
                "UNKNOWN",
                "DENIED"
            ]
        )

    with col3:
        st.selectbox(
            "Camera",
            [
                "All",
                "Gate 01",
                "Gate 02"
            ]
        )

    st.divider()

    st.write(
        "ANPR history akan ditampilkan di sini."
    )