import streamlit as st


def show_monitoring():

    st.subheader("Monitoring Management")

    col1, col2 = st.columns([2, 1])

    with col1:

        st.info(
            "LIVE CAMERA FEED"
        )

    with col2:

        st.subheader("Last Detection")

        st.markdown(
            """
            ### B 1234 ABC
            **Status:** AUTHORIZED

            Resident: John Doe

            OCR Confidence: 96.8%

            Detection Confidence: 98.2%

            Processing: 184 ms
            """
        )

    st.divider()

    st.write(
        "Monitoring management akan dibuat "
        "di tahap berikutnya."
    )