import streamlit as st

from views.dashboard import show_dashboard
from views.residents import show_residentsCRUD
from views.vehicles import show_vehicles_API
from views.streamlit_logs_page import show_anpr_logs

st.set_page_config(
    page_title="ANPR System",
    page_icon="🚗",
    layout="wide"
)

st.title("Admin Dashboard")

st.write(
        f"Welcome, Username! This is your dashboard where you can monitor and manage the ANPR system."
    )

st.divider()

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Dashboard",
        "Residents",
        "Vehicles",
        "ANPR Logs",
    ]
)


with tab1:
    show_dashboard()

with tab2:
    show_residentsCRUD()

with tab3:
    show_vehicles_API()

with tab4:
    show_anpr_logs()




st.markdown("""
<style>

    /* Main background */
    .stApp {
        background-color: #f5f7fa;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #111827;
    }

    section[data-testid="stSidebar"] * {
        color: white;
    }

    /* Cards */
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
    }

    .metric-title {
        font-size: 14px;
        color: #6b7280;
    }

    .metric-value {
        font-size: 30px;
        font-weight: 700;
        margin-top: 5px;
    }

    /* Status */
    .status-online {
        color: #16a34a;
        font-weight: 600;
    }

    .status-offline {
        color: #dc2626;
        font-weight: 600;
    }

    /* Plate */
    .plate {
        background-color: #f8fafc;
        border: 3px solid #111827;
        border-radius: 8px;
        padding: 12px 20px;
        font-size: 28px;
        font-weight: 800;
        text-align: center;
        letter-spacing: 3px;
    }

</style>
""", unsafe_allow_html=True)


with st.sidebar:

    # st.markdown("## 🚗 ANPR")
    # st.caption("Residence Autogate System")

    st.divider()

    # page = st.radio(
    #     "MENU",
    #     [
    #         "Dashboard",
    #         "Monitoring",
    #         "Residents",
    #         "Vehicles",
    #         "Cameras",
    #         "ANPR Logs",
    #         "Users",
    #         "Settings"
    #     ]
    # )

    st.divider()

    st.markdown("### System")

    st.markdown(
        '<span class="status-online">● System Online</span>',
        unsafe_allow_html=True
    )

    st.caption("v1.0.0")


