import streamlit as st

from views.dashboard import show_dashboard
from views.monitoring import show_monitoring
from views.residents import show_residentsCRUD
from views.vehicles import show_vehicles
from views.anpr_logs import show_anpr_logs
from views.vehicle_crud import show_vehicles_API

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

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    [
        "📊 Dashboard",
        "🎥 Monitoring",
        "👤 Residents",
        "🚗 Vehicles",
        "📋 ANPR Logs",
        "Vehicle API"
    ]
)


with tab1:
    show_dashboard()

with tab2:
    show_monitoring()

with tab3:
    show_residentsCRUD()

with tab4:
    show_vehicles()

with tab5:
    show_anpr_logs()

with tab6:
    show_vehicles_API()



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



# ==============================
# PAGE ROUTER
# ==============================

# if page == "Dashboard":
#     from pages.dashboard import show_dashboard
#     show_dashboard()

# elif page == "Monitoring":
#     from pages.monitoring import show_monitoring
#     show_monitoring()

# elif page == "Residents":
#     from pages.residents import show_residents
#     show_residents()

# elif page == "Vehicles":
#     from pages.vehicles import show_vehicles
#     show_vehicles()

# elif page == "Cameras":
#     from pages.cameras import show_cameras
#     show_cameras()

# elif page == "ANPR Logs":
#     from pages.anpr_logs import show_anpr_logs
#     show_anpr_logs()

# elif page == "Users":
#     from pages.users import show_users
#     show_users()

# elif page == "Settings":
#     from pages.settings import show_settings
#     show_settings()
