import os
from dotenv import load_dotenv

 
# Sesuaikan dengan docker-compose.yml 
# Kalau port MySQL di host sudah diganti (mis. 3307:3306), pakai port itu di sini.
DB_HOST = "localhost"
DB_PORT = "3306" #bisa ganti sesuai kebutuhan
DB_USER = "anpr"
DB_PASSWORD = "anprpassword"
DB_NAME = "anpr_system"
 
DATABASE_URL = (
    f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)