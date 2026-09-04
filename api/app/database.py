from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
 
from .config import DATABASE_URL
 
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
 
 
def get_db():
    """Dependency FastAPI: buka session per-request, tutup setelah selesai."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
 