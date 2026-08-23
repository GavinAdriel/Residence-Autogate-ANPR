-- =========================================================
-- ANPR AUTOGATE SYSTEM
-- =========================================================

CREATE DATABASE IF NOT EXISTS anpr_system
    CHARACTER SET utf8mb4
    COLLATE utf8mb4_unicode_ci;

USE anpr_system;


-- =========================================================
-- 1. USERS
-- =========================================================

CREATE TABLE IF NOT EXISTS Users (
    User_ID INT AUTO_INCREMENT PRIMARY KEY,
    Username VARCHAR(50) NOT NULL UNIQUE,
    Password_Hash VARCHAR(255) NOT NULL,
    Full_Name VARCHAR(100) NOT NULL,

    Role VARCHAR(20) NOT NULL DEFAULT 'user',

    Is_Active BOOLEAN NOT NULL DEFAULT TRUE,

    Created_At DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT chk_user_role
        CHECK (Role IN ('admin', 'user'))
);


-- =========================================================
-- 2. RESIDENT
-- =========================================================

CREATE TABLE IF NOT EXISTS Resident (
    Resident_ID INT AUTO_INCREMENT PRIMARY KEY,

    Resident_Name VARCHAR(100) NOT NULL,
    Resident_Address VARCHAR(255),
    Resident_Phone_Number VARCHAR(20),

    Created_At DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    Updated_At DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
        ON UPDATE CURRENT_TIMESTAMP
);


-- =========================================================
-- 3. VEHICLE
-- =========================================================

CREATE TABLE IF NOT EXISTS Vehicle (
    Vehicle_ID INT AUTO_INCREMENT PRIMARY KEY,

    License_Plate_Number VARCHAR(20) NOT NULL UNIQUE,
    Normalized_Plate VARCHAR(20) NOT NULL,

    Resident_ID INT NOT NULL,

    Vehicle_Type VARCHAR(50),

    Created_At DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    Updated_At DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
        ON UPDATE CURRENT_TIMESTAMP,

    CONSTRAINT fk_vehicle_resident
        FOREIGN KEY (Resident_ID)
        REFERENCES Resident(Resident_ID)
        ON UPDATE CASCADE
        ON DELETE RESTRICT,

    INDEX idx_vehicle_normalized_plate (Normalized_Plate)
);


-- =========================================================
-- 4. CAMERA
-- =========================================================

CREATE TABLE IF NOT EXISTS Camera (
    Camera_ID INT AUTO_INCREMENT PRIMARY KEY,

    Camera_Name VARCHAR(100) NOT NULL,
    Type VARCHAR(50),
    Location VARCHAR(100),
    IP_Address VARCHAR(50),

    Is_Active BOOLEAN NOT NULL DEFAULT TRUE,

    Created_At DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);


-- =========================================================
-- 5. ANPR LOG
-- =========================================================

CREATE TABLE IF NOT EXISTS ANPR_Log (
    Log_ID INT AUTO_INCREMENT PRIMARY KEY,

    -- Time
    Inserted_Time DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,

    -- Camera
    Camera_ID INT NOT NULL,

    -- OCR result
    License_Plate_Number VARCHAR(20) NOT NULL,
    Normalized_Plate VARCHAR(20) NOT NULL,

    -- Optional guard/manual plate
    Guard_Plate VARCHAR(20),

    -- Vehicle matching
    Vehicle_ID INT NULL,

    -- ANPR result
    Classification VARCHAR(32),
    Direction VARCHAR(20),
    Event_Kind VARCHAR(32),

    -- Gate authorization
    Grant_Method VARCHAR(32),

    -- Entry / Exit state
    Entry_State VARCHAR(32) NOT NULL DEFAULT 'OPEN',

    -- Performance metrics
    Detection_Confidence DECIMAL(5,4),
    OCR_Confidence DECIMAL(5,4),
    Processing_Time_MS DECIMAL(10,2),

    -- Environment
    Environment_Label VARCHAR(32),

    -- Image reference
    Image_Ref VARCHAR(512),

    -- Event that closes this entry
    Closed_By_Log_ID INT NULL,

    CONSTRAINT fk_log_camera
        FOREIGN KEY (Camera_ID)
        REFERENCES Camera(Camera_ID)
        ON UPDATE CASCADE
        ON DELETE RESTRICT,

    CONSTRAINT fk_log_vehicle
        FOREIGN KEY (Vehicle_ID)
        REFERENCES Vehicle(Vehicle_ID)
        ON UPDATE CASCADE
        ON DELETE SET NULL,

    CONSTRAINT fk_log_closed_by
        FOREIGN KEY (Closed_By_Log_ID)
        REFERENCES ANPR_Log(Log_ID)
        ON UPDATE CASCADE
        ON DELETE SET NULL,

    INDEX idx_log_normalized_plate (Normalized_Plate),
    INDEX idx_log_open_entries (
        Normalized_Plate,
        Entry_State,
        Event_Kind
    ),
    INDEX idx_log_camera (Camera_ID),
    INDEX idx_log_vehicle (Vehicle_ID),
    INDEX idx_log_inserted_time (Inserted_Time)
);


-- =========================================================
-- 6. IMAGES
-- =========================================================

CREATE TABLE IF NOT EXISTS Images (
    Image_ID INT AUTO_INCREMENT PRIMARY KEY,

    Log_ID INT NOT NULL,

    Snapshot_Path VARCHAR(512) NOT NULL,
    Thumbnail_Path VARCHAR(512),

    Captured_At DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT fk_images_log
        FOREIGN KEY (Log_ID)
        REFERENCES ANPR_Log(Log_ID)
        ON UPDATE CASCADE
        ON DELETE CASCADE,

    INDEX idx_images_captured_at (Captured_At)
);


-- =========================================================
-- 7. OPTIONAL: SYSTEM SETTINGS
-- =========================================================

CREATE TABLE IF NOT EXISTS System_Settings (
    Setting_ID INT AUTO_INCREMENT PRIMARY KEY,

    Setting_Name VARCHAR(100) NOT NULL UNIQUE,
    Setting_Value VARCHAR(255),

    Updated_At DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
        ON UPDATE CURRENT_TIMESTAMP
);