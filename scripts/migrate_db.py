#!/usr/bin/env python3
"""
Database migration script to add separate columns for patient information
"""

import os
from sqlalchemy import text
from dotenv import load_dotenv
from app.db import engine

load_dotenv()

def migrate_database():
    """Add new patient information columns to existing table"""
    
    migration_sql = """
    ALTER TABLE patient_records 
    ADD COLUMN IF NOT EXISTS patient_name VARCHAR(255),
    ADD COLUMN IF NOT EXISTS patient_phone VARCHAR(20), 
    ADD COLUMN IF NOT EXISTS patient_age VARCHAR(10),
    ADD COLUMN IF NOT EXISTS patient_gender VARCHAR(10),
    ADD COLUMN IF NOT EXISTS onset VARCHAR(100),
    ADD COLUMN IF NOT EXISTS allergies TEXT,
    ADD COLUMN IF NOT EXISTS current_medications TEXT,
    ADD COLUMN IF NOT EXISTS pain_scale VARCHAR(10);
    """
    
    try:
        with engine.connect() as conn:
            # For MySQL, we need to handle the columns one by one since IF NOT EXISTS may not be supported
            columns_to_add = [
                ("patient_name", "VARCHAR(255)"),
                ("patient_phone", "VARCHAR(20)"),
                ("patient_age", "VARCHAR(10)"),
                ("patient_gender", "VARCHAR(10)"),
                ("onset", "VARCHAR(100)"),
                ("allergies", "TEXT"),
                ("current_medications", "TEXT"),
                ("pain_scale", "VARCHAR(10)")
            ]
            
            for column_name, column_type in columns_to_add:
                try:
                    conn.execute(text(f"ALTER TABLE patient_records ADD COLUMN {column_name} {column_type}"))
                    print(f"✓ Added column: {column_name}")
                except Exception as e:
                    if "Duplicate column name" in str(e) or "already exists" in str(e):
                        print(f"- Column {column_name} already exists, skipping")
                    else:
                        print(f"✗ Error adding column {column_name}: {e}")
                        
            conn.commit()
            print("\n✓ Database migration completed successfully!")
            
    except Exception as e:
        print(f"✗ Migration failed: {e}")
        return False
        
    return True

if __name__ == "__main__":
    print("Starting database migration...")
    migrate_database()