# verify_files.py
import os
import pandas as pd
import numpy as np

def verify_files():
    data_dir = 'data/'
    
    # Check if data directory exists
    if not os.path.exists(data_dir):
        print(f"❌ Data directory '{data_dir}' does not exist!")
        return False
    
    print(f"✅ Data directory '{data_dir}' exists")
    
    # List all files in data directory
    files = os.listdir(data_dir)
    print(f"\n📁 Files in data directory:")
    for file in files:
        file_path = os.path.join(data_dir, file)
        size = os.path.getsize(file_path)
        print(f"   {file} ({size} bytes)")
    
    # Try to read each CSV file
    print(f"\n🔍 Testing CSV file readability:")
    for file in files:
        if file.endswith('.csv'):
            file_path = os.path.join(data_dir, file)
            try:
                df = pd.read_csv(file_path)
                print(f"   ✅ {file}: {len(df)} rows, {len(df.columns)} columns")
                print(f"      Columns: {list(df.columns)}")
                print(f"      First few rows:")
                print(df.head(2).to_string(index=False))
                print()
            except Exception as e:
                print(f"   ❌ {file}: Error reading - {e}")
                print()
    
    return True

if __name__ == "__main__":
    verify_files()