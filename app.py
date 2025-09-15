# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv
import pandas as pd
import sys

load_dotenv()

app = Flask(__name__)
CORS(app)

# Define data paths
data_paths = {
    'doctors': 'data/doctors.csv',
    'dataset': 'data/dataset.csv',
    'symptom_description': 'data/symptom_Description.csv',
    'symptom_precaution': 'data/symptom_precaution.csv',
    'symptom_severity': 'data/Symptom-severity.csv',
    'specialization_mapping': 'data/symptoms_specialization_mapping.csv'
}

# Check if all files exist
print("Checking required data files...")
missing_files = []
for key, path in data_paths.items():
    if not os.path.exists(path):
        missing_files.append(path)
        print(f"❌ Missing: {path}")

if missing_files:
    print(f"\n❌ {len(missing_files)} files are missing. Creating sample data...")
    
    # Create sample data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Create sample doctors.csv
    if not os.path.exists(data_paths['doctors']):
        pd.DataFrame({
            'Doctor Name': ['Dr. John Smith', 'Dr. Sarah Johnson', 'Dr. Mike Wilson'],
            'Specialization': ['Dermatology', 'Cardiology', 'Neurology'],
            'Doctor Info': ['MBBS, MD', 'MBBS, DM', 'MBBS, DM'],
            'Hospital/Chamber': ['ABC Hospital', 'XYZ Hospital', 'PQR Hospital'],
            'Address': ['123 Main St', '456 Oak St', '789 Pine St'],
            'Appointment Days': ['Mon-Wed', 'Tue-Thu', 'Wed-Fri'],
            'Fees': ['1000 TK', '1500 TK', '1200 TK']
        }).to_csv(data_paths['doctors'], index=False)
        print(f"✓ Created sample: {data_paths['doctors']}")
    
    # Create sample dataset.csv
    if not os.path.exists(data_paths['dataset']):
        pd.DataFrame({
            'Disease': ['Fungal infection', 'Allergy', 'Malaria'],
            'Symptom_1': ['itching', 'itching', 'chills'],
            'Symptom_2': ['skin_rash', 'skin_rash', 'vomiting'],
            'Symptom_3': ['nodal_skin_eruptions', 'continuous_sneezing', 'high_fever'],
            'Symptom_4': ['dishromic_patches', None, 'sweating'],
            'Symptom_5': [None, None, None],
            'Symptom_6': [None, None, None],
            'Symptom_7': [None, None, None],
            'Symptom_8': [None, None, None],
            'Symptom_9': [None, None, None]
        }).to_csv(data_paths['dataset'], index=False)
        print(f"✓ Created sample: {data_paths['dataset']}")
    
    # Create other sample files similarly...
    print("Sample data created. Please replace with your actual data files.")

# Initialize chatbot
try:
    from ml_models import HealthcareChatbot
    chatbot = HealthcareChatbot(data_paths)
    print("Healthcare chatbot initialized successfully")
except Exception as e:
    print(f"Error initializing chatbot: {e}")
    print("Please check if all data files exist in the data/ directory")
    chatbot = None
    # Optionally exit if chatbot is essential
    # sys.exit(1)

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        if not chatbot:
            return jsonify({'error': 'Chatbot not initialized'}), 500
            
        data = request.json
        user_input = data.get('message', '')
        
        if not user_input:
            return jsonify({'error': 'No message provided'}), 400
        
        response = chatbot.process_query(user_input)
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/doctors', methods=['GET'])
def get_doctors():
    try:
        if not chatbot:
            return jsonify({'error': 'Chatbot not initialized'}), 500
            
        specialization = request.args.get('specialization')
        max_fee = request.args.get('max_fee', type=int)
        location = request.args.get('location')
        
        doctors = chatbot.recommend_doctors(specialization, max_fee, location)
        return jsonify({'doctors': doctors})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    status = 'healthy' if chatbot else 'unhealthy'
    return jsonify({'status': status})

@app.route('/api/test', methods=['GET'])
def test_endpoint():
    """Simple test endpoint to verify server is running"""
    return jsonify({
        'message': 'Server is running!',
        'chatbot_initialized': chatbot is not None
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting healthcare chatbot server on port {port}...")
    app.run(debug=True, host='0.0.0.0', port=port)