import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

# Download NLTK data
try:
    nltk.download('stopwords')
    nltk.download('punkt')
except:
    pass

class DataProcessor:
    def __init__(self):
        self.doctors_df = None
        self.symptoms_df = None
        self.symptom_desc_df = None
        self.symptom_precaution_df = None
        self.symptom_severity_df = None
        self.specialization_mapping_df = None
        
        self.vectorizer = TfidfVectorizer(max_features=3000, stop_words='english')
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        
        self.specialization_keywords = {}
        self.symptom_keywords = {}
        
    def load_data(self, data_paths):
        """Load all CSV files"""
        try:
            self.doctors_df = pd.read_csv(data_paths['doctors'])
            self.symptoms_df = pd.read_csv(data_paths['dataset'])
            self.symptom_desc_df = pd.read_csv(data_paths['symptom_description'])
            self.symptom_precaution_df = pd.read_csv(data_paths['symptom_precaution'])
            self.symptom_severity_df = pd.read_csv(data_paths['symptom_severity'])
            self.specialization_mapping_df = pd.read_csv(data_paths['specialization_mapping'])
            print("✅ All data files loaded successfully")
            return True
        except Exception as e:
            print(f"❌ Error loading data files: {e}")
            return False
        
    def preprocess_text(self, text):
        """Preprocess text for NLP"""
        if pd.isna(text):
            return ""
        
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        tokens = nltk.word_tokenize(text)
        tokens = [self.stemmer.stem(token) for token in tokens if token not in self.stop_words]
        return ' '.join(tokens)
    
    def preprocess_data(self):
        """Preprocess all data"""
        # Build specialization keywords mapping
        for _, row in self.specialization_mapping_df.iterrows():
            specialty = row['medical_specialty']
            if pd.notna(row['keywords']):
                keywords = str(row['keywords']).split(',')
                self.specialization_keywords[specialty] = [k.strip().lower() for k in keywords]
        
        # Build symptom keywords
        self.symptom_keywords = {}
        for _, row in self.symptoms_df.iterrows():
            disease = row['Disease']
            symptoms = []
            for i in range(1, 10):
                col_name = f'Symptom_{i}'
                if col_name in row and pd.notna(row[col_name]):
                    symptoms.append(str(row[col_name]).strip().lower())
            self.symptom_keywords[disease.lower()] = symptoms
            
        print("✅ Data preprocessing completed")
    
    def extract_fee(self, fee_str):
        """Extract fee from string"""
        if pd.isna(fee_str):
            return float('inf')
        
        fee_str = str(fee_str)
        numbers = re.findall(r'\d+', fee_str)
        return min(map(int, numbers)) if numbers else float('inf')
    
    def get_specialization_from_disease(self, disease):
        """Map disease to medical specialization"""
        disease_lower = disease.lower()
        
        # Check if disease exists in symptom description
        if self.symptom_desc_df is not None and 'Disease' in self.symptom_desc_df.columns:
            disease_exists = any(self.symptom_desc_df['Disease'].str.lower() == disease_lower)
            if disease_exists:
                # Try to find matching specialization
                for spec, keywords in self.specialization_keywords.items():
                    if any(keyword in disease_lower for keyword in keywords):
                        return spec
        
        # Default mapping based on common knowledge
        specialization_map = {
            'fungal infection': 'Dermatology',
            'allergy': 'Allergy / Immunology',
            'malaria': 'Infectious Diseases',
            'common cold': 'General Medicine',
            'pneumonia': 'Pulmonology',
            'migraine': 'Neurology',
            'hypertension': 'Cardiology',
            'diabetes': 'Endocrinology',
            'asthma': 'Pulmonology',
            'arthritis': 'Rheumatology'
        }
        
        return specialization_map.get(disease_lower, 'General Medicine')