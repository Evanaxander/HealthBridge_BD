import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
import joblib
from data_preprocessor import DataProcessor

class SymptomClassifier:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        self.vectorizer = CountVectorizer(max_features=1000)
        self.symptom_columns = None
        self.data_processor = DataProcessor()
        
    def prepare_training_data(self, symptoms_df):
        """Prepare data for ML training"""
        # Create symptom columns
        all_symptoms = set()
        for i in range(1, 10):
            col_name = f'Symptom_{i}'
            if col_name in symptoms_df.columns:
                all_symptoms.update(symptoms_df[col_name].dropna().unique())
        
        self.symptom_columns = list(all_symptoms)
        
        # Create feature matrix
        X = pd.DataFrame(0, index=symptoms_df.index, columns=self.symptom_columns)
        y = symptoms_df['Disease']
        
        for idx, row in symptoms_df.iterrows():
            for i in range(1, 10):
                col_name = f'Symptom_{i}'
                if col_name in row and pd.notna(row[col_name]):
                    symptom = str(row[col_name]).strip()
                    if symptom in self.symptom_columns:
                        X.loc[idx, symptom] = 1
        
        return X, y
    
    def train(self, symptoms_df):
        """Train the model"""
        X, y = self.prepare_training_data(symptoms_df)
        
        # Check if we have enough data
        if len(X) < 2:
            print("❌ Not enough data for training")
            return 0, 0
            
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.model.fit(X_train, y_train)
        
        # Evaluate
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        print(f"Training accuracy: {train_score:.3f}")
        print(f"Test accuracy: {test_score:.3f}")
        
        return train_score, test_score
    
    def predict_symptoms(self, symptom_list):
        """Predict disease from symptoms"""
        if not self.symptom_columns:
            raise ValueError("Model not trained yet")
        
        # Create feature vector
        X_pred = pd.DataFrame(0, index=[0], columns=self.symptom_columns)
        for symptom in symptom_list:
            symptom_clean = symptom.strip().lower()
            for col_symptom in self.symptom_columns:
                if col_symptom.strip().lower() == symptom_clean:
                    X_pred.loc[0, col_symptom] = 1
                    break
        
        try:
            probabilities = self.model.predict_proba(X_pred)[0]
            diseases = self.model.classes_
            
            results = []
            for disease, prob in zip(diseases, probabilities):
                if prob > 0.1:  # Only include diseases with >10% probability
                    results.append({
                        'disease': disease,
                        'probability': float(prob),
                        'confidence': 'High' if prob > 0.7 else 'Medium' if prob > 0.4 else 'Low'
                    })
            
            # Sort by probability
            results.sort(key=lambda x: x['probability'], reverse=True)
            return results[:3]  # Return top 3
        except:
            # Fallback if model prediction fails
            return []
    
    def save_model(self, filepath):
        """Save trained model"""
        joblib.dump({
            'model': self.model,
            'symptom_columns': self.symptom_columns
        }, filepath)
    
    def load_model(self, filepath):
        """Load trained model"""
        data = joblib.load(filepath)
        self.model = data['model']
        self.symptom_columns = data['symptom_columns']

class HealthcareChatbot:
    def __init__(self, data_paths):
        self.data_processor = DataProcessor()
        if not self.data_processor.load_data(data_paths):
            raise Exception("Failed to load data files")
        self.data_processor.preprocess_data()
        
        self.symptom_classifier = SymptomClassifier()
        self.train_model()
    
    def train_model(self):
        """Train the ML model"""
        try:
            train_score, test_score = self.symptom_classifier.train(self.data_processor.symptoms_df)
            if test_score > 0.5:
                print("✅ Symptom classification model trained successfully")
            else:
                print("⚠️  Model accuracy is low, using fallback methods")
        except Exception as e:
            print(f"❌ Error training model: {e}")
    
    def analyze_symptoms(self, user_input):
        """Analyze user symptoms and predict disease"""
        processed_input = self.data_processor.preprocess_text(user_input)
        
        # Extract symptoms from input using severity list
        detected_symptoms = []
        if self.data_processor.symptom_severity_df is not None:
            for symptom_row in self.data_processor.symptom_severity_df.itertuples():
                symptom = str(getattr(symptom_row, 'Symptom', '')).lower()
                if symptom and symptom in processed_input:
                    detected_symptoms.append(symptom)
        
        if detected_symptoms:
            # Use ML model for prediction
            try:
                predictions = self.symptom_classifier.predict_symptoms(detected_symptoms)
                
                if predictions:
                    results = []
                    for pred in predictions:
                        disease = pred['disease']
                        specialization = self.data_processor.get_specialization_from_disease(disease)
                        results.append({
                            'disease': disease,
                            'probability': pred['probability'],
                            'confidence': pred['confidence'],
                            'matched_symptoms': detected_symptoms,
                            'specialization': specialization
                        })
                    
                    return results
            except Exception as e:
                print(f"ML prediction failed: {e}")
        
        # Fallback to keyword matching if ML model fails
        return self._keyword_based_analysis(processed_input, detected_symptoms)
    
    def _keyword_based_analysis(self, processed_input, detected_symptoms):
        """Fallback keyword-based symptom analysis"""
        matched_diseases = []
        
        for disease, symptoms in self.data_processor.symptom_keywords.items():
            symptom_count = sum(1 for symptom in symptoms if symptom in processed_input)
            if symptom_count > 0:
                matched_diseases.append({
                    'disease': disease,
                    'match_score': symptom_count,
                    'matched_symptoms': [s for s in symptoms if s in processed_input],
                    'specialization': self.data_processor.get_specialization_from_disease(disease)
                })
        
        # Sort by match score
        matched_diseases.sort(key=lambda x: x['match_score'], reverse=True)
        
        # Convert to similar format as ML predictions
        results = []
        for disease_info in matched_diseases[:3]:
            results.append({
                'disease': disease_info['disease'],
                'probability': disease_info['match_score'] / 10,  # Approximate probability
                'confidence': 'Medium' if disease_info['match_score'] > 1 else 'Low',
                'matched_symptoms': disease_info['matched_symptoms'],
                'specialization': disease_info['specialization']
            })
        
        return results
    
    def get_disease_info(self, disease):
        """Get disease description and precautions"""
        info = {'description': '', 'precautions': []}
        
        # Get description
        if self.data_processor.symptom_desc_df is not None and 'Disease' in self.data_processor.symptom_desc_df.columns:
            desc_match = self.data_processor.symptom_desc_df[
                self.data_processor.symptom_desc_df['Disease'].str.lower() == disease.lower()
            ]
            if not desc_match.empty and 'Description' in desc_match.columns:
                info['description'] = desc_match.iloc[0]['Description']
        
        if not info['description']:
            info['description'] = "No description available for this condition."
        
        # Get precautions
        if self.data_processor.symptom_precaution_df is not None and 'Disease' in self.data_processor.symptom_precaution_df.columns:
            prec_match = self.data_processor.symptom_precaution_df[
                self.data_processor.symptom_precaution_df['Disease'].str.lower() == disease.lower()
            ]
            if not prec_match.empty:
                precautions = []
                for i in range(1, 5):
                    col_name = f'Precaution_{i}'
                    if col_name in prec_match.columns and pd.notna(prec_match.iloc[0][col_name]):
                        precautions.append(prec_match.iloc[0][col_name])
                info['precautions'] = precautions
        
        if not info['precautions']:
            info['precautions'] = ["No specific precautions available. Please consult a doctor."]
        
        return info
    
    def recommend_doctors(self, specialization, max_fee=None, location=None):
        """Recommend doctors based on specialization and filters"""
        if self.data_processor.doctors_df is None:
            return []
            
        doctors = self.data_processor.doctors_df.copy()
        
        # Filter by specialization
        if specialization:
            doctors = doctors[doctors['Specialization'].str.contains(specialization, case=False, na=False)]
        
        # Filter by fee
        if max_fee is not None:
            doctors['min_fee'] = doctors['Fees'].apply(self.data_processor.extract_fee)
            doctors = doctors[doctors['min_fee'] <= max_fee]
        
        # Filter by location (simple implementation)
        if location:
            location_lower = location.lower()
            doctors = doctors[doctors['Address'].str.lower().str.contains(location_lower, na=False)]
        
        return doctors.head(3).to_dict('records')
    
    def process_query(self, user_input):
        """Main method to process user queries"""
        response = {
            'type': None,
            'data': None,
            'message': None
        }
        
        input_lower = user_input.lower()
        
        # Check if it's a symptom query
        symptom_keywords = ['symptom', 'pain', 'itch', 'rash', 'fever', 'headache', 
                           'not feeling well', 'hurt', 'unwell', 'sick', 'nauseous',
                           'dizzy', 'cough', 'cold', 'sore throat']
        if any(keyword in input_lower for keyword in symptom_keywords):
            diseases = self.analyze_symptoms(user_input)
            if diseases:
                response['type'] = 'symptom_analysis'
                response['data'] = []
                
                for disease_info in diseases:
                    disease = disease_info['disease']
                    disease_data = {
                        'disease': disease,
                        'specialization': disease_info['specialization'],
                        'matched_symptoms': disease_info['matched_symptoms'],
                        'confidence': disease_info['confidence'],
                        'info': self.get_disease_info(disease),
                        'recommended_doctors': self.recommend_doctors(disease_info['specialization'])
                    }
                    response['data'].append(disease_data)
            else:
                response['type'] = 'message'
                response['message'] = "I couldn't identify specific symptoms. Please describe how you're feeling in more detail."
        
        # Check if it's a doctor search query
        elif any(keyword in input_lower for keyword in ['doctor', 'specialist', 'appointment', 'consult', 'dr.', 'medical']):
            # Extract specialization from query
            found_specialization = None
            for spec, keywords in self.data_processor.specialization_keywords.items():
                if any(keyword in input_lower for keyword in keywords):
                    found_specialization = spec
                    break
            
            # If no specialization found, try to extract from common terms
            if not found_specialization:
                specialization_map = {
                    'skin': 'Dermatology',
                    'heart': 'Cardiology',
                    'bone': 'Orthopedics',
                    'child': 'Pediatrics',
                    'brain': 'Neurology',
                    'stomach': 'Gastroenterology',
                    'eye': 'Ophthalmology',
                    'teeth': 'Dentistry',
                    'mental': 'Psychiatry',
                    'pregnancy': 'Gynecology',
                    'lung': 'Pulmonology',
                    'allergy': 'Allergy / Immunology',
                    'diabetes': 'Endocrinology'
                }
                
                for term, spec in specialization_map.items():
                    if term in input_lower:
                        found_specialization = spec
                        break
            
            if found_specialization:
                # Extract fee limit if mentioned
                fee_match = re.search(r'(\d+)\s*(tk|taka|fee|price|cost)', input_lower)
                max_fee = int(fee_match.group(1)) if fee_match else None
                
                # Extract location if mentioned
                location = None
                locations = ['dhaka', 'uttara', 'dhanmondi', 'gulshan', 'banani', 'mirpur']  # Add more locations
                for loc in locations:
                    if loc in input_lower:
                        location = loc
                        break
                
                doctors = self.recommend_doctors(found_specialization, max_fee, location)
                response['type'] = 'doctor_recommendation'
                response['data'] = {
                    'specialization': found_specialization,
                    'doctors': doctors
                }
            else:
                response['type'] = 'message'
                response['message'] = "Please specify what type of specialist you're looking for (e.g., dermatologist, cardiologist)."
        
        else:
            response['type'] = 'message'
            response['message'] = "I can help you with symptom analysis or finding doctors. Please tell me how you're feeling or what type of doctor you need."
        
        return response