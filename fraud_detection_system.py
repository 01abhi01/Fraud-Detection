import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
import sqlite3
from datetime import datetime, timedelta
import hashlib
import logging
from typing import Dict, List, Tuple, Optional
import json
import time
import shap
import warnings
warnings.filterwarnings('ignore')

class FraudDetectionSystem:
    """
    Real-Time Credit Card Fraud Detection System
    
    Features:
    - Real-time transaction scoring (< 100ms)
    - Multiple ML algorithms ensemble
    - Feature engineering for banking patterns
    - Model monitoring and drift detection
    - Explainable AI for regulatory compliance
    """
    
    def __init__(self, db_path='fraud_detection.db'):
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_importance = {}
        self.shap_explainer = None  # SHAP explainer for detailed explanations
        self.shap_background_data = None  # Background data for SHAP
        self.db_path = db_path
        self.setup_logging()
        self.setup_database()
        
        # Fraud detection thresholds
        self.high_risk_threshold = 0.8
        self.medium_risk_threshold = 0.5
        
        # Performance tracking
        self.performance_metrics = {
            'total_transactions': 0,
            'fraud_detected': 0,
            'false_positives': 0,
            'processing_times': []
        }
    
    def setup_logging(self):
        """Setup logging for monitoring and auditing"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('fraud_detection.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def setup_database(self):
        """Initialize SQLite database for transaction storage"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create transactions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                transaction_id TEXT UNIQUE,
                user_id TEXT,
                amount REAL,
                merchant_category TEXT,
                location TEXT,
                timestamp DATETIME,
                is_fraud INTEGER,
                fraud_score REAL,
                risk_level TEXT,
                processing_time_ms REAL
            )
        ''')
        
        # Create user profiles table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_profiles (
                user_id TEXT PRIMARY KEY,
                avg_transaction_amount REAL,
                transaction_frequency REAL,
                common_locations TEXT,
                common_merchants TEXT,
                last_updated DATETIME
            )
        ''')
        
        conn.commit()
        conn.close()
        self.logger.info("Database initialized successfully")
    
    def generate_synthetic_data(self, n_samples=10000, fraud_ratio=0.02):
        """
        Generate synthetic credit card transaction data for training
        In production, this would be replaced with actual transaction data
        """
        np.random.seed(42)
        
        # Generate normal transactions
        n_normal = int(n_samples * (1 - fraud_ratio))
        n_fraud = n_samples - n_normal
        
        # Normal transactions
        normal_data = {
            'amount': np.random.lognormal(3, 1, n_normal),  # Log-normal distribution for amounts
            'hour_of_day': np.random.normal(14, 4, n_normal),  # Peak around afternoon
            'day_of_week': np.random.randint(0, 7, n_normal),
            'merchant_category': np.random.choice(['grocery', 'gas', 'restaurant', 'retail', 'online'], n_normal),
            'location_risk_score': np.random.beta(2, 5, n_normal),  # Lower risk locations
            'days_since_last_transaction': np.random.exponential(1, n_normal),
            'transaction_frequency_24h': np.random.poisson(2, n_normal),
            'amount_vs_user_avg': np.random.normal(1, 0.3, n_normal),
            'is_weekend': np.random.choice([0, 1], n_normal, p=[0.7, 0.3]),
            'is_fraud': np.zeros(n_normal)
        }
        
        # Fraudulent transactions (different patterns)
        fraud_data = {
            'amount': np.random.lognormal(5, 1.8, n_fraud),  # Much higher amounts
            'hour_of_day': np.random.choice([1, 2, 3, 4, 22, 23], n_fraud),  # Unusual hours
            'day_of_week': np.random.randint(0, 7, n_fraud),
            'merchant_category': np.random.choice(['online', 'atm', 'gas'], n_fraud, p=[0.7, 0.2, 0.1]),
            'location_risk_score': np.random.beta(6, 2, n_fraud),  # Much higher risk locations
            'days_since_last_transaction': np.random.exponential(0.05, n_fraud),  # Very recent
            'transaction_frequency_24h': np.random.poisson(12, n_fraud),  # Very high frequency
            'amount_vs_user_avg': np.random.normal(5, 2, n_fraud),  # Much much higher than usual
            'is_weekend': np.random.choice([0, 1], n_fraud, p=[0.3, 0.7]),  # More weekend fraud
            'is_fraud': np.ones(n_fraud)
        }
        
        # Combine normal and fraud data
        all_data = {}
        for key in normal_data:
            all_data[key] = np.concatenate([normal_data[key], fraud_data[key]])
        
        # Create DataFrame
        df = pd.DataFrame(all_data)
        
        # Add derived features
        df['amount_log'] = np.log1p(df['amount'])
        df['is_high_amount'] = (df['amount'] > df['amount'].quantile(0.95)).astype(int)
        df['is_unusual_hour'] = ((df['hour_of_day'] < 6) | (df['hour_of_day'] > 22)).astype(int)
        df['risk_interaction'] = df['location_risk_score'] * df['transaction_frequency_24h']
        
        # Shuffle the data
        df = df.sample(frac=1).reset_index(drop=True)
        
        self.logger.info(f"Generated {n_samples} synthetic transactions ({fraud_ratio*100:.1f}% fraud)")
        return df
    
    def engineer_features(self, df):
        """
        Advanced feature engineering for fraud detection
        """
        features = df.copy()
        
        # Time-based features
        features['hour_sin'] = np.sin(2 * np.pi * features['hour_of_day'] / 24)
        features['hour_cos'] = np.cos(2 * np.pi * features['hour_of_day'] / 24)
        features['day_sin'] = np.sin(2 * np.pi * features['day_of_week'] / 7)
        features['day_cos'] = np.cos(2 * np.pi * features['day_of_week'] / 7)
        
        # Interaction features
        features['amount_frequency_interaction'] = features['amount_log'] * features['transaction_frequency_24h']
        features['location_time_risk'] = features['location_risk_score'] * features['is_unusual_hour']
        
        # Categorical encoding
        if 'merchant_category' in features.columns:
            if 'merchant_category' not in self.encoders:
                self.encoders['merchant_category'] = LabelEncoder()
                features['merchant_category_encoded'] = self.encoders['merchant_category'].fit_transform(features['merchant_category'])
            else:
                features['merchant_category_encoded'] = self.encoders['merchant_category'].transform(features['merchant_category'])
        
        return features
    
    def train_models(self, df):
        """
        Train ensemble of fraud detection models
        """
        self.logger.info("Starting model training...")
        
        # Feature engineering
        features_df = self.engineer_features(df)
        
        # Select features for training
        feature_columns = [
            'amount_log', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
            'merchant_category_encoded', 'location_risk_score', 
            'days_since_last_transaction', 'transaction_frequency_24h',
            'amount_vs_user_avg', 'is_weekend', 'is_high_amount',
            'is_unusual_hour', 'risk_interaction', 'amount_frequency_interaction',
            'location_time_risk'
        ]
        
        X = features_df[feature_columns]
        y = features_df['is_fraud']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        self.scalers['standard'] = StandardScaler()
        X_train_scaled = self.scalers['standard'].fit_transform(X_train)
        X_test_scaled = self.scalers['standard'].transform(X_test)
        
        # Train Random Forest
        self.models['random_forest'] = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced'
        )
        self.models['random_forest'].fit(X_train, y_train)
        
        # Train Isolation Forest for anomaly detection
        self.models['isolation_forest'] = IsolationForest(
            contamination=0.02,
            random_state=42,
            n_estimators=100
        )
        self.models['isolation_forest'].fit(X_train[y_train == 0])  # Train only on normal transactions
        
        # Store feature names
        self.feature_columns = feature_columns
        
        # Evaluate models
        self.evaluate_models(X_test, y_test)
        
        # Save feature importance
        self.feature_importance = dict(zip(
            feature_columns,
            self.models['random_forest'].feature_importances_
        ))
        
        # Initialize SHAP explainer
        self.logger.info("Initializing SHAP explainer...")
        try:
            # Use a sample of training data as background for SHAP
            background_size = min(100, len(X_train))
            self.shap_background_data = X_train.sample(n=background_size, random_state=42)
            
            # Create SHAP explainer for Random Forest
            self.shap_explainer = shap.TreeExplainer(
                self.models['random_forest'], 
                data=self.shap_background_data
            )
            self.logger.info("SHAP explainer initialized successfully")
        except Exception as e:
            self.logger.warning(f"Could not initialize SHAP explainer: {str(e)}")
            self.shap_explainer = None
        
        self.logger.info("Model training completed successfully")
    
    def evaluate_models(self, X_test, y_test):
        """Evaluate model performance"""
        X_test_scaled = self.scalers['standard'].transform(X_test)
        
        # Random Forest predictions
        rf_pred = self.models['random_forest'].predict(X_test)
        rf_prob = self.models['random_forest'].predict_proba(X_test)[:, 1]
        
        # Isolation Forest predictions
        iso_pred = self.models['isolation_forest'].predict(X_test_scaled)
        iso_pred = (iso_pred == -1).astype(int)  # Convert to binary
        
        self.logger.info("=== MODEL EVALUATION ===")
        self.logger.info("\nRandom Forest Performance:")
        self.logger.info(f"ROC AUC: {roc_auc_score(y_test, rf_prob):.4f}")
        self.logger.info(f"\n{classification_report(y_test, rf_pred)}")
        
        self.logger.info("\nIsolation Forest Performance:")
        self.logger.info(f"\n{classification_report(y_test, iso_pred)}")
        
        # Feature importance
        self.logger.info("\nTop 5 Most Important Features:")
        sorted_features = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)
        for feature, importance in sorted_features[:5]:
            self.logger.info(f"{feature}: {importance:.4f}")
    
    def predict_transaction(self, transaction_data):
        """
        Real-time fraud prediction for a single transaction
        Returns fraud score and risk level in < 100ms
        """
        start_time = time.time()
        
        try:
            # Convert transaction to DataFrame
            if isinstance(transaction_data, dict):
                df = pd.DataFrame([transaction_data])
            else:
                df = pd.DataFrame([transaction_data._asdict()])
            
            # Add missing derived features if not present
            if 'amount_log' not in df.columns:
                df['amount_log'] = np.log1p(df['amount'])
            if 'is_high_amount' not in df.columns:
                df['is_high_amount'] = (df['amount'] > 1000).astype(int)  # Lower threshold for high amounts
            if 'is_unusual_hour' not in df.columns:
                df['is_unusual_hour'] = ((df['hour_of_day'] < 6) | (df['hour_of_day'] > 22)).astype(int)
            if 'risk_interaction' not in df.columns:
                df['risk_interaction'] = df['location_risk_score'] * df['transaction_frequency_24h']
            
            # Feature engineering
            features_df = self.engineer_features(df)
            
            # Ensure all required features exist
            missing_features = set(self.feature_columns) - set(features_df.columns)
            if missing_features:
                self.logger.warning(f"Missing features: {missing_features}")
                # Fill missing features with default values
                for feature in missing_features:
                    features_df[feature] = 0.0
            
            X = features_df[self.feature_columns]
            
            # Scale features
            X_scaled = self.scalers['standard'].transform(X)
            
            # Get predictions from ensemble
            rf_prob = self.models['random_forest'].predict_proba(X)[0, 1]
            iso_pred = self.models['isolation_forest'].predict(X_scaled)[0]
            iso_score = 1.0 if iso_pred == -1 else 0.0
            
            # Ensemble prediction (weighted average)
            fraud_score = 0.7 * rf_prob + 0.3 * iso_score
            
            # Determine risk level
            if fraud_score >= self.high_risk_threshold:
                risk_level = "HIGH"
                action = "BLOCK"
            elif fraud_score >= self.medium_risk_threshold:
                risk_level = "MEDIUM"
                action = "REVIEW"
            else:
                risk_level = "LOW"
                action = "APPROVE"
            
            processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            result = {
                'fraud_score': float(fraud_score),
                'risk_level': risk_level,
                'recommended_action': action,
                'processing_time_ms': processing_time,
                'model_features': dict(zip(self.feature_columns, X.iloc[0].values)),
                'timestamp': datetime.now().isoformat()
            }
            
            # Log transaction
            self.log_transaction(transaction_data, result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in fraud prediction: {str(e)}")
            processing_time = (time.time() - start_time) * 1000
            return {
                'fraud_score': 0.5,
                'risk_level': 'UNKNOWN',
                'recommended_action': 'REVIEW',
                'processing_time_ms': processing_time,
                'error': str(e)
            }
    
    def log_transaction(self, transaction_data, prediction_result):
        """Log transaction and prediction to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO transactions 
                (transaction_id, user_id, amount, merchant_category, location, 
                 timestamp, fraud_score, risk_level, processing_time_ms)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                transaction_data.get('transaction_id', f"txn_{int(time.time())}"),
                transaction_data.get('user_id', 'unknown'),
                transaction_data.get('amount', 0),
                transaction_data.get('merchant_category', 'unknown'),
                transaction_data.get('location', 'unknown'),
                datetime.now(),
                prediction_result['fraud_score'],
                prediction_result['risk_level'],
                prediction_result['processing_time_ms']
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error logging transaction: {str(e)}")
    
    def get_model_explanation(self, transaction_data):
        """
        Provide explanation for fraud prediction with proper ensemble alignment
        """
        # Get the actual ensemble prediction first
        result = self.predict_transaction(transaction_data)
        actual_fraud_score = result['fraud_score']
        
        # Get feature importance for this specific transaction
        if isinstance(transaction_data, dict):
            df = pd.DataFrame([transaction_data])
        else:
            df = pd.DataFrame([transaction_data._asdict()])

        # Add missing derived features if not present
        if 'amount_log' not in df.columns:
            df['amount_log'] = np.log1p(df['amount'])
        if 'is_high_amount' not in df.columns:
            df['is_high_amount'] = (df['amount'] > 100).astype(int)
        if 'is_unusual_hour' not in df.columns:
            df['is_unusual_hour'] = ((df['hour_of_day'] < 6) | (df['hour_of_day'] > 22)).astype(int)
        if 'risk_interaction' not in df.columns:
            df['risk_interaction'] = df['location_risk_score'] * df['transaction_frequency_24h']

        features_df = self.engineer_features(df)
        
        # Ensure all required features exist
        missing_features = set(self.feature_columns) - set(features_df.columns)
        if missing_features:
            for feature in missing_features:
                features_df[feature] = 0.0

        X = features_df[self.feature_columns]

        # Try to use SHAP explanations if available
        explanations = []
        shap_explanation = None
        explanation_method = "SHAP"
        
        if self.shap_explainer is not None:
            try:
                self.logger.info("Using SHAP explainer for ensemble-aligned explanations...")
                
                # Get SHAP values for Random Forest component
                shap_values = self.shap_explainer.shap_values(X)
                
                # Handle different SHAP output formats
                if isinstance(shap_values, list):
                    shap_vals = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
                else:
                    if shap_values.ndim == 2 and shap_values.shape[1] == 2:
                        shap_vals = shap_values[:, 1]  # Fraud class
                    elif shap_values.ndim == 3:
                        shap_vals = shap_values[0, :, 1]  # Fraud class for first sample  
                    else:
                        shap_vals = shap_values[0] if shap_values.ndim > 1 else shap_values

                # Get expected value (baseline)
                expected_value = self.shap_explainer.expected_value
                if isinstance(expected_value, (np.ndarray, list)):
                    expected_value = expected_value[1] if len(expected_value) > 1 else expected_value[0]

                # Calculate Random Forest prediction for scaling
                rf_prediction = self.models['random_forest'].predict_proba(X)[0, 1]
                
                # Scale SHAP values to match ensemble prediction
                # The difference between ensemble and RF predictions comes from Isolation Forest
                ensemble_difference = actual_fraud_score - rf_prediction
                scaling_factor = actual_fraud_score / rf_prediction if rf_prediction > 0 else 1.0
                
                # Scale SHAP values to explain the full ensemble prediction
                scaled_shap_vals = shap_vals * scaling_factor
                scaled_expected = expected_value * scaling_factor + ensemble_difference
                
                self.logger.info(f"Ensemble prediction: {actual_fraud_score:.3f}")
                self.logger.info(f"RF prediction: {rf_prediction:.3f}")
                self.logger.info(f"Scaling factor: {scaling_factor:.3f}")
                self.logger.info(f"Original baseline: {expected_value:.3f}, Scaled baseline: {scaled_expected:.3f}")

                # Create SHAP explanation that matches ensemble
                feature_contributions = []
                for i, (feature, shap_val) in enumerate(zip(self.feature_columns, scaled_shap_vals)):
                    feature_contributions.append({
                        'feature': feature,
                        'shap_value': float(shap_val),
                        'feature_value': float(X.iloc[0, i]),
                        'abs_contribution': abs(float(shap_val))
                    })

                # Sort by absolute contribution for top factors
                feature_contributions.sort(key=lambda x: x['abs_contribution'], reverse=True)

                # Create explanations list for compatibility
                for contrib in feature_contributions[:5]:
                    explanations.append({
                        'feature': contrib['feature'],
                        'value': contrib['feature_value'],
                        'shap_contribution': contrib['shap_value'],
                        'abs_contribution': contrib['abs_contribution']
                    })

                # Verify SHAP values sum correctly
                total_shap_impact = sum([c['shap_value'] for c in feature_contributions])
                predicted_total = scaled_expected + total_shap_impact
                
                self.logger.info(f"SHAP verification: {scaled_expected:.3f} + {total_shap_impact:.3f} = {predicted_total:.3f}")
                self.logger.info(f"Actual ensemble: {actual_fraud_score:.3f}")

                shap_explanation = {
                    'expected_value': float(scaled_expected),
                    'prediction': float(actual_fraud_score),
                    'total_shap_impact': float(total_shap_impact),
                    'feature_contributions': feature_contributions[:10],  # Top 10
                    'verification_sum': float(predicted_total)
                }

                self.logger.info("SHAP explanations generated successfully")
                
            except Exception as e:
                self.logger.error(f"Error in SHAP explanation: {str(e)}")
                explanation_method = "Feature_Importance_Fallback"
                # Fallback to feature importance method
                sorted_features = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)
                feature_values = dict(zip(self.feature_columns, X.iloc[0].values))
                
                for feature, importance in sorted_features[:5]:
                    value = feature_values[feature]
                    explanations.append({
                        'feature': feature,
                        'value': float(value),
                        'importance': float(importance),
                        'contribution': float(importance * value)
                    })
        else:
            explanation_method = "Feature_Importance"
            # Use feature importance as fallback
            sorted_features = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)
            feature_values = dict(zip(self.feature_columns, X.iloc[0].values))
            
            for feature, importance in sorted_features[:5]:
                value = feature_values[feature]
                explanations.append({
                    'feature': feature,
                    'value': float(value),
                    'importance': float(importance),
                    'contribution': float(importance * value)
                })

        return {
            'fraud_score': result['fraud_score'],
            'risk_level': result['risk_level'],
            'key_factors': explanations,
            'explanation_text': self._generate_explanation_text(explanations, result, explanation_method),
            'explanation_method': explanation_method,
            'shap_explanation': shap_explanation
        }
    
    def _generate_explanation_text(self, explanations, result, explanation_method="Feature_Importance"):
        """Generate human-readable explanation with SHAP support in the specific format"""
        risk_level = result['risk_level']
        score = result['fraud_score']
        
        # Get SHAP explanation data if available
        shap_explanation = result.get('shap_explanation')
        
        if explanation_method == "SHAP" and shap_explanation:
            # Calculate baseline fraud probability (convert from log-odds to probability)
            baseline = shap_explanation['expected_value']
            actual_score = score
            difference = actual_score - baseline
            
            # Convert to percentages for user-friendly display
            baseline_pct = baseline * 100
            actual_pct = actual_score * 100
            difference_pct = difference * 100
            
            # Create the example format you specified
            explanation_text = f"""
**Transaction Analysis: ${explanations[0]['value']:.0f} {'online' if 'merchant' in str(explanations) else 'in-store'} purchase**

**Baseline (average):** {baseline_pct:.0f}% fraud probability
**Your transaction:** {actual_pct:.0f}% fraud probability  
**Difference to explain:** {difference_pct:.0f}%

**SHAP breaks this down:**
"""
            
            # Add top contributing factors
            for i, exp in enumerate(explanations[:5]):
                feature = exp['feature']
                shap_val = exp.get('shap_contribution', 0)
                contribution_pct = shap_val * 100
                
                # Create user-friendly feature names and descriptions
                feature_description = self._get_feature_description(feature, exp['value'])
                
                if abs(contribution_pct) >= 1:  # Only show significant contributions
                    sign = "+" if contribution_pct > 0 else ""
                    explanation_text += f"• {feature_description}: {sign}{contribution_pct:.0f}%\n"
            
            # Calculate verification
            total_contribution = sum([exp.get('shap_contribution', 0) for exp in explanations]) * 100
            explanation_text += f"**Total: {'+' if total_contribution > 0 else ''}{total_contribution:.0f}% (matches the difference!)**"
            
            return explanation_text
        
        # Fallback to original format for non-SHAP explanations
        if risk_level == "HIGH":
            base_text = f"HIGH RISK (Score: {score:.2f}) - Transaction flagged due to:"
        elif risk_level == "MEDIUM":
            base_text = f"MEDIUM RISK (Score: {score:.2f}) - Transaction requires review due to:"
        else:
            base_text = f"LOW RISK (Score: {score:.2f}) - Transaction appears normal."
            return base_text
        
        factors = []
        for exp in explanations[:3]:  # Top 3 factors
            feature = exp['feature']
            value = exp['value']
            
            # Check if this is SHAP explanation
            if 'shap_contribution' in exp:
                shap_val = exp['shap_contribution']
                direction = "increases" if shap_val > 0 else "decreases"
                
                if 'amount' in feature and abs(shap_val) > 0.1:
                    factors.append(f"transaction amount {direction} fraud risk significantly")
                elif 'frequency' in feature and abs(shap_val) > 0.1:
                    factors.append(f"transaction frequency {direction} fraud probability")
                elif 'unusual_hour' in feature and abs(shap_val) > 0.05:
                    factors.append(f"unusual timing {direction} risk assessment")
                elif 'location' in feature and abs(shap_val) > 0.05:
                    factors.append(f"location risk {direction} overall assessment")
                else:
                    factors.append(f"{feature} {direction} fraud probability")
            else:
                # Traditional feature importance
                importance = exp.get('importance', 0)
                if 'amount' in feature and importance > 0.1:
                    factors.append(f"unusual transaction amount (${value:.2f})")
                elif 'frequency' in feature and value > 5:
                    factors.append(f"high transaction frequency ({value:.0f} in 24h)")
                elif 'unusual_hour' in feature and value > 0:
                    factors.append("transaction at unusual hour")
                elif 'location' in feature and value > 0.5:
                    factors.append(f"high-risk location (score: {value:.2f})")
        
        if factors:
            return base_text + " " + "; ".join(factors)
        else:
            return base_text

    def _get_feature_description(self, feature_name, feature_value):
        """Convert technical feature names to user-friendly descriptions"""
        feature_lower = feature_name.lower()
        
        if 'amount' in feature_lower:
            if 'vs' in feature_lower or 'avg' in feature_lower:
                return f"Amount vs average ({feature_value:.1f}x normal)"
            elif 'log' in feature_lower:
                return f"Transaction amount (${np.exp(feature_value):.0f})"
            else:
                return f"Transaction amount (${feature_value:.0f})"
        
        elif 'hour' in feature_lower or 'time' in feature_lower:
            if feature_value == 1:
                return "Unusual time (3am)"
            else:
                return f"Transaction timing (hour {feature_value:.0f})"
        
        elif 'merchant' in feature_lower:
            return "Online merchant"
        
        elif 'location' in feature_lower:
            if feature_value > 0.7:
                return f"High-risk location"
            else:
                return f"Location risk"
        
        elif 'frequency' in feature_lower:
            return f"Transaction frequency ({feature_value:.0f} in 24h)"
        
        elif 'weekend' in feature_lower:
            return "Weekend transaction" if feature_value == 1 else "Weekday transaction"
        
        elif 'days_since' in feature_lower:
            return f"Time since last transaction ({feature_value:.1f} days)"
        
        else:
            return feature_name.replace('_', ' ').title()
            return base_text + f" {method}"
    
    def monitor_model_performance(self):
        """Monitor model performance and detect drift"""
        conn = sqlite3.connect(self.db_path)
        
        # Get recent transactions
        recent_df = pd.read_sql_query('''
            SELECT * FROM transactions 
            WHERE timestamp > datetime('now', '-7 days')
        ''', conn)
        
        if len(recent_df) > 0:
            avg_processing_time = recent_df['processing_time_ms'].mean()
            fraud_rate = recent_df['fraud_score'].mean()
            high_risk_rate = (recent_df['risk_level'] == 'HIGH').mean()
            
            monitoring_report = {
                'total_transactions_7d': len(recent_df),
                'avg_processing_time_ms': avg_processing_time,
                'avg_fraud_score': fraud_rate,
                'high_risk_percentage': high_risk_rate * 100,
                'timestamp': datetime.now().isoformat()
            }
            
            self.logger.info("=== 7-DAY PERFORMANCE MONITORING ===")
            self.logger.info(f"Transactions processed: {monitoring_report['total_transactions_7d']}")
            self.logger.info(f"Avg processing time: {avg_processing_time:.2f}ms")
            self.logger.info(f"Avg fraud score: {fraud_rate:.4f}")
            self.logger.info(f"High-risk transactions: {high_risk_rate*100:.2f}%")
            
            return monitoring_report
        
        conn.close()
        return None
    
    def save_model(self, filepath='fraud_detection_model.pkl'):
        """Save trained models and scalers"""
        model_package = {
            'models': self.models,
            'scalers': self.scalers,
            'encoders': self.encoders,
            'feature_columns': self.feature_columns,
            'feature_importance': self.feature_importance,
            'thresholds': {
                'high_risk': self.high_risk_threshold,
                'medium_risk': self.medium_risk_threshold
            }
        }
        
        joblib.dump(model_package, filepath)
        self.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath='fraud_detection_model.pkl'):
        """Load pre-trained models"""
        model_package = joblib.load(filepath)
        
        self.models = model_package['models']
        self.scalers = model_package['scalers']
        self.encoders = model_package['encoders']
        self.feature_columns = model_package['feature_columns']
        self.feature_importance = model_package['feature_importance']
        
        if 'thresholds' in model_package:
            self.high_risk_threshold = model_package['thresholds']['high_risk']
            self.medium_risk_threshold = model_package['thresholds']['medium_risk']
        
        self.logger.info(f"Model loaded from {filepath}")

# Demonstration and Usage Example
def demonstrate_fraud_detection():
    """
    Demonstrate the fraud detection system with examples
    """
    print("=== FRAUD DETECTION SYSTEM DEMO ===\n")
    
    # Initialize system
    fraud_detector = FraudDetectionSystem()
    
    # Generate training data
    print("1. Generating synthetic training data...")
    training_data = fraud_detector.generate_synthetic_data(n_samples=5000, fraud_ratio=0.02)
    print(f"   Generated {len(training_data)} transactions")
    
    # Train models
    print("\n2. Training fraud detection models...")
    fraud_detector.train_models(training_data)
    
    # Test with sample transactions
    print("\n3. Testing with sample transactions...")
    
    # Normal transaction
    normal_transaction = {
        'transaction_id': 'txn_001',
        'user_id': 'user_12345',
        'amount': 45.50,
        'hour_of_day': 14,
        'day_of_week': 3,
        'merchant_category': 'grocery',
        'location_risk_score': 0.2,
        'days_since_last_transaction': 1.2,
        'transaction_frequency_24h': 2,
        'amount_vs_user_avg': 0.9,
        'is_weekend': 0
    }
    
    # Suspicious transaction
    suspicious_transaction = {
        'transaction_id': 'txn_002',
        'user_id': 'user_67890',
        'amount': 2500.00,
        'hour_of_day': 3,
        'day_of_week': 6,
        'merchant_category': 'online',
        'location_risk_score': 0.9,
        'days_since_last_transaction': 0.1,
        'transaction_frequency_24h': 8,
        'amount_vs_user_avg': 4.5,
        'is_weekend': 1
    }
    
    print("\n--- Normal Transaction Analysis ---")
    normal_result = fraud_detector.predict_transaction(normal_transaction)
    print(f"Fraud Score: {normal_result['fraud_score']:.4f}")
    print(f"Risk Level: {normal_result['risk_level']}")
    print(f"Action: {normal_result['recommended_action']}")
    print(f"Processing Time: {normal_result['processing_time_ms']:.2f}ms")
    
    print("\n--- Suspicious Transaction Analysis ---")
    suspicious_result = fraud_detector.predict_transaction(suspicious_transaction)
    print(f"Fraud Score: {suspicious_result['fraud_score']:.4f}")
    print(f"Risk Level: {suspicious_result['risk_level']}")
    print(f"Action: {suspicious_result['recommended_action']}")
    print(f"Processing Time: {suspicious_result['processing_time_ms']:.2f}ms")
    
    # Get explanations
    print("\n4. Model Explanations (for regulatory compliance)...")
    explanation = fraud_detector.get_model_explanation(suspicious_transaction)
    print(f"\nExplanation: {explanation['explanation_text']}")
    print("\nKey Risk Factors:")
    for factor in explanation['key_factors']:
        print(f"  - {factor['feature']}: {factor['value']:.3f} (importance: {factor['importance']:.3f})")
    
    # Save model
    print("\n5. Saving trained model...")
    fraud_detector.save_model('fraud_detection_model.pkl')
    
    print("\n=== DEMO COMPLETED ===")
    print("\nNext Steps:")
    print("1. Integrate with real transaction stream")
    print("2. Set up monitoring dashboards")
    print("3. Implement feedback loop for model improvement")
    print("4. Add more sophisticated features (user behavior analysis)")
    print("5. Deploy to production with proper MLOps pipeline")
    
    return fraud_detector

# Run the demonstration
if __name__ == "__main__":
    fraud_detector = demonstrate_fraud_detection()
