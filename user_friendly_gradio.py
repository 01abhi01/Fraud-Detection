"""
User-Friendly Fraud Detection System with Gradio Interface
Natural language inputs that internally calculate technical parameters
"""
import gradio as gr
import pandas as pd
import numpy as np
from fraud_detection_system import FraudDetectionSystem
import time
from datetime import datetime

class UserFriendlyFraudDetector:
    def __init__(self):
        print("🔧 Initializing User-Friendly Fraud Detection System...")
        self.detector = FraudDetectionSystem()
        
        # Train the model
        print("📊 Training models with synthetic data...")
        training_data = self.detector.generate_synthetic_data(n_samples=5000, fraud_ratio=0.03)
        self.detector.train_models(training_data)
        print("✅ System ready!")
    
    def auto_calculate_location_risk(self, location_description, merchant_category):
        """Automatically calculate location risk based on user description"""
        location_lower = location_description.lower()
        
        # High risk indicators
        if any(word in location_lower for word in ['foreign', 'abroad', 'international', 'unfamiliar', 'first time', 'risky area', 'bad neighborhood', 'never been']):
            return 0.8
        # Medium-high risk
        elif any(word in location_lower for word in ['new city', 'different state', 'vacation', 'travel', 'airport', 'hotel']):
            return 0.6
        # Medium risk for ATMs or online
        elif merchant_category.lower() in ['atm', 'online']:
            return 0.5
        # Low risk indicators
        elif any(word in location_lower for word in ['home', 'usual', 'regular', 'neighborhood', 'local', 'familiar', 'always shop here']):
            return 0.2
        # Default medium-low risk
        else:
            return 0.3
    
    def auto_calculate_frequency(self, spending_pattern):
        """Calculate transaction frequency based on spending pattern"""
        pattern_lower = spending_pattern.lower()
        if any(word in pattern_lower for word in ['multiple', 'many', 'frequent', 'lots of', 'several times today']):
            return 8
        elif any(word in pattern_lower for word in ['few', 'occasional', 'rarely', 'first transaction today']):
            return 1
        elif any(word in pattern_lower for word in ['normal', 'regular', 'typical', 'usual amount']):
            return 3
        else:
            return 2
    
    def auto_calculate_amount_ratio(self, amount, spending_description):
        """Calculate how much this amount compares to user's typical spending"""
        spending_lower = spending_description.lower()
        
        # Base ratio calculation based on amount
        if amount > 5000:
            base_ratio = 8.0
        elif amount > 2000:
            base_ratio = 4.0
        elif amount > 500:
            base_ratio = 2.0
        else:
            base_ratio = 1.0
            
        # Adjust based on user description
        if any(word in spending_lower for word in ['much more', 'way more', 'unusual', 'never spend this much', 'emergency purchase']):
            return base_ratio * 1.5
        elif any(word in spending_lower for word in ['bit more', 'slightly more', 'little more']):
            return base_ratio * 0.8
        elif any(word in spending_lower for word in ['normal', 'typical', 'usual', 'always spend this']):
            return 1.0
        else:
            return base_ratio
    
    def predict_fraud(self, amount, hour, merchant_category, location_description, 
                     spending_description, spending_pattern, is_weekend, days_since):
        """Make fraud prediction with natural language inputs"""
        try:
            # Auto-calculate technical parameters internally
            location_risk = self.auto_calculate_location_risk(location_description, merchant_category)
            frequency = self.auto_calculate_frequency(spending_pattern)
            amount_ratio = self.auto_calculate_amount_ratio(amount, spending_description)
            
            # Create transaction data
            transaction_data = {
                'amount': float(amount),
                'hour_of_day': int(hour),
                'day_of_week': 6 if is_weekend else 3,
                'merchant_category': merchant_category.lower(),
                'location_risk_score': location_risk,
                'days_since_last_transaction': float(days_since),
                'transaction_frequency_24h': frequency,
                'amount_vs_user_avg': amount_ratio,
                'is_weekend': 1 if is_weekend else 0
            }
            
            # Get prediction
            result = self.detector.predict_transaction(transaction_data)
            explanation = self.detector.get_model_explanation(transaction_data)
            
            # Format results in user-friendly language
            fraud_score = result['fraud_score']
            risk_level = result['risk_level']
            action = result['recommended_action']
            processing_time = result['processing_time_ms']
            
            # Risk level styling
            risk_colors = {
                'LOW': '🟢',
                'MEDIUM': '🟡', 
                'HIGH': '🔴'
            }
            
            # Create user-friendly result
            result_text = f"""
## 🔍 Transaction Analysis

### What You Told Us:
- **Purchase Amount:** ${amount:,.2f}
- **Time:** {hour:02d}:00 {'(Weekend)' if is_weekend else '(Weekday)'}
- **Store Type:** {merchant_category.title()}
- **Location:** {location_description}
- **Spending Pattern:** {spending_description}
- **Recent Activity:** {spending_pattern}

### 📊 Our Assessment:
- **Risk Level:** {risk_colors.get(risk_level, '⚪')} **{risk_level} RISK**
- **Fraud Probability:** {fraud_score*100:.1f}%
- **Decision:** **{action}**
- **Analysis Time:** {processing_time:.1f}ms

### 💡 Why This Decision?
{explanation['explanation_text']}

### 🔍 Main Risk Factors:
"""
            
            for i, factor in enumerate(explanation['key_factors'][:3], 1):
                feature_name = self._humanize_feature_name(factor['feature'])
                importance = factor['importance']
                result_text += f"{i}. **{feature_name}** (influence: {importance:.1%})\n"
            
            # Set status message
            if risk_level == 'HIGH':
                status = "🚨 TRANSACTION BLOCKED - Please contact your bank"
            elif risk_level == 'MEDIUM':
                status = "⚠️ TRANSACTION UNDER REVIEW - Additional verification may be required"
            else:
                status = "✅ TRANSACTION APPROVED - Purchase completed successfully"
                
            return result_text, status
                
        except Exception as e:
            error_msg = f"❌ Error analyzing transaction: {str(e)}"
            return error_msg, "❌ System Error"
    
    def _humanize_feature_name(self, feature_name):
        """Convert technical feature names to user-friendly descriptions"""
        name_mapping = {
            'amount': 'Transaction Amount',
            'hour_of_day': 'Time of Purchase',
            'amount_vs_user_avg': 'Amount vs Your Normal Spending',
            'location_risk_score': 'Location Safety Score',
            'transaction_frequency_24h': 'How Often You Shop',
            'days_since_last_transaction': 'Time Since Last Purchase',
            'is_unusual_hour': 'Unusual Shopping Time',
            'amount_percentile': 'Amount Compared to Others',
            'risk_interaction': 'Combined Risk Factors',
            'location_time_risk': 'Location Risk at This Time',
            'merchant_category_encoded': 'Type of Store',
            'is_weekend': 'Weekend Shopping',
            'hour_sin': 'Time Pattern',
            'hour_cos': 'Time Pattern',
            'day_sin': 'Day Pattern',
            'day_cos': 'Day Pattern'
        }
        return name_mapping.get(feature_name, feature_name.replace('_', ' ').title())

def create_interface():
    detector = UserFriendlyFraudDetector()
    
    with gr.Blocks(title="🛡️ Smart Fraud Detection", theme=gr.themes.Soft()) as interface:
        gr.Markdown("""
        # 🛡️ Smart Fraud Detection System
        ### Tell us about your transaction in your own words - we'll handle the technical analysis!
        
        Just describe your purchase naturally, and our AI will determine if it's safe or suspicious.
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## 💳 Transaction Details")
                
                amount = gr.Number(
                    label="💰 How much are you spending? ($)",
                    value=100.0,
                    minimum=0.01,
                    info="Enter the purchase amount"
                )
                
                hour = gr.Slider(
                    label="🕐 What time is it? (24-hour format)",
                    minimum=0,
                    maximum=23,
                    value=14,
                    step=1,
                    info="0 = midnight, 12 = noon, 23 = 11 PM"
                )
                
                merchant_category = gr.Dropdown(
                    label="🏪 What type of store/service?",
                    choices=["grocery", "gas", "restaurant", "retail", "online", "atm"],
                    value="grocery",
                    info="Select the category that best matches"
                )
                
                is_weekend = gr.Checkbox(
                    label="📅 Is this a weekend?",
                    value=False,
                    info="Check if today is Saturday or Sunday"
                )
                
                days_since = gr.Number(
                    label="📊 Days since your last purchase",
                    value=1.0,
                    minimum=0.0,
                    info="How many days ago was your last transaction?"
                )
                
            with gr.Column(scale=1):
                gr.Markdown("## 📝 Describe Your Purchase")
                
                location_description = gr.Textbox(
                    label="📍 Where are you shopping?",
                    placeholder="e.g., 'at my usual grocery store', 'online from home', 'unfamiliar area while traveling', 'foreign country'",
                    lines=2,
                    info="Describe the location in your own words"
                )
                
                spending_description = gr.Textbox(
                    label="💸 How does this amount compare to your normal spending?",
                    placeholder="e.g., 'normal amount', 'much more than usual', 'emergency purchase', 'typical for this store'",
                    lines=2,
                    info="Tell us about this purchase amount"
                )
                
                spending_pattern = gr.Textbox(
                    label="🔄 How often have you been shopping lately?",
                    placeholder="e.g., 'normal shopping pattern', 'multiple purchases today', 'first transaction today', 'shopping more than usual'",
                    lines=2,
                    info="Describe your recent shopping activity"
                )
                
        with gr.Row():
            with gr.Column(scale=1):
                analyze_btn = gr.Button("🔍 Analyze Transaction", variant="primary", size="lg")
                
        with gr.Row():
            with gr.Column(scale=2):
                result_output = gr.Markdown(label="Analysis Results")
            with gr.Column(scale=1):
                status_output = gr.Textbox(label="Transaction Status", interactive=False)
        
        # Example buttons
        gr.Markdown("## 🎯 Try These Examples:")
        with gr.Row():
            normal_btn = gr.Button("✅ Normal Purchase", variant="secondary")
            suspicious_btn = gr.Button("🚨 Suspicious Purchase", variant="secondary")
            travel_btn = gr.Button("✈️ Travel Purchase", variant="secondary")
        
        # Wire up the prediction
        analyze_btn.click(
            fn=detector.predict_fraud,
            inputs=[amount, hour, merchant_category, location_description, 
                   spending_description, spending_pattern, is_weekend, days_since],
            outputs=[result_output, status_output]
        )
        
        # Example button functions
        def load_normal_example():
            return (
                50.0,  # amount
                15,    # hour
                "grocery",  # merchant
                "my regular neighborhood store",  # location
                "normal weekly grocery shopping",  # spending
                "typical shopping pattern",  # pattern
                False, # weekend
                2.0    # days since
            )
        
        def load_suspicious_example():
            return (
                5000.0,  # amount
                3,       # hour
                "online", # merchant
                "unfamiliar website, first time ordering", # location
                "way more than I usually spend",  # spending
                "multiple large purchases today", # pattern
                True,    # weekend
                0.1      # days since
            )
        
        def load_travel_example():
            return (
                200.0,   # amount
                22,      # hour
                "restaurant", # merchant
                "foreign country while on vacation", # location
                "bit more than usual but reasonable for travel", # spending
                "few purchases since arriving", # pattern
                False,   # weekend
                1.0      # days since
            )
        
        normal_btn.click(
            fn=load_normal_example,
            outputs=[amount, hour, merchant_category, location_description, 
                    spending_description, spending_pattern, is_weekend, days_since]
        )
        
        suspicious_btn.click(
            fn=load_suspicious_example,
            outputs=[amount, hour, merchant_category, location_description, 
                    spending_description, spending_pattern, is_weekend, days_since]
        )
        
        travel_btn.click(
            fn=load_travel_example,
            outputs=[amount, hour, merchant_category, location_description, 
                    spending_description, spending_pattern, is_weekend, days_since]
        )
        
        gr.Markdown("""
        ---
        ### 🔒 How It Works:
        Our AI analyzes your description using advanced machine learning to automatically calculate:
        - Location risk based on your description
        - Spending pattern analysis from your words
        - Behavioral anomaly detection
        - 16 different fraud indicators
        
        **No technical knowledge required - just describe your purchase naturally!**
        """)
    
    return interface

if __name__ == "__main__":
    print("🚀 Launching User-Friendly Fraud Detection System...")
    interface = create_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_api=False
    )
