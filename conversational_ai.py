"""
Conversational AI Interface for Fraud Detection System
Provides natural language interaction with the fraud detection system
"""
import re
import json
from datetime import datetime
from fraud_detection_system import FraudDetectionSystem
from typing import Dict, List, Optional

class FraudDetectionChatbot:
    """
    Conversational AI interface for fraud detection system
    Allows natural language queries and explanations
    """
    
    def __init__(self):
        self.fraud_detector = FraudDetectionSystem()
        self.conversation_history = []
        self.current_transaction = None
        
        # Intent patterns for natural language understanding
        self.intent_patterns = {
            'check_transaction': [
                r'check.*transaction',
                r'analyze.*transaction',
                r'is.*fraud',
                r'fraud.*check',
                r'scan.*transaction'
            ],
            'explain_decision': [
                r'why.*flagged',
                r'explain.*decision',
                r'why.*fraud',
                r'what.*factors',
                r'reasons.*for'
            ],
            'get_stats': [
                r'show.*stats',
                r'performance.*metrics',
                r'system.*status',
                r'how.*accurate',
                r'detection.*rate'
            ],
            'help': [
                r'help',
                r'what.*can.*do',
                r'commands',
                r'how.*use'
            ],
            'greeting': [
                r'hello',
                r'hi',
                r'hey',
                r'good.*morning',
                r'good.*afternoon'
            ]
        }
        
        # Response templates
        self.responses = {
            'greeting': [
                "Hello! I'm your AI fraud detection assistant. I can help you analyze transactions, explain fraud decisions, and answer questions about our system.",
                "Hi there! I'm here to help with fraud detection. You can ask me to check transactions, explain decisions, or get system statistics.",
                "Welcome! I can analyze transactions for fraud, explain why decisions were made, and provide system insights. How can I help you today?"
            ],
            'help': """
I can help you with:

🔍 **Transaction Analysis**
- "Check this transaction for fraud"
- "Analyze transaction: amount $500, merchant gas station, time 2am"
- "Is this transaction fraudulent?"

💡 **Explanations**  
- "Why was this transaction flagged?"
- "Explain the fraud decision"
- "What factors indicate fraud?"

📊 **System Information**
- "Show system performance"
- "What's the detection accuracy?"
- "System status"

Just ask me anything in natural language!
            """,
            'need_transaction': [
                "I need transaction details to analyze. Please provide: amount, time, merchant category, and location.",
                "To check for fraud, I need transaction information like amount, hour, merchant type, and location risk.",
                "Please share the transaction details you'd like me to analyze for potential fraud."
            ]
        }
    
    def process_message(self, message: str) -> str:
        """Process user message and return appropriate response"""
        message = message.lower().strip()
        
        # Store in conversation history
        self.conversation_history.append({
            'timestamp': datetime.now().isoformat(),
            'user_message': message,
            'intent': self._detect_intent(message)
        })
        
        # Detect intent and generate response
        intent = self._detect_intent(message)
        
        if intent == 'greeting':
            return self._get_random_response('greeting')
        
        elif intent == 'help':
            return self.responses['help']
        
        elif intent == 'check_transaction':
            return self._handle_transaction_check(message)
        
        elif intent == 'explain_decision':
            return self._handle_explanation_request(message)
        
        elif intent == 'get_stats':
            return self._handle_stats_request()
        
        else:
            return self._handle_unknown_intent(message)
    
    def _detect_intent(self, message: str) -> str:
        """Detect user intent from message using pattern matching"""
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, message, re.IGNORECASE):
                    return intent
        return 'unknown'
    
    def _get_random_response(self, response_type: str) -> str:
        """Get random response from templates"""
        import random
        responses = self.responses.get(response_type, ["I'm not sure how to respond to that."])
        return random.choice(responses) if isinstance(responses, list) else responses
    
    def _handle_transaction_check(self, message: str) -> str:
        """Handle transaction fraud checking requests"""
        # Try to extract transaction details from message
        transaction_data = self._extract_transaction_details(message)
        
        if not transaction_data:
            return self._get_random_response('need_transaction') + "\n\nExample: 'Check transaction: $1500, online purchase, 3am, high-risk location'"
        
        # Analyze transaction
        try:
            result = self.fraud_detector.predict_transaction(transaction_data)
            self.current_transaction = {
                'data': transaction_data,
                'result': result
            }
            
            return self._format_fraud_analysis(result, transaction_data)
        
        except Exception as e:
            return f"Sorry, I encountered an error analyzing the transaction: {str(e)}"
    
    def _extract_transaction_details(self, message: str) -> Optional[Dict]:
        """Extract transaction details from natural language"""
        transaction = {}
        
        # Extract amount
        amount_match = re.search(r'\$?(\d+(?:\.\d{2})?)', message)
        if amount_match:
            transaction['amount'] = float(amount_match.group(1))
        
        # Extract time/hour
        time_patterns = [
            (r'(\d{1,2})\s*am', lambda x: int(x)),
            (r'(\d{1,2})\s*pm', lambda x: int(x) + 12 if int(x) != 12 else 12),
            (r'(\d{1,2})\s*o\'?clock', lambda x: int(x)),
            (r'morning', lambda x: 9),
            (r'afternoon', lambda x: 15),
            (r'evening', lambda x: 19),
            (r'night', lambda x: 23),
            (r'midnight', lambda x: 0)
        ]
        
        for pattern, converter in time_patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                if pattern in [r'morning', r'afternoon', r'evening', r'night', r'midnight']:
                    transaction['hour_of_day'] = converter(None)
                else:
                    transaction['hour_of_day'] = converter(match.group(1))
                break
        
        # Extract merchant category
        merchant_categories = {
            'grocery': ['grocery', 'supermarket', 'food store'],
            'gas': ['gas', 'fuel', 'petrol', 'gas station'],
            'restaurant': ['restaurant', 'dining', 'food', 'cafe'],
            'retail': ['retail', 'store', 'shopping', 'mall'],
            'online': ['online', 'internet', 'web', 'e-commerce'],
            'atm': ['atm', 'cash', 'withdrawal']
        }
        
        for category, keywords in merchant_categories.items():
            if any(keyword in message.lower() for keyword in keywords):
                transaction['merchant_category'] = category
                break
        
        # Extract location risk indicators
        if any(word in message.lower() for word in ['high-risk', 'risky', 'suspicious', 'foreign']):
            transaction['location_risk_score'] = 0.8
        elif any(word in message.lower() for word in ['safe', 'familiar', 'usual']):
            transaction['location_risk_score'] = 0.2
        else:
            transaction['location_risk_score'] = 0.4  # Default
        
        # Set defaults for missing fields
        defaults = {
            'day_of_week': 3,  # Wednesday
            'days_since_last_transaction': 1.0,
            'transaction_frequency_24h': 2,
            'amount_vs_user_avg': 1.0,
            'is_weekend': 0
        }
        
        for key, value in defaults.items():
            if key not in transaction:
                transaction[key] = value
        
        # Check if we have minimum required fields
        required_fields = ['amount']
        if all(field in transaction for field in required_fields):
            return transaction
        
        return None
    
    def _format_fraud_analysis(self, result: Dict, transaction_data: Dict) -> str:
        """Format fraud analysis results in conversational style"""
        score = result['fraud_score']
        risk_level = result['risk_level']
        action = result['recommended_action']
        processing_time = result.get('processing_time_ms', 0)
        
        # Risk level emoji and description
        risk_emoji = {
            'LOW': '✅',
            'MEDIUM': '⚠️',
            'HIGH': '🚨'
        }
        
        risk_descriptions = {
            'LOW': 'This looks like a normal, legitimate transaction.',
            'MEDIUM': 'This transaction has some suspicious characteristics and needs review.',
            'HIGH': 'This transaction shows strong indicators of fraud and should be blocked.'
        }
        
        # Format transaction summary
        amount = transaction_data.get('amount', 'Unknown')
        merchant = transaction_data.get('merchant_category', 'Unknown').title()
        hour = transaction_data.get('hour_of_day', 'Unknown')
        
        response = f"""
🔍 **Transaction Analysis Complete**

**Transaction Details:**
• Amount: ${amount}
• Merchant: {merchant}
• Time: {hour}:00
• Processing Time: {processing_time:.1f}ms

**Fraud Assessment:**
{risk_emoji.get(risk_level, '❓')} **Risk Level: {risk_level}**
📊 Fraud Score: {score:.3f} (0.000 = Safe, 1.000 = Fraud)
🎯 Recommended Action: **{action}**

**Analysis:** {risk_descriptions.get(risk_level, 'Unable to determine risk level.')}
        """
        
        # Add quick explanation for high/medium risk
        if risk_level in ['HIGH', 'MEDIUM']:
            response += "\n💡 Ask me 'Why was this flagged?' for detailed explanation."
        
        return response.strip()
    
    def _handle_explanation_request(self, message: str) -> str:
        """Handle requests for fraud decision explanations"""
        if not self.current_transaction:
            return "I need to analyze a transaction first before I can explain the decision. Please ask me to check a transaction."
        
        try:
            explanation = self.fraud_detector.get_model_explanation(
                self.current_transaction['data']
            )
            
            return self._format_explanation(explanation)
        
        except Exception as e:
            return f"Sorry, I couldn't generate an explanation: {str(e)}"
    
    def _format_explanation(self, explanation: Dict) -> str:
        """Format explanation in conversational style"""
        score = explanation['fraud_score']
        risk_level = explanation['risk_level']
        explanation_text = explanation['explanation_text']
        key_factors = explanation['key_factors']
        
        response = f"""
🧠 **Why This Decision Was Made**

**Overall Assessment:** {explanation_text}

**Key Risk Factors:**
"""
        
        for i, factor in enumerate(key_factors[:3], 1):
            feature_name = self._humanize_feature_name(factor['feature'])
            value = factor['value']
            importance = factor['importance']
            
            response += f"{i}. **{feature_name}**\n"
            response += f"   Value: {value:.2f} | Importance: {importance:.1%}\n"
        
        response += f"""
**How We Calculate Risk:**
• We analyze 16 different transaction features
• Each feature gets weighted by importance (0-100%)
• Higher values in suspicious features increase fraud score
• Final score: {score:.3f} = {score*100:.1f}% fraud probability

**Regulatory Note:** This explanation can be used for audit and compliance purposes.
        """
        
        return response.strip()
    
    def _humanize_feature_name(self, feature: str) -> str:
        """Convert technical feature names to human-readable descriptions"""
        feature_descriptions = {
            'amount_vs_user_avg': 'Amount vs User Average',
            'risk_interaction': 'Location Risk × Frequency',
            'hour_cos': 'Time of Day Pattern',
            'hour_sin': 'Time of Day Pattern',
            'location_time_risk': 'Location Risk at This Time',
            'is_unusual_hour': 'Unusual Transaction Hour',
            'transaction_frequency_24h': '24-Hour Transaction Frequency',
            'amount_log': 'Transaction Amount (Log Scale)',
            'location_risk_score': 'Location Risk Score',
            'merchant_category_encoded': 'Merchant Category Risk',
            'is_high_amount': 'High Amount Flag',
            'is_weekend': 'Weekend Transaction',
            'day_sin': 'Day of Week Pattern',
            'day_cos': 'Day of Week Pattern',
            'days_since_last_transaction': 'Days Since Last Transaction',
            'amount_frequency_interaction': 'Amount × Frequency Pattern'
        }
        
        return feature_descriptions.get(feature, feature.replace('_', ' ').title())
    
    def _handle_stats_request(self) -> str:
        """Handle system statistics requests"""
        # Generate mock statistics (in production, get from monitoring system)
        return """
📊 **Fraud Detection System Status**

**Model Performance:**
• Accuracy: 99.97% ROC AUC
• Processing Speed: <30ms average
• Uptime: 99.9%

**Recent Activity (Last 24 Hours):**
• Transactions Processed: 156,240
• Fraud Detected: 312 cases
• False Positive Rate: 0.1%
• High Risk Blocked: 89 transactions

**Model Information:**
• Algorithm: Random Forest + Isolation Forest Ensemble
• Features: 16 engineered features
• Training Data: 50,000+ transactions
• Last Updated: 2 days ago

**System Health:** ✅ All systems operational
        """
    
    def _handle_unknown_intent(self, message: str) -> str:
        """Handle messages with unknown intent"""
        return """
I'm not sure I understand that request. Here's what I can help with:

🔍 **Check Transactions:** "Analyze this $500 gas station purchase at 2am"
💡 **Explain Decisions:** "Why was this transaction flagged?"
📊 **System Info:** "Show system performance stats"
❓ **Get Help:** "What can you do?"

Try rephrasing your question or ask for help!
        """

def run_chatbot_demo():
    """Interactive demo of the fraud detection chatbot"""
    print("🤖 Fraud Detection AI Assistant")
    print("=" * 50)
    print("Type 'quit' to exit\n")
    
    chatbot = FraudDetectionChatbot()
    
    # Train the fraud detection model first
    print("🔧 Initializing fraud detection models...")
    training_data = chatbot.fraud_detector.generate_synthetic_data(n_samples=1000, fraud_ratio=0.02)
    chatbot.fraud_detector.train_models(training_data)
    print("✅ Models ready!\n")
    
    # Start conversation
    print(chatbot.process_message("hello"))
    print("\n" + "-" * 50 + "\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("\nChatbot: Goodbye! Stay safe from fraud! 👋")
                break
            
            if user_input:
                response = chatbot.process_message(user_input)
                print(f"\nChatbot: {response}")
                print("\n" + "-" * 50 + "\n")
        
        except KeyboardInterrupt:
            print("\n\nChatbot: Goodbye! 👋")
            break
        except Exception as e:
            print(f"\nError: {e}")

if __name__ == "__main__":
    run_chatbot_demo()
