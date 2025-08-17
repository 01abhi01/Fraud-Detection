"""
Simple test script for the Fraud Detection System
Run this to test individual components
"""
from fraud_detection_system import FraudDetectionSystem

def test_fraud_detection():
    print("Testing Fraud Detection System Components...")
    
    # Initialize system
    detector = FraudDetectionSystem()
    
    # Test 1: Generate training data
    print("\n1. Testing data generation...")
    data = detector.generate_synthetic_data(n_samples=1000, fraud_ratio=0.05)
    print(f"   ✓ Generated {len(data)} transactions")
    print(f"   ✓ Fraud rate: {data['is_fraud'].mean():.1%}")
    
    # Test 2: Train models
    print("\n2. Testing model training...")
    detector.train_models(data)
    print("   ✓ Models trained successfully")
    
    # Test 3: Single transaction prediction
    print("\n3. Testing transaction prediction...")
    test_transaction = {
        'amount': 150.00,
        'hour_of_day': 15,
        'day_of_week': 2,
        'merchant_category': 'restaurant',
        'location_risk_score': 0.3,
        'days_since_last_transaction': 0.5,
        'transaction_frequency_24h': 3,
        'amount_vs_user_avg': 1.2,
        'is_weekend': 0
    }
    
    result = detector.predict_transaction(test_transaction)
    print(f"   ✓ Fraud Score: {result['fraud_score']:.4f}")
    print(f"   ✓ Risk Level: {result['risk_level']}")
    print(f"   ✓ Processing Time: {result['processing_time_ms']:.2f}ms")
    
    # Test 4: Model explanation
    print("\n4. Testing model explanation...")
    explanation = detector.get_model_explanation(test_transaction)
    print(f"   ✓ Explanation: {explanation['explanation_text']}")
    
    # Test 5: Model persistence
    print("\n5. Testing model save/load...")
    detector.save_model('test_model.pkl')
    
    new_detector = FraudDetectionSystem()
    new_detector.load_model('test_model.pkl')
    
    # Test prediction with loaded model
    new_result = new_detector.predict_transaction(test_transaction)
    assert abs(result['fraud_score'] - new_result['fraud_score']) < 0.001
    print("   ✓ Model save/load working correctly")
    
    print("\n✅ All tests passed! Fraud Detection System is working correctly.")

if __name__ == "__main__":
    test_fraud_detection()
