#!/usr/bin/env python3
"""
Test script to verify SHAP explanations align with ensemble predictions
"""

from fraud_detection_system import FraudDetectionSystem
import numpy as np

def test_shap_alignment():
    """Test the SHAP explanation alignment with ensemble prediction"""
    
    print("🔍 Testing SHAP-Ensemble Alignment...")
    print("=" * 50)
    
    # Initialize the system
    detector = FraudDetectionSystem()
    
    # Generate training data and train models
    print("📊 Training models...")
    data = detector.generate_synthetic_data(n_samples=1000)
    detector.train_models(data)
    
    # Test transaction: $5000 online purchase at 3am (your example)
    test_transaction = {
        'amount': 5000.0,
        'hour_of_day': 3,
        'merchant_category': 'online',
        'location_risk_score': 0.8,
        'transaction_frequency_24h': 8,
        'amount_vs_user_avg': 4.0,
        'is_weekend': True,
        'days_since_last_transaction': 0.1,
        'day_of_week': 6
    }
    
    print(f"\n🎯 Test Transaction: ${test_transaction['amount']:,.0f} online purchase at {test_transaction['hour_of_day']}am")
    print("-" * 50)
    
    # Get prediction and explanation
    explanation = detector.get_model_explanation(test_transaction)
    
    # Extract key information
    fraud_score = explanation['fraud_score']
    shap_explanation = explanation.get('shap_explanation')
    explanation_method = explanation.get('explanation_method')
    
    print(f"📈 **Fraud Score:** {fraud_score:.1%}")
    print(f"🔬 **Method:** {explanation_method}")
    
    if shap_explanation:
        baseline = shap_explanation['expected_value']
        total_shap_impact = shap_explanation['total_shap_impact']
        verification_sum = shap_explanation.get('verification_sum', 0)
        
        print(f"\n📊 **SHAP Analysis:**")
        print(f"   Baseline (average): {baseline:.1%} fraud probability")
        print(f"   Your transaction: {fraud_score:.1%} fraud probability")
        print(f"   Difference to explain: {(fraud_score - baseline):.1%}")
        print(f"   SHAP total impact: {total_shap_impact:.1%}")
        print(f"   Verification (baseline + impact): {verification_sum:.1%}")
        
        print(f"\n🔍 **SHAP breaks this down:**")
        
        # Show top contributing factors
        for i, factor in enumerate(shap_explanation['feature_contributions'][:5], 1):
            feature = factor['feature']
            shap_val = factor['shap_value']
            feature_val = factor['feature_value']
            
            # Convert to user-friendly names
            if 'amount' in feature and ('vs' in feature or 'avg' in feature):
                description = f"Amount vs average ({feature_val:.1f}x)"
            elif 'hour' in feature or 'unusual' in feature:
                description = "Unusual time (3am)"
            elif 'merchant' in feature or 'online' in str(feature_val).lower():
                description = "Online merchant"
            elif 'location' in feature:
                description = "Location risk"
            elif 'frequency' in feature:
                description = f"Transaction frequency ({feature_val:.0f}/day)"
            else:
                description = feature.replace('_', ' ').title()
            
            contribution_pct = shap_val * 100
            if abs(contribution_pct) >= 1:  # Only show significant contributions
                sign = "+" if contribution_pct > 0 else ""
                print(f"   • {description}: {sign}{contribution_pct:.0f}%")
        
        # Verify mathematical consistency
        print(f"\n✅ **Verification:**")
        print(f"   Expected: {baseline:.1%} + {total_shap_impact:.1%} = {baseline + total_shap_impact:.1%}")
        print(f"   Actual ensemble: {fraud_score:.1%}")
        print(f"   Difference: {abs(fraud_score - verification_sum):.3f}")
        
        if abs(fraud_score - verification_sum) < 0.01:
            print("   ✅ SHAP explanations align with ensemble prediction!")
        else:
            print("   ⚠️  SHAP explanations may need adjustment")
    else:
        print("❌ SHAP explanations not available")
    
    print("\n" + "=" * 50)
    print("Test completed!")

if __name__ == "__main__":
    test_shap_alignment()
