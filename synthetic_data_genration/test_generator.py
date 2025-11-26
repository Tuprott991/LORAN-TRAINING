"""
Quick test script to verify the synthetic data generator setup.
Generates a small batch to test the connection and output format.
"""

from data_generator import SyntheticDataGenerator
from config import PROJECT_ID, LOCATION, SERVICE_ACCOUNT_KEY, MODEL_NAME
import json

def test_generation():
    """Test generating a small batch of records."""
    print("🧪 Testing Synthetic Data Generator\n")
    
    try:
        # Initialize generator
        print("1️⃣ Initializing Vertex AI connection...")
        generator = SyntheticDataGenerator(
            project_id=PROJECT_ID,
            location=LOCATION,
            service_account_key=SERVICE_ACCOUNT_KEY,
            model_name=MODEL_NAME
        )
        
        # Test small batch
        print("\n2️⃣ Generating test batch (3 records)...")
        results = generator.generate_batch(num_records=3, start_id=1)
        
        if results:
            print(f"\n✅ Successfully generated {len(results)} records!")
            print("\n📄 Sample output:")
            print(json.dumps(results[0], indent=2, ensure_ascii=False))
            
            # Validate structure
            print("\n3️⃣ Validating structure...")
            for i, record in enumerate(results, 1):
                assert "id" in record, f"Record {i} missing 'id'"
                assert "caption" in record, f"Record {i} missing 'caption'"
                assert "queries" in record, f"Record {i} missing 'queries'"
                assert "negatives" in record, f"Record {i} missing 'negatives'"
                assert len(record["queries"]) >= 5, f"Record {i} has too few queries"
                assert len(record["negatives"]) >= 2, f"Record {i} has too few negatives"
            
            print("✅ All records have correct structure!")
            print("\n🎉 Test passed! Generator is working correctly.")
            print("\n💡 Run the main script to generate full dataset:")
            print("   python data_generator.py --num_records 100")
            
        else:
            print("❌ No results generated. Check error messages above.")
            
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_generation()
