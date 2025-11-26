import os
import json
import argparse
from typing import List, Dict
from pathlib import Path
import vertexai
from vertexai.generative_models import GenerativeModel
from google.oauth2 import service_account
from tqdm import tqdm
import time

from config import (
    PROJECT_ID,
    LOCATION,
    SERVICE_ACCOUNT_KEY,
    MODEL_NAME,
    DEFAULT_OUTPUT_PATH,
    DEFAULT_NUM_RECORDS,
    RECORDS_PER_BATCH,
    PROMPT_TEMPLATE
)


class SyntheticDataGenerator:
    """Generate synthetic memory retrieval queries using Vertex AI Gemini."""
    
    def __init__(self, project_id: str, location: str, service_account_key: str, model_name: str):
        """
        Initialize the generator with Vertex AI credentials.
        
        Args:
            project_id: Google Cloud project ID
            location: GCP region (e.g., 'us-central1')
            service_account_key: Path to service account JSON key
            model_name: Name of the Gemini model to use
        """
        self.project_id = project_id
        self.location = location
        self.model_name = model_name
        
        # Initialize credentials
        credentials = service_account.Credentials.from_service_account_file(
            service_account_key,
            scopes=['https://www.googleapis.com/auth/cloud-platform']
        )
        
        # Initialize Vertex AI
        vertexai.init(
            project=project_id,
            location=location,
            credentials=credentials
        )
        
        # Initialize the model
        self.model = GenerativeModel(model_name)
        print(f"✓ Initialized Vertex AI with model: {model_name}")
    
    def generate_batch(self, num_records: int, start_id: int) -> List[Dict]:
        """
        Generate a batch of memory records with captions created by the LLM.
        
        Args:
            num_records: Number of records to generate in this batch
            start_id: Starting ID number for the batch
            
        Returns:
            List of generated memory records
        """
        prompt = PROMPT_TEMPLATE.format(
            num_records=num_records,
            start_id=start_id
        )
        
        try:
            # Generate content
            response = self.model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.95,
                    "top_p": 0.95,
                    "top_k": 40,
                    "max_output_tokens": 32768,
                }
            )
            
            # Extract text response
            response_text = response.text.strip()
            
            # Clean up response - remove markdown code blocks if present
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            response_text = response_text.strip()
            
            # Parse JSON
            results = json.loads(response_text)
            
            # Ensure we got a list
            if not isinstance(results, list):
                results = [results]
            
            return results
            
        except json.JSONDecodeError as e:
            print(f"⚠ JSON parsing error in batch starting at ID {start_id}: {e}")
            print(f"Response preview: {response_text[:300]}...")
            return []
        except Exception as e:
            print(f"⚠ Error generating batch starting at ID {start_id}: {e}")
            return []
    
    def generate_dataset(
        self,
        output_path: str,
        num_records: int,
        batch_size: int = RECORDS_PER_BATCH
    ) -> List[Dict]:
        """
        Generate synthetic dataset with LLM-created captions.
        
        Args:
            output_path: Path to save the generated JSON data
            num_records: Total number of records to generate
            batch_size: Number of records to generate per API call
            
        Returns:
            List of generated data records
        """
        print(f"\n🚀 Generating {num_records} synthetic memory records...")
        print(f"� Batch size: {batch_size} records per API call")
        print(f"� Output path: {output_path}\n")
        
        results = []
        num_batches = (num_records + batch_size - 1) // batch_size
        
        # Generate in batches with progress bar
        with tqdm(total=num_records, desc="Generating memories") as pbar:
            for batch_idx in range(num_batches):
                start_id = batch_idx * batch_size + 1
                records_in_batch = min(batch_size, num_records - len(results))
                
                print(f"\n🔄 Batch {batch_idx + 1}/{num_batches}: Generating {records_in_batch} records (IDs {start_id}-{start_id + records_in_batch - 1})")
                
                batch_results = self.generate_batch(records_in_batch, start_id)
                
                if batch_results:
                    results.extend(batch_results)
                    pbar.update(len(batch_results))
                    print(f"✓ Successfully generated {len(batch_results)} records")
                else:
                    print(f"⚠ Batch failed, skipping...")
                    pbar.update(records_in_batch)
                
                # Rate limiting - delay between batches
                if batch_idx < num_batches - 1:
                    time.sleep(1)
        
        # Save to file
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Print statistics
        total_queries = sum(len(r.get('queries', [])) for r in results)
        total_negatives = sum(len(r.get('negatives', [])) for r in results)
        errors = sum(1 for r in results if 'error' in r)
        
        print(f"\n✅ Generation complete!")
        print(f"📊 Statistics:")
        print(f"   - Total records generated: {len(results)}")
        print(f"   - Total queries: {total_queries}")
        print(f"   - Total negatives: {total_negatives}")
        print(f"   - Failed records: {errors}")
        print(f"   - Average queries per record: {total_queries / len(results) if results else 0:.1f}")
        print(f"   - Average negatives per record: {total_negatives / len(results) if results else 0:.1f}")
        print(f"   - Saved to: {output_path}")
        
        return results


def main():
    """Main function to run the synthetic data generator."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic memory retrieval queries using Vertex AI Gemini. "
                    "The LLM will create diverse memory captions automatically."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Path to save the generated JSON file (default: {DEFAULT_OUTPUT_PATH})"
    )
    parser.add_argument(
        "--num_records",
        type=int,
        default=DEFAULT_NUM_RECORDS,
        help=f"Number of records to generate (default: {DEFAULT_NUM_RECORDS})"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=RECORDS_PER_BATCH,
        help=f"Number of records per API call (default: {RECORDS_PER_BATCH})"
    )
    
    args = parser.parse_args()
    
    print(f"🤖 Using Vertex AI Gemini to generate diverse memory captions and queries")
    print(f"� Target: {args.num_records} records in batches of {args.batch_size}")
    
    # Initialize generator
    generator = SyntheticDataGenerator(
        project_id=PROJECT_ID,
        location=LOCATION,
        service_account_key=SERVICE_ACCOUNT_KEY,
        model_name=MODEL_NAME
    )
    
    # Generate dataset
    generator.generate_dataset(
        output_path=args.output_path,
        num_records=args.num_records,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()
