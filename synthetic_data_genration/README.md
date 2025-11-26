# Synthetic Data Generator for Memory Retrieval

This tool generates realistic user-style queries for memory retrieval from personal multimedia collections using Vertex AI Gemini-2.5-Pro.

## Features

- **LLM-Generated Captions**: Gemini creates diverse, realistic memory captions automatically
- **Batch Processing**: Efficient generation with configurable batch sizes
- **Natural Query Paraphrases**: 5-10 diverse ways users might search for each memory
- **Contrastive Learning**: 2-3 plausible negative samples per caption
- **Configurable Scale**: Generate from hundreds to thousands of records
- **Progress Tracking**: Real-time progress bars and statistics
- **Robust Error Handling**: Automatic retry and fallback mechanisms

## Setup

1. **Install dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```

2. **Configure Google Cloud credentials:**
   - Ensure your service account JSON key is in the parent directory
   - The key should be named `flash-spark-467403-r8-167a4e4656aa.json`
   - Or update the path in `config.py`

## Usage

### Basic usage (default settings):
```powershell
python data_generator.py
```

This will:
- Generate 5000 records by default
- LLM creates diverse captions automatically
- Process in batches of 10 records per API call
- Save output to `synthetic_data.json`

### Custom number of records:
```powershell
python data_generator.py --num_records 1000
```

### Custom output path:
```powershell
python data_generator.py --output_path "output/my_data.json"
```

### Adjust batch size:
```powershell
python data_generator.py --batch_size 20
```

Larger batches are more efficient but may occasionally fail. Default is 10.

### Combined options:
```powershell
python data_generator.py --output_path "data/queries.json" --num_records 2000 --batch_size 15
```

### Quick test run:
```powershell
python data_generator.py --num_records 20 --output_path "test_output.json"
```

## Output Format

The generated JSON file will contain an array of objects:

```json
[
  {
    "id": "memory_0001",
    "caption": "John graduating from HCMUS, April 2023",
    "queries": [
      "Show me John's graduation photos",
      "Our son John finishing university",
      "John walking across stage at graduation",
      "Pictures of John in cap and gown",
      "John celebrating graduation with family"
    ],
    "negatives": [
      "Family vacation in Da Lat 2022",
      "John playing basketball at the park",
      "Graduation ceremony of Anna at HCMUS"
    ]
  }
]
```

## How It Works

1. **Batch Generation**: The script divides your target number of records into batches
2. **LLM Creation**: For each batch, Gemini generates diverse memory captions covering:
   - Family events, vacations, celebrations, milestones
   - Various timeframes (dates, seasons, relative times)
   - Different subjects (people, pets, groups, locations)
   - Mix of formal events and everyday moments
3. **Query Generation**: For each caption, creates 5-10 natural paraphrases
4. **Negative Samples**: Adds 2-3 plausible but incorrect queries per caption
5. **JSON Output**: Saves structured data ready for training

## Configuration

Edit `config.py` to customize:
- Project ID and location
- Model name (default: gemini-2.5-pro)
- Default output path and number of records
- Batch size for API calls
- Prompt template for caption diversity

## Troubleshooting

- **Authentication errors:** Verify your service account key path and permissions
- **API errors:** Check that Vertex AI API is enabled in your GCP project
- **Rate limiting:** The script includes automatic delays between requests
- **JSON parsing errors:** The script will log errors and continue processing
