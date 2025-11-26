# Synthetic Data Generation System

## Overview
This system uses **Vertex AI Gemini-2.5-Pro** to automatically generate diverse, realistic memory retrieval datasets. The LLM creates both the memory captions AND the natural language queries, ensuring maximum diversity and realism.

## Key Improvements Over Original Approach

### ❌ Old Approach (Constrained)
- Used fixed list of 20 sample captions
- Limited diversity in memory types
- Manual caption creation required
- Repetitive patterns in generated queries

### ✅ New Approach (LLM-Driven)
- **LLM generates captions dynamically** during generation
- Unlimited diversity in memory types, subjects, locations, timeframes
- No manual caption creation needed
- Natural variation across thousands of records
- Batch processing for efficiency

## What Gets Generated

For each memory record, the system creates:

1. **Diverse Caption**: Realistic memory description
   - Examples: family events, vacations, milestones, everyday moments
   - Includes specific details: names, dates, locations, emotions
   - Varies in formality and specificity

2. **5-10 Natural Queries**: How users would search for this memory
   - Different phrasings and perspectives
   - Temporal references ("last Christmas")
   - Relational references ("my son John")
   - Emotional cues ("I miss...")
   - Contextual hints ("graduation ceremony")

3. **2-3 Negative Samples**: Plausible but incorrect queries
   - Similar enough to be confusing
   - Actually refer to different memories
   - Useful for contrastive learning

## Example Output

```json
{
  "id": "memory_0042",
  "caption": "Luna the golden retriever playing fetch in the park, sunny afternoon",
  "queries": [
    "Show me Luna at the park",
    "Our dog playing fetch outside",
    "That sunny day with Luna",
    "Golden retriever having fun",
    "When Luna was playing in the park",
    "Pictures of our dog catching the ball",
    "Luna enjoying the outdoor afternoon"
  ],
  "negatives": [
    "Cat playing with yarn at home",
    "Kids playing soccer in the field",
    "Luna sleeping on the couch"
  ]
}
```

## Diversity Achieved

The LLM is instructed to create captions covering:

- **Event Types**: birthdays, graduations, weddings, vacations, holidays, everyday moments
- **Subjects**: family members (with names), friends, pets, groups, individuals
- **Locations**: home (specific rooms), outdoor venues, cities, countries, landmarks
- **Timeframes**: specific dates, seasons, years, relative times ("last summer", "two years ago")
- **Formality**: mix of special occasions and mundane moments
- **Emotions**: celebratory, nostalgic, casual, formal

## Scalability

- **Small test**: 10-20 records for validation
- **Development**: 100-500 records for prototyping
- **Training**: 1,000-5,000 records for model training
- **Production**: 10,000+ records for robust systems

### Performance
- Batch size: 10 records per API call (configurable)
- Generation time: ~2-3 seconds per batch
- Rate: ~200-300 records per minute
- Cost-efficient with batching

## Use Cases

1. **Training Retrieval Models**: 
   - Query-caption similarity learning
   - Contrastive learning with negatives
   - Cross-modal retrieval (text-to-image/video)

2. **Testing Search Systems**:
   - Evaluate natural language understanding
   - Benchmark query variations
   - Test edge cases

3. **Data Augmentation**:
   - Expand small labeled datasets
   - Balance training data
   - Cover underrepresented scenarios

## Quality Control

The system ensures quality through:
- High temperature (0.95) for diversity
- Structured prompt with clear requirements
- JSON schema validation
- Error handling and retry logic
- Post-generation statistics and validation
