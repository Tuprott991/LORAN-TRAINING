import os
from pathlib import Path

# Google Cloud Configuration
PROJECT_ID = "flash-spark-467403-r8"
LOCATION = "us-central1"  # or your preferred region
SERVICE_ACCOUNT_KEY = os.path.join(Path(__file__).parent.parent, "flash-spark-467403-r8-167a4e4656aa.json")

# Vertex AI Configuration
MODEL_NAME = "gemini-2.5-pro"  # or "gemini-1.5-pro" depending on availability

# Generation Configuration
DEFAULT_OUTPUT_PATH = os.path.join(Path(__file__).parent, "synthetic_data.json")
DEFAULT_NUM_RECORDS = 500
RECORDS_PER_BATCH = 10  # Number of records to generate per API call

# Prompt template for batch generation
PROMPT_TEMPLATE = """You are a creative assistant that generates **realistic user-style queries** for memory retrieval from personal multimedia collections (images, videos, audio/voice clips, or albums).

Your task is to:
1. **Create {num_records} diverse, realistic memory captions** describing different scenes, events, people, or moments (e.g., family gatherings, vacations, milestones, everyday moments, celebrations, activities).
2. For each caption, produce **5-10 diverse, realistic paraphrases** of how a human might ask to retrieve that memory.
3. For each caption, generate **2-3 plausible but incorrect queries** for contrastive learning.

**Caption Diversity Requirements:**
- Mix different types of memories: family events, personal milestones, vacations, holidays, everyday moments, celebrations, activities, pets, friends, work events, etc.
- Include various timeframes: specific dates, seasons, years, relative times ("last summer", "two years ago")
- Vary the subjects: specific people (names), family relations, friends, pets, groups
- Include different locations: home, outdoor venues, cities, countries, specific rooms, landmarks
- Mix formal and informal events
- Include both special occasions and mundane moments

**Query Generation Requirements:**
1. Each paraphrase should feel **natural and varied**, like a real user would type or speak.
2. Include **different angles of reference**, e.g., temporal ("last Christmas"), relational ("my son John"), emotional ("I miss John"), or contextual cues ("graduation ceremony").
3. Avoid rigid templated sentences; include **different sentence structures, colloquial phrasing, and implied context**.
4. Ensure queries are **specific enough** to retrieve the target memory but not overly verbose.
5. For negative samples, generate **plausible but incorrect queries** that could be confused with this caption but do NOT actually match it. Make them realistic and varied.

**Output Format (JSON ONLY, no additional text):**
[
  {{
    "id": "memory_0001",
    "caption": "<diverse_realistic_caption>",
    "queries": [
      "<natural_query_1>",
      "<natural_query_2>",
      "...",
      "<natural_query_N>"
    ],
    "negatives": [
      "<plausible_negative_1>",
      "<plausible_negative_2>",
      "<plausible_negative_3>"
    ]
  }},
  ...
]

**Examples of diverse captions and queries:**

1. Caption: "John graduating from HCMUS, April 2023"
   Queries: ["Show me John's graduation photos", "Our son John finishing university", "John walking across stage at graduation", "Pictures of John in cap and gown", "When John got his degree"]
   Negatives: ["Family vacation in Da Lat 2022", "John playing basketball at the park", "Anna's graduation ceremony"]

2. Caption: "Family in living room during Christmas 2024"
   Queries: ["See our living room decorations for Xmas", "Family gathering at home last Christmas", "Pictures of us celebrating Christmas together", "Christmas morning in the family room"]
   Negatives: ["Halloween party at my friend's house", "Summer trip to the beach", "Thanksgiving dinner with family"]

3. Caption: "Luna the golden retriever playing fetch in the park, sunny afternoon"
   Queries: ["Our dog Luna at the park", "Show me Luna playing fetch", "That sunny day with Luna outside", "Golden retriever having fun in the park"]
   Negatives: ["Cat playing with yarn indoors", "Kids playing soccer in the park", "Luna sleeping on the couch"]

Now generate **{num_records} completely diverse memory records** following the format above. Return ONLY the JSON array. Start IDs from memory_{start_id:04d}.
"""
