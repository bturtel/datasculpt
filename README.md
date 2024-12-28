
# DataSculpt  
AI-Powered Unstructured Data Analysis

DataSculpt leverages AI to transform unstructured text (e.g. Reddit posts, HackerNews, user reviews) into structured insights. DataSculpt provides an end-to-end pipeline for data collection, filtering, and analysis.

[See an example analysis of AI therapy discussions on Reddit](https://www.pensiveapp.com/reports/ai-therapy-reddit-analysis).

## Overview

- **Data Collection** – Pulls data from multiple sources (e.g., Reddit, HackerNews) using modular `DataSource` classes  
- **Automated Filtering & Extraction** – Uses AI models to filter irrelevant data and extract structured fields  
- **Analysis & Visualization** – Outputs JSON or DataFrames for further analysis, plus integrated plotting in notebooks  


## Workflow
1. **Load Config** – main.ipynb reads your TOML config.
2. **Initialize Data Sources** – Based on the config, DataSculpt creates the relevant DataSource instances (e.g., Reddit, HackerNews).
3. **Gather Data** – Each data source fetches records, merges them into a master DataFrame.
4. **Filter Data** – An AI-based filter checks each row to see if it's relevant to your use case.
5. **Extract Fields** – Another AI model extracts structured information into new columns.
6. **Visualize & Analyze** – You can then use Pandas, the built-in plotting functions, or external tools to explore and visualize the data.

Start with **`example.ipynb`** for a more detailed example.


## Minimal Usage Example

```python
import toml
import pandas as pd

# (1) Load configs for data sources and fields
config = toml.load("my_use_case.toml")

# (2) Create data sources based on the config
from datasource import RedditDataSource, HackerNewsDataSource
from processor import Processor, FILTER_PROMPT

data_sources = []
for ds_conf in config["data_sources"]:
    ds_type = ds_conf.pop("type", None)
    if ds_type == "reddit":
        data_sources.append(RedditDataSource(**ds_conf))
    elif ds_type == "hackernews":
        data_sources.append(HackerNewsDataSource(**ds_conf))

# (3) Create a Processor with your LLM details
processor = Processor(
    use_case_description=config["use_case"],
    filter_prompt=FILTER_PROMPT,
    extraction_schema=config["fields"],
    filter_llm_client=<Your LLM Client>,
    filter_model="<model_for_filtering>",
    extract_llm_client=<Your LLM Client>,
    extract_model="<model_for_extraction>"
)

# (4) Gather data
dfs = [source.get_data() for source in data_sources]
df = pd.concat(dfs, ignore_index=True).drop_duplicates(subset='id')

# (5) Filter and extract
df_filtered = processor.filter_data(df)
df_extracted = processor.extract_fields(df_filtered)

# (6) Done! Inspect your structured dataframe
print(df_extracted.head())
```

## Creating a Config for a new use case
1. **Create a TOML file** (e.g., `my_use_case.toml`).
2. **Define `use_case`** to describe your research or topic.  
3. **Under `[fields]`,** specify the extraction schema. For each key, define the data type (e.g., `type = "string"`, `type = "array"`) and include a `description`.  Required fields include `relevant_sample` and `relevant_sample_explanation` for filtering.
4. **Specify your data sources** under `[[data_sources]]`.  
   - Example:
     ```toml
     [[data_sources]]
     type = "reddit"
     query = "AI"
     include_comments = false
     limit = 100
     ```

## How to Add New Data Sources
1. **Create a subclass** of `BaseDataSource` in `datasource.py`.
2. **Implement `get_data()`** so it returns a Pandas DataFrame with at a minimum the following columns:
   - `id`
   - `text`
   - `url`
   - `context_text` is optional, but recommended to include for analyzing comments or other text that might make sense without additional context.
   - Any additional metadata for your analysis (score, created_utc, etc.).
3. **Import your new datasource** in `main.ipynb`, and add logic to handle your new source type:
   ```python
   if ds_type == "mynewsource":
       data_sources.append(MyNewSourceDataSource(**ds_conf))

