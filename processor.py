import pandas as pd
import json
from typing import Dict, Any
from tqdm import tqdm

class Processor:
    def __init__(
        self,
        use_case_description: str,
        filter_prompt: str,
        extraction_schema: Dict[str, Any],
        filter_llm_client,
        extract_llm_client,
        filter_model: str,
        extract_model: str
    ):
        """
        Args:
            use_case_description: Text describing the overall use case.
            filter_prompt: Prompt to ask the filter LLM to determine if a sample is relevant.
            extraction_schema: Schema of fields to extract from relevant samples.
            filter_llm_client: An LLM client instance for filtering (e.g. cheaper model).
            extract_llm_client: An LLM client instance for extraction (e.g. better model).
            filter_model: Model name for filter LLM.
            extract_model: Model name for extraction LLM.
        """
        self.use_case_description = use_case_description
        self.filter_prompt = filter_prompt
        self.extraction_schema = extraction_schema
        self.filter_llm_client = filter_llm_client
        self.extract_llm_client = extract_llm_client
        self.filter_model = filter_model
        self.extract_model = extract_model

    def filter_data(self, df: pd.DataFrame, output_file: str = None) -> pd.DataFrame:
        """
        Filters data using the filter LLM.
        
        Args:
            df: Input DataFrame.
            output_file: If provided, writes the resulting DataFrame to this file (JSON).
        
        Returns:
            A new DataFrame with 'is_relevant' and 'relevance_explanation' columns added.
        """
        # Work on a copy to avoid SettingWithCopyWarning
        df = df.copy()
        if df.empty:
            df["is_relevant"] = []
            df["relevance_explanation"] = []
            if output_file:
                df.to_json(output_file, orient='records', date_format='iso')
            return df

        is_relevant = []
        explanations = []
        
        # Using tqdm for progress
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Filtering data"):
            metadata = row['metadata']
            is_comment = metadata.get('is_comment', False)
            if is_comment:
                prompt = f"""
{self.use_case_description}

You are given a COMMENT and the ORIGINAL POST it responds to. The original post is context only.
Focus ONLY on the COMMENT portion for determining relevance.

COMMENT and Context:
{row['combined_text']}

Question: {self.filter_prompt}
Answer 'Yes' or 'No' and a brief explanation.
"""
            else:
                prompt = f"""
{self.use_case_description}

Analyze the following POST:
{row['combined_text']}

Question: {self.filter_prompt}
Answer 'Yes' or 'No' and a brief explanation.
"""

            response = self.filter_llm_client.chat.completions.create(
                model=self.filter_model,
                messages=[{"role":"user","content":prompt}],
                temperature=0
            )
            answer = response.choices[0].message.content.strip().lower()
            relevant = answer.startswith('yes')
            is_relevant.append(relevant)
            explanations.append(answer)

        df["is_relevant"] = is_relevant
        df["relevance_explanation"] = explanations

        if output_file:
            df.to_json(output_file, orient='records', date_format='iso')

        return df

    def extract_fields(self, df: pd.DataFrame, output_file: str = None) -> pd.DataFrame:
        """
        Extracts fields using the extraction LLM.
        
        Args:
            df: DataFrame that has 'is_relevant' column (from filter step).
            output_file: If provided, writes the resulting DataFrame to this file (JSON).
        
        Returns:
            A new DataFrame with extracted fields added according to the schema.
        """
        # Work on a copy to avoid SettingWithCopyWarning
        df = df.copy()

        if df.empty:
            # Add extraction fields as None
            for f in self.extraction_schema.keys():
                df[f] = None
            if output_file:
                df.to_json(output_file, orient='records', date_format='iso')
            return df

        relevant_df = df[df["is_relevant"] == True]

        if relevant_df.empty:
            # No relevant samples, still add extraction fields as None
            for f in self.extraction_schema.keys():
                df[f] = None
            if output_file:
                df.to_json(output_file, orient='records', date_format='iso')
            return df

        schema = {
            "name": "extract_fields",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": self.extraction_schema,
                "required": list(self.extraction_schema.keys()),
                "additionalProperties": False
            }
        }

        results = []
        # Using tqdm for progress
        for _, row in tqdm(relevant_df.iterrows(), total=len(relevant_df), desc="Extracting fields"):
            metadata = row['metadata']
            is_comment = metadata.get('is_comment', False)
            if is_comment:
                extraction_prompt = f"""
{self.use_case_description}

You are given a COMMENT and the ORIGINAL POST it responds to. The original post is context only.
Focus ONLY on the COMMENT portion for extracting the fields.

COMMENT and Context:
{row['combined_text']}

Extract the requested fields as JSON according to this schema:
{json.dumps(self.extraction_schema, indent=2)}

Return only valid JSON:
"""
            else:
                extraction_prompt = f"""
{self.use_case_description}

Analyze the following POST:
{row['combined_text']}

Extract the requested fields as JSON according to this schema:
{json.dumps(self.extraction_schema, indent=2)}

Return only valid JSON:
"""

            response = self.extract_llm_client.chat.completions.create(
                model=self.extract_model,
                messages=[
                    {"role":"system","content":self.use_case_description},
                    {"role":"user","content":extraction_prompt}
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": schema
                },
                temperature=0
            )
            content = response.choices[0].message.content.strip()
            try:
                data = json.loads(content)
            except:
                data = {k: None for k in self.extraction_schema.keys()}
            results.append((row['id'], data))

        extracted_df = pd.DataFrame([{"id": rid, **res} for rid, res in results])
        df = pd.merge(df, extracted_df, on="id", how="left")

        # For posts not relevant, ensure fields exist as None
        for f in self.extraction_schema.keys():
            if f not in df.columns:
                df[f] = None

        if output_file:
            df.to_json(output_file, orient='records', date_format='iso')

        return df
