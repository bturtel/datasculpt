import pandas as pd
import json
from typing import Dict, Any
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import numpy as np

FILTER_PROMPT = "Is this text relevant to the described use case? Answer 'Yes' or 'No' and explain briefly."

class Processor:
    def __init__(
        self,
        use_case_description: str,
        filter_prompt: str,
        extraction_schema: Dict[str, Any],
        filter_llm_client,
        extract_llm_client,
        filter_model: str,
        extract_model: str,
        max_workers: int = 4
    ):
        self.use_case_description = use_case_description
        self.filter_prompt = filter_prompt
        self.extraction_schema = extraction_schema
        self.filter_llm_client = filter_llm_client
        self.extract_llm_client = extract_llm_client
        self.filter_model = filter_model
        self.extract_model = extract_model
        self.max_workers = max_workers

    def _build_filter_prompt(self, text: str, context: str) -> str:
        if context.strip():
            return f"""
{self.use_case_description}

You have a main TEXT and an optional CONTEXT. Focus ONLY on the TEXT for determining relevance.

TEXT:
{text}

CONTEXT:
{context}

Question: {self.filter_prompt}
Answer 'Yes' or 'No' and a brief explanation.
"""
        else:
            return f"""
{self.use_case_description}

Analyze the following TEXT:
{text}

Question: {self.filter_prompt}
Answer 'Yes' or 'No' and a brief explanation.
"""

    def _filter_single(self, args):
        """Helper function to run a single filter LLM call."""
        text, context = args
        prompt = self._build_filter_prompt(text, context)
        response = self.filter_llm_client.chat.completions.create(
            model=self.filter_model,
            messages=[{"role":"user","content":prompt}],
            temperature=0
        )
        answer = response.choices[0].message.content.strip().lower()
        relevant = answer.startswith('yes')
        return relevant, answer

    def filter_data(self, df: pd.DataFrame, output_file: str = None) -> pd.DataFrame:
        df = df.copy()
        if df.empty:
            df["is_relevant"] = []
            df["relevance_explanation"] = []
            if output_file:
                df.to_json(output_file, orient='records', date_format='iso')
            return df

        # Prepare input for parallel calls
        inputs = [(row['text'], row['context_text']) for _, row in df.iterrows()]

        # Run in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(tqdm(executor.map(self._filter_single, inputs), total=len(inputs), desc="Filtering data"))

        is_relevant = [res[0] for res in results]
        explanations = [res[1] for res in results]

        df["is_relevant"] = is_relevant
        df["relevance_explanation"] = explanations

        if output_file:
            df.to_json(output_file, orient='records', date_format='iso')

        return df

    def _build_extraction_prompt(self, text: str, context: str) -> str:
        schema_str = json.dumps(self.extraction_schema, indent=2)
        if context.strip():
            return f"""
{self.use_case_description}

You have a main TEXT and a CONTEXT. Extract the requested fields from the TEXT only.

TEXT:
{text}

CONTEXT:
{context}

Extract fields as JSON according to the schema:
{schema_str}

Return only valid JSON:
"""
        else:
            return f"""
{self.use_case_description}

Analyze the TEXT and extract requested fields.

TEXT:
{text}

Extract fields as JSON according to the schema:
{schema_str}

Return only valid JSON:
"""

    def _extract_single(self, args):
        """Helper function to run a single extraction LLM call."""
        text, context = args
        prompt = self._build_extraction_prompt(text, context)
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

        response = self.extract_llm_client.chat.completions.create(
            model=self.extract_model,
            messages=[
                {"role":"system","content":self.use_case_description},
                {"role":"user","content":prompt}
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
        return data

    def extract_fields(self, df: pd.DataFrame, output_file: str = None) -> pd.DataFrame:
        df = df.copy()
        if df.empty:
            for f in self.extraction_schema.keys():
                df[f] = None
            if output_file:
                df.to_json(output_file, orient='records', date_format='iso')
            return df

        relevant_df = df[df["is_relevant"] == True]

        if relevant_df.empty:
            for f in self.extraction_schema.keys():
                df[f] = None
            if output_file:
                df.to_json(output_file, orient='records', date_format='iso')
            return df

        # Prepare input for parallel calls
        inputs = [(row['text'], row['context_text']) for _, row in relevant_df.iterrows()]

        # Run extractions in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(tqdm(executor.map(self._extract_single, inputs), total=len(inputs), desc="Extracting fields"))

        # Merge results back
        # relevant_df and results correspond one-to-one in order
        extracted_df = pd.DataFrame(results)
        # Add id from relevant_df
        extracted_df['id'] = relevant_df['id'].values

        df = pd.merge(df, extracted_df, on="id", how="left")

        for f in self.extraction_schema.keys():
            if f not in df.columns:
                df[f] = None

        if output_file:
            df.to_json(output_file, orient='records', date_format='iso')

        return df
