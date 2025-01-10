import json
import toml
import pandas as pd
from typing import Dict, Any, Optional, List
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import openai

from llm import LLM

ALLOWED_TYPES = {
    "string",
    "number",
    "boolean",
    "integer",
    "object",
    "array",
    "enum",
    "anyOf"
}

DEFAULT_SYSTEM_PROMPT = "You are an AI extracting information according to the provided schema into JSON format."

class Processor:
    """
    Single-step pipeline for extracting structured JSON fields from rows in a DataFrame.

    - system_prompt: Always set by default to DEFAULT_SYSTEM_PROMPT.
    - instructions: optional; if present, appended to the user message as "INSTRUCTIONS: ...".
    - user_template: optional. If set, placeholders {field} will be replaced with row data.
      If NOT set, we generate user content from 'fields' or all columns.
    - fields: optional list of columns to include if user_template is None.
      If fields is None, use all row keys.

    - extraction_schema: dict with e.g.:
        { 
          "relevant_sample": { "type": "boolean", "description": "..." },
          "sentiment":       { "type": "integer", "description": "..." }
        }
      Allowed types: string, number, boolean, integer, object, array, enum, anyOf.

    - llm: If None, defaults to an LLM() with model="gpt4o-mini".
    - from_toml: loads instructions/user_template/extraction_schema from TOML, but no model mention.
    - concurrency: use max_workers for parallel calls.
    """

    def __init__(
        self,
        extraction_schema: Dict[str, Dict[str, Any]],
        instructions: Optional[str] = None,
        system_prompt: Optional[str] = None,
        user_template: Optional[str] = None,
        fields: Optional[List[str]] = None,
        llm: Optional[LLM] = None,
        max_workers: int = 4
    ):
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.instructions = (instructions or "").strip()
        self.user_template = (user_template or "").strip()
        self.fields = fields if fields else None
        self.max_workers = max_workers

        if llm is None:
            self.llm = LLM()  # defaults to gpt4o-mini
        else:
            self.llm = llm

        if not extraction_schema:
            raise ValueError("extraction_schema cannot be empty.")

        self.extraction_schema = {}
        for field_name, info in extraction_schema.items():
            t = info.get("type", "").lower()
            if t not in ALLOWED_TYPES:
                raise ValueError(
                    f"Field '{field_name}' has invalid type '{t}'. "
                    f"Allowed: {ALLOWED_TYPES}"
                )
            self.extraction_schema[field_name] = {
                "type": t,
                "description": info.get("description", ""),
                # if there's "items" for array, or "enum" data, we just store them here
                # but not used for type coercion right now
                "items": info.get("items", None)
            }

    @classmethod
    def from_toml(
        cls,
        filepath: str,
        llm: Optional[LLM] = None
    ) -> "Processor":
        data = toml.load(filepath)
        p = data.get("processor", {})
        if "extraction_schema" not in data:
            raise ValueError("TOML must contain 'extraction_schema' at top level.")
        e_schema = data["extraction_schema"]

        instructions = p.get("instructions", "")
        user_template = p.get("user_template", None)

        schema = {}
        for k, v in e_schema.items():
            schema[k] = {
                "type": v.get("type", "").lower(),
                "description": v.get("description", ""),
                "items": v.get("items")  # optional
            }

        # This class method won't read fields from TOML, so assume user sets them in code if they want
        return cls(
            instructions=instructions,
            user_template=user_template,
            extraction_schema=schema,
            llm=llm,
            max_workers=p.get("max_workers", 4)
        )

    def add_extraction_field(self, name: str, field_type: str, description: str = "", items: Any = None):
        ft = field_type.lower()
        if ft not in ALLOWED_TYPES:
            raise ValueError(f"Invalid type '{field_type}'. Must be in {ALLOWED_TYPES}")
        self.extraction_schema[name] = {
            "type": ft,
            "description": description,
            "items": items
        }

    def _build_schema_for_llm(self) -> Dict[str, Any]:
        props = {}
        required_fields = []
        for fn, meta in self.extraction_schema.items():
            # base property
            prop = {"type": meta["type"]}
            # if it's an array with "items", we can specify in the schema
            if meta["type"] == "array" and meta["items"]:
                prop["items"] = meta["items"]
            # if it's an enum, you might have "enum": [...]
            # or if it's anyOf, etc. We won't fully parse, we just trust user input.

            props[fn] = prop
            required_fields.append(fn)

        return {
            "name": "extract_fields",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": props,
                "required": required_fields,
                "additionalProperties": False
            }
        }

    def _coerce_value(self, value: Any, field_type: str) -> Any:
        """Coerce value to the declared field_type if possible."""
        if field_type == "boolean":
            val_str = str(value).lower()
            if val_str in ["true", "yes", "1"]:
                return True
            elif val_str in ["false", "no", "0"]:
                return False
            else:
                return None
        elif field_type == "integer":
            try:
                return int(value)
            except:
                return None
        elif field_type == "number":
            try:
                return float(value)
            except:
                return None
        return value  # string, object, array, enum, anyOf => no extra parsing

    def _build_user_message(self, row_data: Dict[str, Any]) -> str:
        """
        If user_template is set, we do placeholders.
        Otherwise:
          - If self.fields is set, we only pass those
          - If self.fields is None, pass all row columns
          Each line: "field_name: value"
        Then optionally prepend instructions as "INSTRUCTIONS: ..."
        """
        if self.user_template:
            # Use placeholders
            prompt = self.user_template
            for k, v in row_data.items():
                ph = f"{{{k}}}"
                if ph in prompt:
                    prompt = prompt.replace(ph, str(v) if v else "")
            if self.instructions:
                return f"INSTRUCTIONS: {self.instructions}\n\n{prompt}"
            else:
                return prompt
        else:
            # Build line-based content
            relevant_keys = self.fields if self.fields else row_data.keys()
            lines = []
            for k in relevant_keys:
                val = row_data.get(k, "")
                lines.append(f"{k}: {val}")
            main_content = "\n".join(lines)
            if self.instructions:
                return f"INSTRUCTIONS: {self.instructions}\n\n{main_content}"
            else:
                return main_content

    def _process_single(self, row_data: Dict[str, Any]) -> Dict[str, Any]:
        openai.api_key = self.llm.api_key
        openai.api_base = self.llm.base_url

        system_msg = {"role": "system", "content": self.system_prompt}
        user_msg = {"role": "user", "content": self._build_user_message(row_data)}

        try:
            schema = self._build_schema_for_llm()
            resp = self.llm.client.chat.completions.create(
                model=self.llm.model,
                messages=[system_msg, user_msg],
                response_format={"type": "json_schema", "json_schema": schema},
                temperature=0
            )
            content = resp.choices[0].message.content.strip()
        except Exception as e:
            raise RuntimeError(f"API call failed: {str(e)}\nSystem: {system_msg}\nUser: {user_msg}")

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse LLM response as JSON: {str(e)}\nRaw content: {content}")

        # Type coercion with validation
        result = {}
        for fn, meta in self.extraction_schema.items():
            t = meta["type"]
            value = self._coerce_value(parsed.get(fn, None), t)
            if value is None and fn in parsed:  # If coercion failed but field was present
                raise ValueError(f"Failed to coerce field '{fn}' value '{parsed[fn]}' to type '{t}'")
            result[fn] = value
        return result

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            for f in self.extraction_schema:
                df[f] = None
            return df

        rows = df.to_dict("records")
        results = []

        if self.max_workers > 1:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [executor.submit(self._process_single, r) for r in rows]
                for f in tqdm(futures, desc="LLM Processing"):
                    # Let exceptions bubble up - no try/except
                    results.append(f.result())
        else:
            for r in tqdm(rows, desc="LLM Processing"):
                # Let exceptions bubble up - no try/except
                results.append(self._process_single(r))

        df_ex = pd.DataFrame(results)
        return pd.concat([df.reset_index(drop=True), df_ex], axis=1)