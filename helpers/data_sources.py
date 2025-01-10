# sculptor/helpers/data_sources.py
import praw
import requests
import pandas as pd
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type

class BaseDataSource(ABC):
    """
    Returns a DataFrame with zero or more arbitrary columns.
    No required fields.
    """
    _registry: Dict[str, Type['BaseDataSource']] = {}
    
    @classmethod
    def register(cls, name: str):
        """Class decorator to register data sources"""
        def wrapper(wrapped_class):
            cls._registry[name] = wrapped_class
            return wrapped_class
        return wrapper
    
    @classmethod
    def get_source_class(cls, name: str) -> Type['BaseDataSource']:
        """Get a data source class by its registered name."""
        if name not in cls._registry:
            available = list(cls._registry.keys())
            raise ValueError(f"Unknown data source type: {name}. Available types: {available}")
        return cls._registry[name]
    
    @abstractmethod
    def get_data(self) -> pd.DataFrame:
        pass

@BaseDataSource.register("csv")
class CSVDataSource(BaseDataSource):
    """
    Reads any CSV. No required columns.
    """
    def __init__(self, filepath: str, **kwargs):
        self.filepath = filepath
        self.kwargs = kwargs
    def get_data(self) -> pd.DataFrame:
        df = pd.read_csv(self.filepath, **self.kwargs)
        df = df.drop_duplicates().reset_index(drop=True)
        return df

@BaseDataSource.register("list")
class ListDataSource(BaseDataSource):
    """
    Accepts a list of dicts with any keys.
    No required fields.
    """
    def __init__(self, data: List[Dict[str, Any]]):
        self.data = data
    def get_data(self) -> pd.DataFrame:
        if not self.data:
            return pd.DataFrame()
        df = pd.DataFrame(self.data)
        df = df.drop_duplicates().reset_index(drop=True)
        return df

@BaseDataSource.register("reddit")
class RedditDataSource(BaseDataSource):
    """
    Example Reddit DataSource. 
    """
    def __init__(self, query: str,
                 client_id: str,
                 client_secret: str, 
                 user_agent: str,
                 include_comments: bool = False,
                 limit: Optional[int] = None,
                 subreddits: Optional[List[str]] = None,
                 sort: str = 'relevance', time_filter: str = 'year'):
        self.query = query
        self.include_comments = include_comments
        self.limit = limit
        self.subreddits = subreddits or ["all"]
        self.sort = sort
        self.time_filter = time_filter
        self.reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent
        )

    def get_data(self) -> pd.DataFrame:
        rows, total = [], 0
        for sub_name in self.subreddits:
            sub = self.reddit.subreddit(sub_name)
            results = sub.search(self.query, sort=self.sort, time_filter=self.time_filter,
                                 limit=self.limit if self.limit else None)
            for p in results:
                if self.limit and total >= self.limit:
                    break
                rows.append({
                    "id": f"{p.id}_post",
                    "text": p.selftext or p.title or "",
                    "title": p.title or "",
                    "context_text": "",
                    "url": f"https://reddit.com{p.permalink}",
                    "subreddit": p.subreddit.display_name,
                    "score": p.score,
                    "created_utc": p.created_utc,
                    "is_comment": False,
                    "comment_id": None
                })
                total += 1
                if self.include_comments:
                    p.comments.replace_more(limit=0)
                    for c in p.comments:
                        if self.limit and total >= self.limit:
                            break
                        rows.append({
                            "id": f"{p.id}_comment_{c.id}",
                            "text": getattr(c, "body", ""),
                            "title": f"[Comment] {p.title}",
                            "context_text": f"Original Post Title: {p.title}",
                            "url": f"https://reddit.com{p.permalink}",
                            "subreddit": p.subreddit.display_name,
                            "score": getattr(c, "score", None),
                            "created_utc": getattr(c, "created_utc", None),
                            "is_comment": True,
                            "comment_id": c.id
                        })
                        total += 1
                if self.limit and total >= self.limit:
                    break
        df = pd.DataFrame(rows)
        if not df.empty:
            df.drop_duplicates(inplace=True)
            if "created_utc" in df.columns and pd.api.types.is_numeric_dtype(df["created_utc"]):
                df["created_utc"] = pd.to_datetime(df["created_utc"], unit="s", errors="coerce")
        return df.reset_index(drop=True)

@BaseDataSource.register("hackernews")
class HackerNewsDataSource(BaseDataSource):
    """
    Example HackerNews DataSource.
    No required fields; we produce whatever columns we choose.
    """
    def __init__(self, query: str, include_comments: bool = False,
                 limit: Optional[int] = None, tags: Optional[List[str]] = None):
        self.query = query
        self.include_comments = include_comments
        self.limit = limit
        self.tags = tags or ["story"]

    def _build_url(self, page: int) -> str:
        base = "https://hn.algolia.com/api/v1/search"
        return f"{base}?query={self.query}&tags={','.join(self.tags)}&hitsPerPage=100&page={page}"

    def _fetch_stories(self) -> List[Dict[str, Any]]:
        out, total, page = [], 0, 0
        while True:
            r = requests.get(self._build_url(page))
            if r.status_code != 200:
                break
            data = r.json()
            hits = data.get("hits", [])
            if not hits:
                break
            for h in hits:
                out.append(h)
                total += 1
                if self.limit and total >= self.limit:
                    break
            if self.limit and total >= self.limit:
                break
            page += 1
        return out

    def _fetch_comments(self, sid: str) -> List[Dict[str, Any]]:
        url = f"https://hn.algolia.com/api/v1/search?tags=comment,story_{sid}&hitsPerPage=100"
        out, page = [], 0
        while True:
            r = requests.get(f"{url}&page={page}")
            if r.status_code != 200:
                break
            d = r.json()
            hits = d.get("hits", [])
            if not hits:
                break
            out.extend(hits)
            if len(hits) < 100:
                break
            page += 1
        return out

    def get_data(self) -> pd.DataFrame:
        stories = self._fetch_stories()
        rows, total = [], 0
        for s in stories:
            sid = str(s.get("objectID", ""))
            title = s.get("title", "")
            txt = s.get("story_text", "") or s.get("comment_text", "")
            url = s.get("url", f"https://news.ycombinator.com/item?id={sid}")
            rows.append({
                "id": f"{sid}_story",
                "text": txt or title,
                "title": title,
                "context_text": "",
                "url": url,
                "score": s.get("points", None),
                "created_utc": s.get("created_at_i", None),
                "is_comment": False,
                "comment_id": None
            })
            total += 1
            if self.include_comments:
                c_list = self._fetch_comments(sid)
                for c in c_list:
                    cid = str(c.get("objectID", ""))
                    c_txt = c.get("comment_text", "")
                    rows.append({
                        "id": f"{sid}_comment_{cid}",
                        "text": c_txt,
                        "title": f"[Comment] {title}",
                        "context_text": f"Story Title: {title}\nURL: {url}",
                        "url": f"https://news.ycombinator.com/item?id={cid}",
                        "score": c.get("points", None),
                        "created_utc": c.get("created_at_i", None),
                        "is_comment": True,
                        "comment_id": cid
                    })
                    total += 1
                    if self.limit and total >= self.limit:
                        break
            if self.limit and total >= self.limit:
                break
        df = pd.DataFrame(rows)
        if not df.empty:
            df.drop_duplicates(inplace=True)
            if pd.api.types.is_numeric_dtype(df["created_utc"]):
                df["created_utc"] = pd.to_datetime(df["created_utc"], unit="s", errors="coerce")
        return df.reset_index(drop=True)