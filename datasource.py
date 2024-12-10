import pandas as pd
import toml
import praw
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List

secrets = toml.load("secrets.toml")

class BaseDataSource(ABC):
    @abstractmethod
    def get_data(self) -> pd.DataFrame:
        """
        Returns a DataFrame with columns:
          - id: unique string per sample
          - combined_text: text to be analyzed by LLM
          - metadata: dict with source-specific info (e.g. subreddit, score, is_comment)
        """
        pass

class RedditDataSource(BaseDataSource):
    def __init__(self, query: str, limit: int, include_comments: bool, subreddits: Optional[List[str]]=None):
        """
        If no subreddits are provided or an empty list is given, defaults to ['all'].
        """
        self.query = query
        self.limit = limit
        self.include_comments = include_comments
        self.subreddits = subreddits if subreddits and len(subreddits) > 0 else ["all"]

        self.reddit = praw.Reddit(
            client_id=secrets["reddit"]["client_id"],
            client_secret=secrets["reddit"]["client_secret"],
            user_agent=secrets["reddit"]["user_agent"]
        )

    def get_data(self) -> pd.DataFrame:
        posts = []
        for subreddit in self.subreddits:
            sub = self.reddit.subreddit(subreddit)
            search_results = sub.search(
                self.query, sort='relevance', time_filter='year', limit=self.limit
            )
            for post in search_results:
                original_text = f"Title: {post.title}\nBody: {getattr(post, 'selftext', '')}"
                post_id = post.id
                post_metadata = {
                    'subreddit': post.subreddit.display_name,
                    'score': post.score,
                    'created_utc': post.created_utc,
                    'url': f"https://reddit.com{post.permalink}",
                    'num_comments': post.num_comments,
                    'is_comment': False
                }
                posts.append({
                    'id': f"{post_id}_post",
                    'combined_text': original_text,
                    'metadata': post_metadata
                })

                if self.include_comments:
                    post.comments.replace_more(limit=0)
                    for c in post.comments:
                        if hasattr(c, 'body'):
                            comment_body = c.body
                            combined_comment_text = (
                                f"Comment: {comment_body}\n"
                                f"In Response to Original Post:\n{original_text}"
                            )
                            comment_metadata = {
                                'subreddit': post.subreddit.display_name,
                                'score': getattr(c, 'score', None),
                                'created_utc': getattr(c, 'created_utc', None),
                                'url': f"https://reddit.com{post.permalink}",
                                'is_comment': True,
                                'original_post': original_text,
                                'comment_id': c.id
                            }
                            posts.append({
                                'id': f"{post_id}_comment_{c.id}",
                                'combined_text': combined_comment_text,
                                'metadata': comment_metadata
                            })

        df = pd.DataFrame(posts)
        if df.empty:
            return pd.DataFrame(columns=['id', 'combined_text', 'metadata'])
        df = df.drop_duplicates(subset='id', keep='first').reset_index(drop=True)
        return df

class CSVDataSource(BaseDataSource):
    def __init__(self, file_path: str, text_column: str, metadata_columns: Optional[List[str]]=None):
        self.file_path = file_path
        self.text_column = text_column
        self.metadata_columns = metadata_columns or []

    def get_data(self) -> pd.DataFrame:
        df = pd.read_csv(self.file_path)
        df = df.rename(columns={self.text_column: "combined_text"})
        if "id" not in df.columns:
            df["id"] = df.index.astype(str)
        meta = df[self.metadata_columns].to_dict(orient='records') if self.metadata_columns else [{} for _ in range(len(df))]
        df["metadata"] = meta
        # If user included other columns, that's fine; we only need these three guaranteed.
        return df[["id", "combined_text", "metadata"]]
