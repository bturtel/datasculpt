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
        Returns a DataFrame with at least:
          - id (str)
          - title (str)
          - text (str)
          - context_text (str)
          - url (str)
          - subreddit (str)
          - score (float or int)
          - created_utc (datetime64[ns]) after conversion
          - is_comment (bool)
          - comment_id (str or None)
          - num_comments (int or None)
        """
        pass

class RedditDataSource(BaseDataSource):
    def __init__(self, query: str, include_comments: bool, limit: Optional[int]=None, subreddits: Optional[List[str]]=None):
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
        total_count = 0

        def build_record(
            id_str: str,
            title: str,
            text: str,
            context_text: str,
            url: str,
            subreddit: str,
            score: Optional[int],
            created_utc: Optional[float],
            is_comment: bool,
            comment_id: Optional[str],
            num_comments: Optional[int]
        ) -> Dict[str, Any]:
            return {
                'id': id_str,
                'title': title,
                'text': text,
                'context_text': context_text,
                'url': url,
                'subreddit': subreddit,
                'score': score,
                'created_utc': created_utc,  # Will convert to datetime after building the DataFrame
                'is_comment': is_comment,
                'comment_id': comment_id,
                'num_comments': num_comments
            }

        for subreddit in self.subreddits:
            sub = self.reddit.subreddit(subreddit)
            search_limit = self.limit if self.limit is not None else None
            search_results = sub.search(
                self.query, sort='relevance', time_filter='year', limit=search_limit
            )

            for p in search_results:
                if self.limit is not None and total_count >= self.limit:
                    break

                post_title = p.title or "No title"
                post_text = p.selftext or ""
                url = f"https://reddit.com{p.permalink}"

                post_record = build_record(
                    id_str=f"{p.id}_post",
                    title=post_title,
                    text=post_text,
                    context_text="",
                    url=url,
                    subreddit=p.subreddit.display_name,
                    score=p.score,
                    created_utc=p.created_utc,  # float timestamp
                    is_comment=False,
                    comment_id=None,
                    num_comments=p.num_comments
                )
                posts.append(post_record)
                total_count += 1

                if self.include_comments:
                    p.comments.replace_more(limit=0)
                    orig_title = post_title
                    orig_body = post_text
                    context_str = f"Original Post:\nTitle: {orig_title}\nBody: {orig_body}"

                    for c in p.comments:
                        if self.limit is not None and total_count >= self.limit:
                            break

                        if hasattr(c, 'body'):
                            comment_body = c.body or ""
                            comment_record = build_record(
                                id_str=f"{p.id}_comment_{c.id}",
                                title="[Comment] " + orig_title,
                                text=comment_body,
                                context_text=context_str,
                                url=url,
                                subreddit=p.subreddit.display_name,
                                score=getattr(c, 'score', None),
                                created_utc=getattr(c, 'created_utc', None),  # float timestamp
                                is_comment=True,
                                comment_id=c.id,
                                num_comments=None
                            )
                            posts.append(comment_record)
                            total_count += 1

            if self.limit is not None and total_count >= self.limit:
                break

        df = pd.DataFrame(posts)
        if df.empty:
            df = pd.DataFrame(columns=[
                'id','title','text','context_text','url','subreddit','score','created_utc',
                'is_comment','comment_id','num_comments'
            ])

        df = df.drop_duplicates(subset='id', keep='first').reset_index(drop=True)

        # Convert created_utc from float seconds to datetime
        if 'created_utc' in df.columns:
            # Only convert if not already datetime
            if pd.api.types.is_numeric_dtype(df['created_utc']):
                df['created_utc'] = pd.to_datetime(df['created_utc'], unit='s', errors='coerce')

        return df
