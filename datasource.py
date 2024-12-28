import pandas as pd
import toml
import praw
import requests
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List

secrets = toml.load("secrets.toml")

class BaseDataSource(ABC):
    @abstractmethod
    def get_data(self) -> pd.DataFrame:
        """
        Returns a DataFrame with at least:
          - id (str)
          - text (str)
          - url (str)
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


class HackerNewsDataSource(BaseDataSource):
    def __init__(
        self,
        query: str,
        include_comments: bool = False,
        limit: Optional[int] = None,
        tags: Optional[List[str]] = None
    ):
        """
        :param query: Search query string (e.g. 'AI', 'ChatGPT', etc.)
        :param include_comments: Whether to fetch comments as separate records
        :param limit: Max number of items to return
        :param tags: Additional tags for Algolia HN search (e.g., ['story', 'comment'])
        """
        self.query = query
        self.include_comments = include_comments
        self.limit = limit
        # By default, search only 'story' if not provided
        self.tags = tags if tags else ["story"]

    def _build_hn_api_url(self, page: int) -> str:
        base_url = "https://hn.algolia.com/api/v1/search"
        # If you want to search comments separately, you can specify tags=comment below. 
        # For now we only search stories here:
        tags_param = ",".join(self.tags)
        return f"{base_url}?query={self.query}&tags={tags_param}&hitsPerPage=100&page={page}"

    def _fetch_stories(self) -> List[Dict[str, Any]]:
        stories = []
        total_fetched = 0
        page = 0

        while True:
            url = self._build_hn_api_url(page)
            resp = requests.get(url)
            if resp.status_code != 200:
                break

            data = resp.json()
            hits = data.get("hits", [])
            if not hits:
                break

            for h in hits:
                stories.append(h)
                total_fetched += 1
                if self.limit is not None and total_fetched >= self.limit:
                    break

            if self.limit is not None and total_fetched >= self.limit:
                break

            page += 1  # move to next page

        return stories

    def _fetch_comments_for_story(self, story_id: str) -> List[Dict[str, Any]]:
        # Example usage: fetch comments for a story (this is a separate call)
        base_url = "https://hn.algolia.com/api/v1/search"
        url = f"{base_url}?tags=comment,story_{story_id}&hitsPerPage=100"
        all_comments = []
        page = 0

        while True:
            resp = requests.get(f"{url}&page={page}")
            if resp.status_code != 200:
                break

            data = resp.json()
            hits = data.get("hits", [])
            if not hits:
                break

            all_comments.extend(hits)
            if len(hits) < 100:
                break
            page += 1

        return all_comments

    def get_data(self) -> pd.DataFrame:
        stories = self._fetch_stories()
        rows = []

        for s in stories:
            # Basic fields
            id_str = str(s.get("objectID", ""))
            title = s.get("title", "")
            text = s.get("story_text", "") or s.get("comment_text", "")
            url = s.get("url", f"https://news.ycombinator.com/item?id={id_str}")
            score = s.get("points", None)
            created_utc = s.get("created_at_i", None)
            author = s.get("author", "")

            rows.append({
                "id": f"{id_str}_story",
                "title": title,
                "text": text,
                "context_text": "",
                "url": url,
                "subreddit": None,  # not applicable to HN
                "score": score,
                "created_utc": created_utc,
                "is_comment": False,
                "comment_id": None,
                "num_comments": s.get("num_comments", None),
                "author": author
            })

            # Optionally fetch comments for each story
            if self.include_comments:
                all_comments = self._fetch_comments_for_story(id_str)
                for c in all_comments:
                    comment_id = str(c.get("objectID", ""))
                    comment_text = c.get("comment_text", "")
                    comment_author = c.get("author", "")
                    comment_created_utc = c.get("created_at_i", None)
                    score_c = c.get("points", None)

                    context_str = f"Story Title: {title}\nStory URL: {url}"
                    rows.append({
                        "id": f"{id_str}_comment_{comment_id}",
                        "title": "[Comment] " + title,
                        "text": comment_text,
                        "context_text": context_str,
                        "url": f"https://news.ycombinator.com/item?id={comment_id}",
                        "subreddit": None,
                        "score": score_c,
                        "created_utc": comment_created_utc,
                        "is_comment": True,
                        "comment_id": comment_id,
                        "num_comments": None,
                        "author": comment_author
                    })

        df = pd.DataFrame(rows).drop_duplicates(subset="id", keep="first").reset_index(drop=True)
        if not df.empty and pd.api.types.is_numeric_dtype(df["created_utc"]):
            df["created_utc"] = pd.to_datetime(df["created_utc"], unit="s", errors="coerce")

        return df
