import praw
import toml
from typing import List, Dict, Optional

secrets = toml.load("secrets.toml")

REDDIT = praw.Reddit(
    client_id=secrets["reddit"]["client_id"],
    client_secret=secrets["reddit"]["client_secret"],
    user_agent=secrets["reddit"]["user_agent"]
)

def get_reddit_posts(query: str, limit: int = 1000, subreddit: Optional[str] = None) -> List[Dict]:
    """
    Collect top Reddit posts based on search query.
    Args:
        query: Search terms
        limit: Maximum number of posts to return
        subreddit: Optional specific subreddit to search
    Returns:
        List of post dictionaries with metadata
    """
    posts = []
    try:
        search_results = (REDDIT.subreddit(subreddit or "all")
                        .search(query, limit=limit, sort='top'))
        
        for post in search_results:
            posts.append({
                'title': post.title,
                'text': post.selftext,
                'score': post.score,
                'created_utc': post.created_utc,
                'id': post.id,
                'subreddit': post.subreddit.display_name,
                'url': f"https://reddit.com{post.permalink}",
                'num_comments': post.num_comments
            })
                
    except Exception as e:
        print(f"Error during search: {e}")
        return []
        
    return posts
