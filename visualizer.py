import pandas as pd
import plotly.express as px
from IPython.display import HTML, display
from collections import Counter
import itertools
import numpy as np

class Visualization:
    def __init__(self, df: pd.DataFrame, fields_schema: dict):
        """
        Args:
            df: DataFrame with processed data (including extracted fields).
            fields_schema: Schema dict for extracted fields.
        """
        self.df = df
        self.fields_schema = fields_schema

    def display_section(self, title: str):
        display(HTML(f'<div style="margin:10px 0;"><h4 style="color:#333;margin:0;padding:5px 0;border-bottom:1px solid #ccc;">{title}</h4></div>'))

    def plot_all_fields(self, show_examples=True, save=False, metadata_fields=None, record_fields=None, title_field=None, extra_fields=None):
        """
        Plot all extracted fields by type.
        
        show_examples: If True, show sample examples after each plot
        metadata_fields: keys in metadata to show in sample cards
        record_fields: df columns to show in sample cards
        title_field: field used as a title if available
        extra_fields: additional df fields to show in sample cards
        """
        if extra_fields:
            record_fields = (record_fields or []) + extra_fields

        for field_name, field_info in self.fields_schema.items():
            ftype = field_info['type']
            if ftype == 'boolean':
                self.plot_binary_distribution(field_name, f"Distribution of {field_name}", show_examples, save, metadata_fields, record_fields, title_field)
            elif ftype == 'integer':
                self.plot_integer_distribution(field_name, f"{field_name} Distribution", show_examples, save, metadata_fields, record_fields, title_field)
            elif ftype == 'array':
                self.plot_list_field(field_name, f"Most Common {field_name.capitalize()}", show_examples=show_examples, save=save, 
                                     metadata_fields=metadata_fields, record_fields=record_fields, title_field=title_field)

    def plot_binary_distribution(self, field_name, title, show_examples=False, save=False, metadata_fields=None, record_fields=None, title_field=None):
        counts = self.df[field_name].value_counts(dropna=False)
        percentages = (counts / len(self.df)) * 100
        fig = px.pie(values=percentages.values, names=percentages.index.astype(str), title=title)
        fig.update_traces(texttemplate='%{value:.1f}%')
        self._save_fig(fig, title, save)
        fig.show()

        if show_examples and field_name in self.df.columns:
            for value in counts.index:
                self.display_section(f"Example Samples for {field_name} = {value}")
                subset = self.df[self.df[field_name] == value]
                if len(subset) > 0:
                    example_posts = subset.sample(min(3, len(subset)))
                    html_output = '<div style="display:flex;flex-wrap:wrap;">'
                    for _, post in example_posts.iterrows():
                        html_output += self.format_sample(post, metadata_fields=metadata_fields, record_fields=record_fields, title_field=title_field)
                    html_output += "</div>"
                    display(HTML(html_output))

    def plot_integer_distribution(self, field_name, title, show_examples=False, save=False, metadata_fields=None, record_fields=None, title_field=None):
        valid_data = self.df[self.df[field_name].notnull()]
        if valid_data.empty:
            return
        fig = px.histogram(valid_data, x=field_name, title=title, nbins=10)
        fig.update_traces(histnorm='percent')
        
        mean_val = valid_data[field_name].mean()
        fig.add_vline(x=mean_val, line_dash="dash", line_color="red",
                      annotation_text=f"Mean: {mean_val:.2f}",
                      annotation_position="top right")
        
        fig.update_layout(yaxis_title="Percent")
        self._save_fig(fig, title, save)
        fig.show()

        if show_examples:
            self.display_section(f"Example Samples with {field_name}")
            example_posts = valid_data.sample(min(3, len(valid_data)))
            html_output = '<div style="display:flex;flex-wrap:wrap;">'
            for _, post in example_posts.iterrows():
                html_output += self.format_sample(post, metadata_fields=metadata_fields, record_fields=record_fields, title_field=title_field)
            html_output += "</div>"
            display(HTML(html_output))

    def plot_list_field(self, field_name, title, limit=10, show_examples=False, save=False, metadata_fields=None, record_fields=None, title_field=None):
        if field_name not in self.df.columns:
            return
        all_items = []
        for val in self.df[field_name].dropna():
            if isinstance(val, list):
                all_items.extend(val)
        
        if not all_items:
            return

        item_counts = Counter(all_items).most_common(limit)
        df_counts = pd.DataFrame(item_counts, columns=[field_name, 'count'])
        total_posts = len(self.df)
        df_counts['percent'] = df_counts['count'].apply(lambda x: (x / total_posts) * 100)
        
        fig = px.bar(df_counts, x=field_name, y='percent', title=title)
        fig.update_layout(yaxis_title="Percent of Posts")
        self._save_fig(fig, title, save)
        fig.show()

        if show_examples:
            self.display_section(f"Example Samples with {field_name}")
            top_item = df_counts[field_name].iloc[0]
            subset = self.df[self.df[field_name].apply(lambda x: isinstance(x, list) and top_item in x)]
            if len(subset) > 0:
                example_posts = subset.sample(min(3, len(subset)))
                html_output = '<div style="display:flex;flex-wrap:wrap;">'
                for _, post in example_posts.iterrows():
                    html_output += self.format_sample(post, metadata_fields=metadata_fields, record_fields=record_fields, title_field=title_field)
                html_output += "</div>"
                display(HTML(html_output))

    def plot_by_time(self, time_field: str, title: str, freq='M', save=False):
        if time_field not in self.df.columns:
            return
        if not pd.api.types.is_datetime64_any_dtype(self.df[time_field]):
            return
        counts = self.df[time_field].dt.to_period(freq).value_counts().sort_index()
        fig = px.line(x=counts.index.astype(str), y=counts.values, title=title, labels={'x': 'Time', 'y': 'Count'})
        self._save_fig(fig, title, save)
        fig.show()

    def plot_correlation(self, numeric_fields: list, title="Correlation Matrix", save=False):
        if not numeric_fields:
            return
        numeric_df = self.df[numeric_fields].dropna()
        if numeric_df.empty:
            return
        corr = numeric_df.corr()
        fig = px.imshow(corr, text_auto=True, title=title, color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
        self._save_fig(fig, title, save)
        fig.show()

    def plot_group_comparison(self, group_field: str, value_field: str, agg='mean', title=None, save=False):
        if group_field not in self.df.columns or value_field not in self.df.columns:
            return
        if agg not in ['mean', 'count', 'sum', 'median']:
            agg = 'mean'
        grouped = self.df.groupby(group_field)[value_field]
        if agg == 'mean':
            result = grouped.mean().dropna()
        elif agg == 'count':
            result = grouped.count()
        elif agg == 'sum':
            result = grouped.sum().dropna()
        elif agg == 'median':
            result = grouped.median().dropna()

        if result.empty:
            return
        t = title or f"{agg.capitalize()} of {value_field} by {group_field}"
        fig = px.bar(x=result.index.astype(str), y=result.values, title=t, labels={'x':group_field,'y':f"{agg}({value_field})"})
        self._save_fig(fig, t, save)
        fig.show()

    def show_samples(self, n=5, metadata_fields=None, record_fields=None, title_field=None, extra_fields=None):
        if extra_fields:
            record_fields = (record_fields or []) + extra_fields

        if self.df.empty:
            print("No samples to display.")
            return
        samples = self.df.sample(min(n, len(self.df)))
        self.display_section(f"Showing {len(samples)} Random Samples")
        html_output = '<div style="display:flex;flex-wrap:wrap;">'
        for _, post in samples.iterrows():
            html_output += self.format_sample(post, metadata_fields=metadata_fields, record_fields=record_fields, title_field=title_field)
        html_output += "</div>"
        display(HTML(html_output))

    def format_sample(self, post, metadata_fields=None, record_fields=None, title_field=None):
        """
        Format a single sample.
        
        Detection:
        - If comment: combined_text starts with "Comment:"
          Show "[Comment]" in title
          Show comment fully
          Show original post truncated to a few lines.
        - If post: just show post content with "Content:"

        Title:
        - If title_field given and found in post or metadata, use that as title.
        - Otherwise try post.get('title').
        - If no title, use "No title".
        - If comment, prepend "[Comment] " to the title.

        URL:
        - Always check post['metadata']['url'] if it exists.
        - If found, make title clickable.

        Fields:
        - metadata_fields: show these keys from metadata
        - record_fields: show these fields from post
        """

        metadata = post.get('metadata', {})

        # Determine if comment
        combined_text = post.get('combined_text', '')
        is_comment = combined_text.startswith("Comment:")

        # Title logic
        title = None
        if title_field:
            # Try in post first
            if title_field in post and pd.notna(post[title_field]):
                title = str(post[title_field])
            # Try in metadata if still no title
            elif title_field in metadata and pd.notna(metadata[title_field]):
                title = str(metadata[title_field])

        if not title:
            # try post['title']
            if 'title' in post and pd.notna(post['title']):
                title = str(post['title'])
            else:
                title = "No title"

        if is_comment:
            title = "[Comment] " + title

        # URL logic - always check metadata['url'] if present
        url = metadata.get('url')
        if pd.isna(url):
            url = None

        # Gather metadata fields
        meta_info = []
        if metadata_fields:
            for mf in metadata_fields:
                val = metadata.get(mf)
                if val is not None and pd.notna(val):
                    if isinstance(val, list):
                        val = ", ".join(val)
                    meta_info.append(f"{mf}: {val}")

        # Gather record fields
        rec_info = []
        if record_fields:
            for rf in record_fields:
                if rf in post and pd.notna(post[rf]):
                    val = post[rf]
                    if isinstance(val, list):
                        val = ", ".join(val)
                    rec_info.append(f"{rf}: {val}")

        # Parse comment/post text
        comment_text = None
        original_post_text = None
        if is_comment:
            # Split by "In Response to Original Post:"
            parts = combined_text.split("In Response to Original Post:")
            comment_text = parts[0].replace("Comment:", "").strip()
            if len(parts) > 1:
                original_post_text = parts[1].strip()
                # Truncate original post to a few lines
                lines = original_post_text.split('\n')
                truncated_lines = lines[:3]
                original_post_text = '\n'.join(truncated_lines)
        else:
            original_post_text = combined_text.strip()

        # Build HTML
        html_parts = []
        html_parts.append(f"<div style='border:1px solid #ddd; border-radius:8px; padding:15px; margin:10px 5px; background-color:#f9f9f9; display:inline-block; vertical-align:top; width:320px; margin-bottom:10px;'>")
        
        # ID
        html_parts.append(f"<div style='color:#333; font-size:0.9em; margin-bottom:5px;'>ID: {post['id']}</div>")

        # Meta info
        if meta_info:
            html_parts.append(f"<div style='color:#666; margin-bottom:10px;'>{' | '.join(meta_info)}</div>")
        # Record info
        if rec_info:
            html_parts.append(f"<div style='color:#666; margin-bottom:10px;'>{' | '.join(rec_info)}</div>")

        # Title line
        if url and url is not None:
            html_parts.append(f"<div style='color:#333; font-size:1.1em; font-weight:bold; margin-bottom:10px;'><a href='{url}' target='_blank' style='text-decoration:none; color:inherit;'>{title}</a></div>")
        else:
            html_parts.append(f"<div style='color:#333; font-size:1.1em; font-weight:bold; margin-bottom:10px;'>{title}</div>")

        # Content
        if is_comment and comment_text:
            comment_html = f"<div style='margin-bottom:10px;'><strong>Comment:</strong><br>{self._truncate_text(comment_text)}</div>"
            html_parts.append(comment_html)

        if original_post_text is not None:
            label = "Original Post:" if is_comment else "Content:"
            html_parts.append(f"<div style='margin-bottom:10px;'><strong>{label}</strong><br>{self._truncate_text(original_post_text)}</div>")

        html_parts.append("</div>")
        return "".join(html_parts)

    def _truncate_text(self, text, length=500):
        return (text[:length] + '...') if len(text) > length else text

    def _save_fig(self, fig, title, save):
        if save:
            import os
            if not os.path.exists('plots'):
                os.makedirs('plots')
            fig.write_image(f"plots/{title.lower().replace(' ', '_')}.png")
