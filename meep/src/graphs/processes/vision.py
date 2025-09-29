"""
Functions to process links.
"""

from typing import Dict, List
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage

import re, httpx, pydantic, asyncio, yt_dlp
import graphs._llm as llm, graphs._data as graph_data, data.jsondb

class URLCache(pydantic.BaseModel):
    urls: Dict[str, str] = {}

URL_CACHE_DB = data.jsondb.JsonDB[URLCache]("./data/databases/url_cache.json", URLCache)


async def process_url(url: str) -> str:
    """
    Process an url and return the processed result.
    """
    message = "No additional information"

    # === IMAGES
    if (
        any(url.endswith(extension) for extension in ["png", "gif", "jpg", "jpeg"])
        or (
            r"cdn.discordapp.com/attachments/" in url
            and any(extension in url for extension in ["png", "gif", "jpg", "jpeg"])
        )
    ):

        """
        Describe the image at the given URL.
        """
        
        # Actual url processing
        try:
            desc: AIMessage = await llm.IMAGE.ainvoke(
                [
                    HumanMessage(
                        content=[
                            {"type": "text", "text": "Describe this image in a paragraph. Locate noteworthy elements relative to one another. Write down any text you may find and locate it."}, 
                            {"type": "image_url", "image_url": {"url": url}}
                        ]
                    )
                ])
            message = desc.content
        except Exception as e:
            message = f"Describe image failed. Error={str(e)}"
            
    # === SOUNDCLOUD
    elif "soundcloud.com" in url:
        
        """
        Get structured information about a SoundCloud song or playlist using yt-dlp.
        Returns a formatted string with quality metadata.
        """
        ydl_opts = {
            'quiet': True,
            'skip_download': True,
            'extract_flat': False,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                info = ydl.extract_info(url, download=False)
            except Exception as e:
                message = f"Error extracting info: {e}"

        # Format output for song or playlist
        if info.get('_type') == 'playlist':
            tracks = info.get('entries', [])
            track_list = '\n'.join([
                f"  - {t.get('title', 'Unknown Title')} by {t.get('uploader', 'Unknown Artist')} ({t.get('webpage_url', '')})"
                for t in tracks
            ])
            message = (
                f"Playlist: {info.get('title', 'Unknown Playlist')}\n"
                f"By: {info.get('uploader', 'Unknown Uploader')}\n"
                f"Tracks:\n{track_list}"
            )
        else:
            message = (
                f"Title: {info.get('title', 'Unknown Title')}\n"
                f"Artist: {info.get('uploader', 'Unknown Artist')}\n"
                f"Duration: {info.get('duration', 'Unknown')} seconds\n"
                f"URL: {info.get('webpage_url', url)}"
            )
            pass  # TODO : Get song details
        
    # === OTHER
    else:  # Get request and page summarization
        
        """
        Attempt a GET request and summarize the information at the page.
        """
        SIZE_LIMIT = 10000
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url)
                response.raise_for_status()
            
            page_content = response.text
            if len(page_content) > SIZE_LIMIT:
                page_content = page_content[:SIZE_LIMIT] + "..."

            desc: AIMessage = await llm.SUMMARIZE.ainvoke([
                SystemMessage(content=f"Summarize this page : {url} {page_content}")
            ])
            message = desc.content
        except Exception as e:
            message = f"Failed to inspect link. Error={str(e)}"

    return f"[{url} {message}]"



async def vision_process_current_channel(history: graph_data.History) -> graph_data.InternalUpdates:
    """Replace URLs in the current channel with processed content."""
    updates = graph_data.InternalUpdates()

    current_channel_messages = list(enumerate(history.get_current_channel().messages))

    # 1. Find all consecutive human messages since the last AI message.
    # And extract them from those messages.
    extracted_urls: Dict[int, List[str]] = {}  # Message Index: URLs
    for i, msg in reversed(current_channel_messages):
        if isinstance(msg, graph_data.HumanMessage_):  # XXX : Only human messages are considered for vision.
            urls = re.findall(r'(?<!\[)https?://\S+', msg.content)
            if urls:
                extracted_urls[i] = urls  # Map message index to its URLs.
        else:
            break

    url_cache = await URL_CACHE_DB.read()

    if extracted_urls:  # NOTE : => Implies there are messages with URLs so there is a current channel with messages.

        # 2. Load the URL cache and use it to preprocess some links using the cache
        url_cache = await URL_CACHE_DB.read()

        processed_urls_replacements: Dict[str, str] = {}
        url_process_queue: List[str] = []
        
        for _, urls in extracted_urls.items():
            for url in urls:
                if url in url_cache.urls:  # URL is known in cache
                    processed_urls_replacements[url] = url_cache.urls[url]
                else:
                    url_process_queue.append(url)

        # 3. Process the URLs that were not covered by the cache
        url_tasks = [
            process_url(url) 
            for url in url_process_queue
        ]

        processed_urls = await asyncio.gather(*url_tasks, return_exceptions=True)
        processed_urls_replacements.update({
            url: processed_url for url, processed_url 
            in zip(url_process_queue, processed_urls)
        })
        
        # 3.5 Add processed URLs to the cache for later use
        async with URL_CACHE_DB as db:
            for url, replacement in zip(url_process_queue, processed_urls):
                db.urls[url] = replacement

        # 4. Update all the URLs
        channel = history.get_current_channel()  # TODO : Create a "add update" method in the channel to handle this.
        updates = graph_data.InternalUpdates(channel_updates={channel.id: graph_data.InternalChannelUpdates()})

        for message_index, urls in extracted_urls.items():
            existing_message = channel.messages[message_index].model_copy(deep=True)

            for url, replacement in processed_urls_replacements.items():
                # Only replace URLs that were in the original message.
                if url in existing_message.content:
                    existing_message.content = existing_message.content.replace(url, replacement)
            updates.channel_updates[channel.id].message_updates[message_index] = existing_message

    return updates
