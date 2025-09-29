"""
Send a request to soundcloud given an url
and return details about either the playlist
or the associated song.
Playlist should yield all song urls.
"""


import yt_dlp

test_song_url = "https://soundcloud.com/jordanastra/ttsbf"
test_playlist_url = "https://soundcloud.com/raphael-nguyen-665476637/sets/harmonique"

def get_info(url: str) -> str:
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
            return f"Error extracting info: {e}"

    # Format output for song or playlist
    if info.get('_type') == 'playlist':
        tracks = info.get('entries', [])
        track_list = '\n'.join([
            f"  - {t.get('title', 'Unknown Title')} by {t.get('uploader', 'Unknown Artist')} ({t.get('webpage_url', '')})"
            for t in tracks
        ])
        return (
            f"Playlist: {info.get('title', 'Unknown Playlist')}\n"
            f"By: {info.get('uploader', 'Unknown Uploader')}\n"
            f"Tracks:\n{track_list}"
        )
    else:
        return (
            f"Title: {info.get('title', 'Unknown Title')}\n"
            f"Artist: {info.get('uploader', 'Unknown Artist')}\n"
            f"Duration: {info.get('duration', 'Unknown')} seconds\n"
            f"URL: {info.get('webpage_url', url)}"
        )
    
    
info = get_info(test_song_url)
print(info)
