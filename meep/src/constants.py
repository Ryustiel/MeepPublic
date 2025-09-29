
SUMMARIZE_SIZE_THRESHOLD = 4000  # (Character number) Summarize messages when cumulated size is > this threshold
SUMMARIZE_DAYS_AGO_THRESHOLD = 2  # (N Days) Summarize messages older than this date

MAX_CONVERSATION_SIZE = 50000  # (Character number) Maximum number of characters from the conversation to the llm

CHANNEL_SIZE_THRESHOLD = 20000  # (Character number) Maximum size of a channel before messages get deleted
MINIMUM_CONTENT_SIZE_PER_SUMMARY = 300  # (Character number) Minimum size of content in a region to consider for summarization

# Tool Calling
QUICK_RESPONSE_TIME = 2  # seconds, after which a webhook will be sent if the response is not yet completed.

# Activity
DEFAULT_ACTIVITY = "conversing"
