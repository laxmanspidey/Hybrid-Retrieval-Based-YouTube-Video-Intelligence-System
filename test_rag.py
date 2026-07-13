import sys
import os

# Add the project directory to path
sys.path.append(r'c:\Users\Laxman Spidey\Downloads\claude')

from rag_engine import YouTubeRAG

# We need the video ID from the URL in the screenshot
# The URL in the screenshot is: https://www.youtube.com/watch?v=vJ1M_NbsC4 (wait, the 'M' could be 'M_' or similar)
# Let's just list the chroma_db directories to find the video ID.
import glob
collections = glob.glob(r'c:\Users\Laxman Spidey\Downloads\claude\chroma_db\video_*')
if not collections:
    print("No collections found")
    sys.exit(1)

video_id = os.path.basename(collections[0]).replace('video_', '')
print(f"Using video_id: {video_id}")

rag = YouTubeRAG(video_id)
result = rag.ask("which weapon is pure stealth")
print("ANSWER:")
print(repr(result['answer']))
