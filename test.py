from youtube_transcript_api import YouTubeTranscriptApi
import json

# Extract the video ID from the URL
video_url = 'https://www.youtube.com/watch?v=vStJoetOxJg&list=PLkDaE6sCZn6FNC6YRfRQc_FbeQrF8BwGI'
video_id = video_url.split('v=')[1].split('&')[0]

try:
    # Fetching the subtitles/transcript using the video ID
    transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
    
    # Define the filename where you want to save the subtitles
    filename = f"{video_id}_subtitles.json"
    
    # Writing the subtitles to a JSON file
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(transcript, f, ensure_ascii=False, indent=4)
    
    print(f"Subtitles have been saved to {filename}")
except Exception as e:
    print(f"An error occurred: {e}")