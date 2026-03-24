import asyncio
import os
import re
import feedparser
import aiohttp
import edge_tts
from openai import AsyncOpenAI
from dotenv import load_dotenv
from datetime import datetime

# Load configuration
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# Configuration parameters
VOICE = "en-US-AndrewNeural"
RATE = "-10%"  # 90% playback speed
OUTPUT_DIR = "uav_briefing_audio"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# List of suggested news sources
news_sources = [
    # --- ArduPilot & UAV Core ---
    {"name": "ArduPilot_Blog", "url": "https://discuss.ardupilot.org/c/blog.rss", "type": "tech"},
    {"name": "ArduPilot_GitHub", "url": "https://github.com/ArduPilot/ardupilot/commits/master.atom", "type": "code_change"},
    {"name": "PX4_Autopilot", "url": "https://px4.io/feed/", "type": "tech"},
    {"name": "sUAS_News", "url": "https://www.suasnews.com/feed/", "type": "industry"},

    # --- Autonomous Driving & Robotics Algorithms ---
    {"name": "IEEE_Spectrum_Robotics", "url": "https://spectrum.ieee.org/rss/robotics/fulltext", "type": "academic"},
    {"name": "The_Robot_Report", "url": "https://www.therobotreport.com/feed/", "type": "industry"},
    {"name": "DeepMind_Blog", "url": "https://deepmind.google/blog/rss.xml", "type": "ai_research"},

    # --- UCI Campus Dynamics ---
    {"name": "UCI_MAE_News", "url": "https://engineering.uci.edu/dept/mae/news/rss", "type": "campus"},
    
    # --- Geek Trends ---
    {"name": "Hacker_News_Top", "url": "https://rsshub.app/hackernews/topten", "type": "hacker"},
    {"name": "GitHub_Trending_CPP", "url": "https://rsshub.app/github/trending/daily/cpp", "type": "code_trend"}
]

class NewsBriefingGenerator:
    def __init__(self):
        self.sources = news_sources

    async def fetch_source(self, session, source):
        """Asynchronously fetch and parse a single RSS/Atom source."""
        try:
            async with session.get(source['url'], timeout=10) as response:
                if response.status == 200:
                    content = await response.text()
                    feed = feedparser.parse(content)
                    entries = []
                    # Fetch only the latest 3 entries to avoid context window overflow
                    for entry in feed.entries[:3]:
                        title = entry.get('title', 'No Title')
                        summary = entry.get('summary', entry.get('description', ''))
                        # Strip HTML tags
                        summary = re.sub(r'<[^>]+>', '', summary)
                        entries.append(f"[{source['name']}] {title}: {summary[:500]}")
                    return "\n".join(entries)
                else:
                    print(f"Failed to fetch {source['name']}: Status {response.status}")
                    return ""
        except Exception as e:
            print(f"Error fetching {source['name']}: {e}")
            return ""

    async def gather_all_news(self):
        """Fetch all news sources concurrently."""
        print(">>> Fetching intelligence from global UAV and robotics sources...")
        async with aiohttp.ClientSession() as session:
            tasks = [self.fetch_source(session, source) for source in self.sources]
            results = await asyncio.gather(*tasks)
            return "\n\n".join([r for r in results if r])

    async def summarize_to_briefing(self, content):
        """Generate a deep summary from a 'UAV Systems Architect' perspective (Podcast script style)."""
        print(">>> Performing hardcore technical extraction (de-AI redundancy)...")
        current_date = datetime.now().strftime("%Y-%m-%d")
        system_prompt = (
            "You are a Senior UAV Systems Architect with 9 years of FPV experience, "
            "an expert in ArduPilot, PX4, and MAVLink. Your mission is to provide a "
            "daily tech briefing for a UCI MAE Master's student.\n\n"
            "Style Requirements:\n"
            "1. HARDCORE: Focus on PID, EKF3, Control Allocation, SLAM, and Path Planning.\n"
            "2. DE-AI: Strip away marketing fluff and 'AI buzzwords' that lack physics backing.\n"
            "3. FORMAT: Podcast script style. Professional, calm, and data-driven.\n"
            "4. SECTIONS: \n"
            "   - Hardcore Title (e.g., EKF3 Stability & MAVLink Analysis)\n"
            "   - 5-Minute Core Summary: 3-5 key technical points.\n"
            "   - 'Pitfall Guide': Hardware/Firmware compatibility warnings.\n"
            "   - Reference Index: List source names.\n"
            "Language: Chinese (as requested for the briefing content), Tone: Professional Technical Chinese."
        )
        
        try:
            response = await client.chat.completions.create(
                model="gpt-4o", # Use GPT-4o for high-quality technical summarization
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Today's raw data ({current_date}):\n\n{content}"}
                ],
                temperature=0.3
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Summarization failed: {e}")
            return "Internal briefing error."

    async def generate_speech(self, text, filename):
        """Generate speech using en-US-AndrewNeural at 90% speed."""
        print(f">>> Generating Andrew's voice briefing [Speed: 90%]: {filename}")
        output_path = os.path.join(OUTPUT_DIR, filename)
        # Note: Andrew is an English voice but handles mixed Chinese/English text robustly in TTS.
        communicate = edge_tts.Communicate(text, VOICE, rate=RATE)
        await communicate.save(output_path)
        print(f">>> Briefing generated: {output_path}")

async def run_pipeline():
    generator = NewsBriefingGenerator()
    
    # 1. Fetch RSS feeds
    all_content = await generator.gather_all_news()
    if not all_content.strip():
        print("No news content found. Check internet or source URLs.")
        return

    # 2. Generate podcast script
    briefing_script = await generator.summarize_to_briefing(all_content)
    
    # 3. Text-to-Speech synthesis
    today_str = datetime.now().strftime("%Y%m%d")
    await generator.generate_speech(briefing_script, f"Briefing_{today_str}.mp3")

if __name__ == "__main__":
    if not OPENAI_API_KEY:
        print("Error: OPENAI_API_KEY is missing in .env")
    else:
        asyncio.run(run_pipeline())