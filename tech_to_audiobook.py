import asyncio
import os
import re
import feedparser
import aiohttp
import edge_tts
from openai import AsyncOpenAI
from dotenv import load_dotenv
from datetime import datetime

# 加载配置
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# 配置参数
VOICE = "en-US-AndrewNeural"
RATE = "-10%"  # 90% 语速
OUTPUT_DIR = "uav_briefing_audio"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 建议抓取源列表
news_sources = [
    # --- ArduPilot & UAV 核心层 ---
    {"name": "ArduPilot_Blog", "url": "https://discuss.ardupilot.org/c/blog.rss", "type": "tech"},
    {"name": "ArduPilot_GitHub", "url": "https://github.com/ArduPilot/ardupilot/commits/master.atom", "type": "code_change"},
    {"name": "PX4_Autopilot", "url": "https://px4.io/feed/", "type": "tech"},
    {"name": "sUAS_News", "url": "https://www.suasnews.com/feed/", "type": "industry"},

    # --- 自动驾驶与机器人算法 ---
    {"name": "IEEE_Spectrum_Robotics", "url": "https://spectrum.ieee.org/rss/robotics/fulltext", "type": "academic"},
    {"name": "The_Robot_Report", "url": "https://www.therobotreport.com/feed/", "type": "industry"},
    {"name": "DeepMind_Blog", "url": "https://deepmind.google/blog/rss.xml", "type": "ai_research"},

    # --- 尔湾校内动态 ---
    {"name": "UCI_MAE_News", "url": "https://engineering.uci.edu/dept/mae/news/rss", "type": "campus"},
    
    # --- 极客趋势 ---
    {"name": "Hacker_News_Top", "url": "https://rsshub.app/hackernews/topten", "type": "hacker"},
    {"name": "GitHub_Trending_CPP", "url": "https://rsshub.app/github/trending/daily/cpp", "type": "code_trend"}
]

class NewsBriefingGenerator:
    def __init__(self):
        self.sources = news_sources

    async def fetch_source(self, session, source):
        """异步抓取并解析单个 RSS/Atom 源"""
        try:
            async with session.get(source['url'], timeout=10) as response:
                if response.status == 200:
                    content = await response.text()
                    feed = feedparser.parse(content)
                    entries = []
                    # 仅取最近的 3 条，避免 context 爆炸
                    for entry in feed.entries[:3]:
                        title = entry.get('title', 'No Title')
                        summary = entry.get('summary', entry.get('description', ''))
                        # 去除 HTML 标签
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
        """并发抓取所有新闻源"""
        print(">>> 正在从全球无人机及机器人技术源站抓取情报...")
        async with aiohttp.ClientSession() as session:
            tasks = [self.fetch_source(session, source) for source in self.sources]
            results = await asyncio.gather(*tasks)
            return "\n\n".join([r for r in results if r])

    async def summarize_to_briefing(self, content):
        """采用‘无人机架构师’视角进行深度总结（播客脚本风格）"""
        print(">>> 正在进行硬核技术脱水总结 (算法去 AI 冗余)...")
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
                model="gpt-4o", # 使用主流模型保证总结质量
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
        """使用 en-US-AndrewNeural 生成语音，语速 90%"""
        print(f">>> 正在生成 Andrew 语音简报 [Speed: 90%]: {filename}")
        output_path = os.path.join(OUTPUT_DIR, filename)
        # 注意：Andrew 虽然是英文人声，但他对中英混排的处理在 TTS 中表现稳健
        communicate = edge_tts.Communicate(text, VOICE, rate=RATE)
        await communicate.save(output_path)
        print(f">>> 简报已生成: {output_path}")

async def run_pipeline():
    generator = NewsBriefingGenerator()
    
    # 1. 抓取 RSS
    all_content = await generator.gather_all_news()
    if not all_content.strip():
        print("No news content found. Check internet or source URLs.")
        return

    # 2. 生成播客脚本
    briefing_script = await generator.summarize_to_briefing(all_content)
    
    # 3. 语音合成
    today_str = datetime.now().strftime("%Y%m%d")
    await generator.generate_speech(briefing_script, f"Briefing_{today_str}.mp3")

if __name__ == "__main__":
    if not OPENAI_API_KEY:
        print("Error: OPENAI_API_KEY is missing in .env")
    else:
        asyncio.run(run_pipeline())