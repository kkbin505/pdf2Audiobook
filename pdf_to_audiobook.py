import asyncio
import re
import os
import sys
import fitz  # PyMuPDF
from openai import AsyncOpenAI
import edge_tts
from tqdm.asyncio import tqdm
from dotenv import load_dotenv
import math

# Load environment variables
load_dotenv()

# Constants
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("Error: OPENAI_API_KEY not found in environment variables or .env file.")
    print("Please create a .env file with OPENAI_API_KEY=your_key_here")
    # We won't exit here to allow importing modules, but main will fail.

OUTPUT_DIR = "output_audiobook"
DEFAULT_VOICE = "en-US-JennyNeural"  # English Female voice
# DEFAULT_VOICE = "zh-CN-YunyangNeural" # Male voice

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

class PDFProcessor:
    """Handles PDF text extraction and splitting."""
    
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.doc = None

    def _open_doc(self):
        if not self.doc:
            self.doc = fitz.open(self.pdf_path)
        return self.doc

    def extract_text(self):
        """Extracts full text from the PDF."""
        try:
            doc = self._open_doc()
            text = ""
            for page in doc:
                text += page.get_text()
            return text
        except Exception as e:
            print(f"Error reading PDF: {e}")
            return None

    def extract_chapters_from_toc(self):
        """
        Attempts to split the PDF based on its Table of Contents (TOC).
        Returns a list of dicts: {'title': str, 'content': str}
        """
        doc = self._open_doc()
        toc = doc.get_toc() # [[lvl, title, page, dest], ...]
        
        if not toc:
            print("No Table of Contents found.")
            return []

        # Filter for top-level chapters (level 1) to match the user's description of "12 chapters"
        # Adjust level if needed, but level 1 is usually the main chapter.
        chapters = []
        
        # Filter logic: Keep only level 1, or level 1 and 2 if 1 is just "Part X"
        # For now, let's grab all level 1 items.
        level_1_toc = [item for item in toc if item[0] == 1]
        
        if len(level_1_toc) < 3: 
            # If too few level 1 items (e.g., just "Cover", "Dedication"), maybe level 2 is the real chapters?
            # Or if the user said 12 chapters and we found 12, great.
            # Let's fallback to using the original full TOC but filtering out very deep levels (e.g. > 2)
            # to avoid over-segmentation.
            filtered_toc = [item for item in toc if item[0] <= 2]
        else:
            filtered_toc = level_1_toc

        print(f"Found {len(filtered_toc)} chapters in TOC.")
        
        for i, (lvl, title, page_num, _) in enumerate(filtered_toc):
            # Page numbers in TOC are 1-based, fitz uses 0-based
            start_page = page_num - 1
            
            # End page is the start of the next chapter, or end of document
            if i + 1 < len(filtered_toc):
                end_page = filtered_toc[i+1][2] - 1
            else:
                end_page = len(doc)
            
            # Extract text from these pages
            chapter_text = ""
            for p in range(start_page, end_page):
                if p < len(doc):
                    chapter_text += doc[p].get_text()
            
            if len(chapter_text.strip()) > 100: # detailed content check
                clean_title = re.sub(r'[\\/*?:"<>|]', "", title)
                chapters.append({"title": clean_title, "content": chapter_text})
                
        return chapters

    def split_text(self, text, max_chars=10000, min_chars=2000):
        """
        Splits text into chapters/sections.
        Priority 1: TOC (if available and reliable).
        Priority 2: Regex patterns (strict) with fragment merging.
        Priority 3: Fixed-length chunks (fallback).
        """
        # 1. Try TOC first
        print("Attempting to split by TOC...")
        toc_chapters = self.extract_chapters_from_toc()
        if toc_chapters and len(toc_chapters) >= 3: # Reasonable number of chapters
            print(f"Successfully split into {len(toc_chapters)} chapters using TOC.")
            return toc_chapters
        
        print("TOC splitting unavailable or insufficient. Falling back to Regex/Text analysis.")
        
        # 2. Refined Regex (Engineering-level de-noising)
        # Targeted Pattern: "CHAPTER <digit>", "EXECUTIVE SUMMARY", "RECOMMENDATIONS", "第<digit>章"
        # We use re.MULTILINE logic via (^|\n)
        chapter_pattern = r"(^|\n)\s*(CHAPTER\s+\d+|EXECUTIVE SUMMARY|RECOMMENDATIONS|第[0-9]+章)\s*(?=\n|$)"
        
        matches = list(re.finditer(chapter_pattern, text, re.IGNORECASE)) # Ignore case for robustness
        chunks = []
        
        if matches:
             # Use regex matches to split
            last_pos = 0
            # Pre-calculate start indices to handle everything, including preamble
            
            # If the first match isn't at the beginning, treat previous text as 'Introduction'
            if matches[0].start() > 500:
                 chunks.append({"title": "Introduction", "content": text[:matches[0].start()].strip()})

            for i, match in enumerate(matches):
                start = match.start()
                # Determine end of current section
                end = matches[i+1].start() if i + 1 < len(matches) else len(text)
                
                # Title is the matched line
                title = match.group(0).strip().replace('\n', ' ')
                content = text[start:end].strip()
                
                # Clean title
                clean_title = re.sub(r'[\\/*?:"<>|]', "", title)
                chunks.append({"title": clean_title, "content": content})
            
            # --- Fragment Merging Logic ---
            merged_chunks = []
            if chunks:
                current_merge = chunks[0]
                
                for next_chunk in chunks[1:]:
                    # If current chunk is too small, merge it into the NEXT one?
                    # Or if NEXT chunk is too small, merge it into CURRENT one?
                    # The user req: "If content < min_chars, merge to previous."
                    # But the first chunk has no previous.
                    
                    # Let's process valid chunks list
                    if len(next_chunk['content']) < min_chars:
                        print(f"Merging small chunk '{next_chunk['title']}' ({len(next_chunk['content'])} chars) into '{current_merge['title']}'")
                        # Merge content
                        current_merge['content'] += "\n\n" + next_chunk['title'] + "\n\n" + next_chunk['content']
                        # Keep current_merge as the active one to potentially receive more small chunks
                    else:
                        # Push current_merge to list and start new
                        merged_chunks.append(current_merge)
                        current_merge = next_chunk
                
                # Append the last active chunk
                merged_chunks.append(current_merge)
                
                # Special check: if the VERY first chunk was small and got merged "forward" logic didn't exist above.
                # The prompt said: "merge to previous".
                # If chunk[i] is small, merge to chunk[i-1].
                
                # Let's re-implement strictly "merge to previous":
                final_chunks = []
                for chunk in chunks:
                    if not final_chunks:
                        final_chunks.append(chunk)
                        continue
                    
                    if len(chunk['content']) < min_chars:
                         print(f"Merging small chunk '{chunk['title']}' ({len(chunk['content'])} chars) into '{final_chunks[-1]['title']}'")
                         final_chunks[-1]['content'] += "\n\n" + chunk['title'] + "\n\n" + chunk['content']
                    else:
                        final_chunks.append(chunk)
                
                return final_chunks

        # 3. Fixed-length splitting (Fallback)
        print("No distinct chapters found using regex. Switching to fixed-length splitting.")
        
        words = text.split('\n')
        current_chunk = []
        current_length = 0
        
        for line in words:
            if current_length + len(line) > max_chars:
                chunks.append("\n".join(current_chunk))
                current_chunk = []
                current_length = 0
            current_chunk.append(line)
            current_length += len(line)
        if current_chunk:
            chunks.append("\n".join(current_chunk))
        
        return [{"title": f"Part {i+1}", "content": chunk} for i, chunk in enumerate(chunks)]

class AISummarizer:
    """Handles interaction with OpenAI API for summarization."""
    
    def __init__(self):
        self.client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        self.model = "gpt-4o-mini"

    async def summarize_chunk(self, text, title):
        """
        Summarizes the text using OpenAI GPT-4o-mini with specific prompt instructions.
        """
        system_prompt = (
            "You are an expert policy analyst and professional podcast host. Your task is to turn a technical research report "
            "into a deep-dive, engaging audio script.\n"
            "Requirements:\n"
            "1. **Structure**: Follow this exact flow: Introduction & Context -> Current Status Analysis -> Deep Dive into Risks (include specific case studies A/B/C) -> Policy Implications -> Future Outlook.\n"
            "2. **Content**: Do NOT summarize briefly. Retain specific examples, data points, and risk assessment details. Explain *why* things matter.\n"
            "3. **Tone**: Authoritative but conversational. Distinct from a dry text-to-speech read. Use transition phrases like 'Now, let's look at...' or 'What this means is...'.\n"
            "4. **Length**: Target 2000-3000 words (approx 15-20 mins spoken). Use full paragraphs.\n"
            "5. **Output**: Plain text only. No Markdown keys. No 'Here is the summary' preambles."
        )
        
        # Helper to call API
        async def call_api(prompt_text, part_info=""):
            user_prompt = f"Please analyze and expand the following section: 【{title} {part_info}】\n\nContent:\n{prompt_text}"
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.7
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"Error customizing text for {title}: {e}")
                return ""

        # Logic to handle large chapters > 15000 chars
        MAX_CHUNK_SIZE = 15000
        if len(text) > MAX_CHUNK_SIZE:
            print(f"Chapter '{title}' is large ({len(text)} chars). Splitting into parts...")
            # Calculate number of parts needed
            num_parts = math.ceil(len(text) / MAX_CHUNK_SIZE)
            chunk_length = math.ceil(len(text) / num_parts)
            
            parts = [text[i:i+chunk_length] for i in range(0, len(text), chunk_length)]
            
            full_summary = ""
            for i, part in enumerate(parts):
                print(f"  - Processing Part {i+1}/{num_parts}...")
                part_summary = await call_api(part, part_info=f"(Part {i+1} of {num_parts})")
                full_summary += f"\n\n[Part {i+1}]\n" + part_summary
            
            return full_summary
            
        else:
            return await call_api(text)

class TTSGenerator:
    """Handles text-to-speech generation using edge-tts."""
    
    def __init__(self, voice=DEFAULT_VOICE):
        self.voice = voice

    async def generate_audio(self, text, output_filename):
        """Generates MP3 audio from text."""
        if not text or len(text.strip()) == 0:
            return False
            
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        try:
            communicate = edge_tts.Communicate(text, self.voice)
            await communicate.save(output_path)
            return True
        except Exception as e:
            print(f"Error generating audio for {output_filename}: {e}")
            return False

async def process_chapter(chapter, summarizer, tts, index):
    """Process a single chapter: Summarize -> TTS."""
    title = chapter['title']
    content = chapter['content']
    
    # AI Summarization
    # print(f"Summarizing: {title}...") # Optional logging
    summary = await summarizer.summarize_chunk(content, title)
    
    if not summary:
        # print(f"Skipping {title} (no summary generated).")
        return False

    # TTS Generation
    # Validated filename format: Chapter_01_Title.mp3
    filename = f"Chapter_{index+1:02d}_{title[:20].strip().replace(' ', '_')}.mp3"
    # print(f"Generating Audio: {filename}...") # Optional logging
    success = await tts.generate_audio(summary, filename)
    
    return success

async def main():
    if len(sys.argv) < 2:
        print("Usage: python pdf_to_audiobook.py <input_pdf_path>")
        return

    pdf_path = sys.argv[1]
    if not os.path.exists(pdf_path):
        print(f"Error: File {pdf_path} not found.")
        return

    if not OPENAI_API_KEY:
        print("Stopping: OpenAI API Key missing.")
        return

    print(f"Processing PDF: {pdf_path}")
    
    # 1. Extract and Split
    processor = PDFProcessor(pdf_path)
    full_text = processor.extract_text()
    if not full_text:
        return
        
    chapters = processor.split_text(full_text)
    print(f"Found/Split into {len(chapters)} chapters/sections.")
    
    # 2. Initialize AI and TTS
    summarizer = AISummarizer()
    tts = TTSGenerator()

    # 3. Process Async
    print("Starting AI Summarization and Audio Generation...")
    
    # Using tqdm for progress tracking
    tasks = []
    for i, chapter in enumerate(chapters):
        tasks.append(process_chapter(chapter, summarizer, tts, i))
    
    # Run all tasks concurrently? 
    # Be careful with rate limits. A semaphore might be needed if chapters are many.
    # For simplicity and order, we can run them with a small semaphore or just gather.
    # Let's use a semaphore to be safe with OpenAI rate limits.
    
    semaphore = asyncio.Semaphore(7) # Max 7 concurrent OpenAI calls
    
    async def fast_run(task):
        async with semaphore:
            return await task
            
    wrapped_tasks = [fast_run(t) for t in tasks]
    
    results = []
    for f in tqdm(asyncio.as_completed(wrapped_tasks), total=len(wrapped_tasks), desc="Processing Chapters"):
        res = await f
        results.append(res)
        
    print(f"Done! {sum(results)} chapters converted successfully.")
    print(f"Check the '{OUTPUT_DIR}' folder.")

if __name__ == "__main__":
    # Windows-specific asyncio policy fix if needed, though usually python 3.8+ handles it
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
    asyncio.run(main())
