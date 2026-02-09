# PDF to Audiobook Converter

A powerful Python tool that converts PDF research reports into engaging, deep-dive audiobooks. It uses **PyMuPDF** for text extraction, **OpenAI GPT-4o-mini** for intelligent "Expand Mode" summarization, and **Edge TTS** for high-quality voice synthesis.

## Features

- **Smart Chapter Splitting**:
  - Automatically detects chapters using the PDF Table of Contents (TOC).
  - Falls back to a robust regex ("Chapter X", "Executive Summary", etc.) if no TOC is found.
  - Merges small fragments (< 2000 chars) to ensure smooth continuity.
- **Deep-Dive Summarization (Expand Mode)**:
  - Transforms dry reports into engaging "Podcast Style" scripts.
  - Follows a strict structure: *Intro -> Status -> Risks -> Policy -> Future*.
  - Handlings large chapters (> 15k chars) by splitting and processing parts specifically to avoid token limits.
- **High-Quality Audio**:
  - Uses Microsoft Edge TTS for natural-sounding speech.
  - Sort-friendly filenames (e.g., `Chapter_01_...`) for seamless playback on mobile apps.
- **Async Processing**:
  - Processes chapters concurrently for faster execution.

## Prerequisites

- Python 3.8+
- An OpenAI API Key

## Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/pdf-to-audiobook.git
    cd pdf-to-audiobook
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: `requirements.txt` should include `pymupdf`, `openai`, `edge-tts`, `tqdm`, `python-dotenv`)*

3.  **Setup Environment Variables**:
    Create a `.env` file in the root directory:
    ```env
    OPENAI_API_KEY=sk-your-api-key-here
    ```

## Usage

Run the script with the path to your PDF file:

```bash
python pdf_to_audiobook.py "path/to/your/report.pdf"
```

The audio files will be generated in the `output_audiobook/` folder.

## Customization

- **Voice**: Change `DEFAULT_VOICE` in `pdf_to_audiobook.py` to any Edge TTS voice (e.g., `zh-CN-YunyangNeural` for Chinese).
- **Prompt**: Modify `system_prompt` in the `AISummarizer` class to tweak the podcast persona or structure.

## License

MIT
