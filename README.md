# AI Story Generator

## 📖 Project Overview
The **AI Story Generator** is a powerful tool that leverages OpenAI's GPT-4 and DALL-E 3 to create immersive, illustrated stories. It generates dynamic story chapters with AI-crafted content and images, compiling them into a professional-quality PDF.
## Checkout the samples it generated
- [Peeking Into Future](outputs/peeking_into_future.pdf)
- [Time Travel Adventure](outputs/time_travel_adventure.pdf)

## 🚀 Features
- **Story Generation:** Uses GPT-4 to generate engaging story chapters.
- **Image Generation:** Creates detailed illustrations for each chapter using DALL-E 3.
- **PDF Compilation:** Converts stories and images into a beautifully formatted PDF.
- **Custom Chapter Counts:** Generate stories with a flexible number of chapters.

## 🛠️ How It Works
1. **Initialize the Generator:** Provide a story prompt and specify the desired number of chapters.
2. **Generate Story Content:** GPT-4 generates detailed chapters based on the input prompt.
3. **Generate Images:** DALL-E 3 creates chapter-specific illustrations.
4. **Compile PDF:** The PDFBuilder class formats and compiles the story into a final PDF document.

## 🧰 Technologies Used
- **Python**
- **OpenAI GPT-4** for text generation
- **DALL-E 3** for image generation
- **LangChain** for prompt engineering
- **ReportLab** for PDF creation
- **pydantic** for data validation
- **asyncio** for asynchronous operations

## 📂 Project Structure
```
├── main.py                 # Entry point for story generation
├── outputs/                # generated stories pdf
├── scripts/
│   └── story_generator.py  # Core AI story and PDF generation logic
└── requirements.txt        # Project dependencies
```

## 🚦 Usage
```bash
# Install dependencies
pip install -r requirements.txt

# Run the story generator
python main.py
```

## 🧑‍💻 Example
```python
import asyncio
from scripts.story_generator import AIStoryGenerator

async def main() -> None:
    generator = AIStoryGenerator("An epic fantasy adventure", chapter_count=5)
    await generator.generate_pdf("fantasy_story.pdf")

asyncio.run(main())
```

## 📑 Requirements
- Python 3.8+
- OpenAI API key (Set in your .env file)

## ⚠️ Error Handling
The system includes robust error handling for:
- Invalid input data
- Failed API calls
- PDF generation issues



