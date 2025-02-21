"""
AI Story Generator Module

This module provides functionality to generate stories using AI, complete with chapters
and illustrations, and compile them into PDF format.

The module uses OpenAI's GPT-4 for text generation and DALL-E 3 for image generation.

Dependencies:
    - openai
    - langchain
    - pydantic
    - python-dotenv
    - logging
"""

import os
import asyncio
import logging
from typing import Optional, List
from dotenv import load_dotenv
from pydantic import BaseModel,ValidationError
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from openai import OpenAI
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Image as PDFImage,
    PageBreak,
    Spacer,
)
from io import BytesIO
import base64

class ImagePrompt(BaseModel):
    image_prompt: str

class Chapter(BaseModel):
    chapter_number: int 
    chapter_title: str
    chapter_content: str
    image_base64: Optional[str] = None

class Story(BaseModel):
    title: str
    chapters: List[Chapter]
    summary: str

"""
PDF Builder Module

This module provides functionality to create formatted PDF documents from story content,
including text and images. It uses ReportLab to generate professional-quality PDFs
with consistent styling and layout.

Dependencies:
    - reportlab
    - io
    - base64
    - output_structure (Chapter model)
"""
class PDFBuilder:
    """
    A class to build formatted PDF documents from story content.

    This class handles the creation of PDF documents with consistent styling,
    including title pages, chapters, and embedded images.

    Attributes:
        doc (SimpleDocTemplate): ReportLab document template
        styles (StyleSheet1): Collection of paragraph styles
        elements (List): List of flowable elements for the PDF

    Example:
        >>> pdf_builder = PDFBuilder("story.pdf")
        >>> pdf_builder.add_title_page("My Story")
        >>> pdf_builder.add_chapter(chapter_obj)
        >>> pdf_builder.build()
    """

    def __init__(self, filename: str) -> None:
        """
        Initialize the PDFBuilder with document settings.

        Args:
            filename (str): The output PDF filename

        Raises:
            ValueError: If filename is empty or invalid
        """
        if not filename.strip() or not filename.endswith('.pdf'):
            raise ValueError("Invalid PDF filename")

        self.doc = SimpleDocTemplate(
            filename,
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72,
        )
        self.styles = getSampleStyleSheet()
        self._create_custom_styles()
        self.elements: List = []

    def _create_custom_styles(self) -> None:
        """
        Create custom paragraph styles for the document.

        Defines styles for chapter titles and story titles with specific
        formatting parameters.
        """
        self.styles.add(
            ParagraphStyle(
                name="ChapterTitle",
                fontSize=18,
                leading=22,
                alignment=1,  # Center alignment
                spaceAfter=14,
                fontName="Helvetica-Bold",
            )
        )
        self.styles.add(
            ParagraphStyle(
                name="StoryTitle",
                fontSize=24,
                leading=28,
                alignment=1,  # Center alignment
                spaceAfter=24,
                fontName="Helvetica-Bold",
            )
        )

    def add_title_page(self, title: str) -> None:
        """
        Add a title page to the document.

        Args:
            title (str): The story title to display on the title page

        Raises:
            ValueError: If title is empty
        """
        if not title.strip():
            raise ValueError("Title cannot be empty")

        self.elements.append(Paragraph(title, self.styles["StoryTitle"]))
        self.elements.append(PageBreak())

    def add_chapter(self, chapter: Chapter) -> None:
        """
        Add a chapter to the document with its title, content, and optional image.

        Args:
            chapter (Chapter): Chapter object containing title, content, and optional image

        Raises:
            ValueError: If chapter object is invalid
            Exception: If image processing fails
        """
        if not isinstance(chapter, Chapter):
            raise ValueError("Invalid chapter object")

        # Add chapter title
        self.elements.append(
            Paragraph(
                f"Chapter {chapter.chapter_number}: {chapter.chapter_title}",
                self.styles["ChapterTitle"],
            )
        )

        # Process and add chapter image if available
        if chapter.image_base64:
            try:
                img_data = base64.b64decode(chapter.image_base64)
                img = PDFImage(BytesIO(img_data), width=5*inch, height=5*inch)
                img.hAlign = "CENTER"
                self.elements.append(img)
                self.elements.append(Spacer(1, 12))
            except Exception as e:
                raise Exception(f"Failed to process chapter image: {e}")

        self.elements.append(
            Paragraph(chapter.chapter_content, self.styles["BodyText"])
        )
        self.elements.append(PageBreak())

    def build(self) -> None:
        """
        Build the final PDF document.

        This method compiles all elements into the final PDF file.

        Raises:
            Exception: If PDF building fails
        """
        try:
            self.doc.build(self.elements)
        except Exception as e:
            raise Exception(f"Failed to build PDF: {e}")





load_dotenv()

class AIStoryGenerator:
    """
    A class to generate AI-powered stories with text and illustrations.

    This class handles the generation of story chapters, creation of corresponding
    illustrations, and compilation into a PDF format.

    Attributes:
        prompt (str): The main story prompt/theme.
        chapter_count (int): Number of chapters to generate.
        client (OpenAI): OpenAI client instance.
        llm (ChatOpenAI): LangChain ChatOpenAI instance.
        story (Optional[Story]): Generated story object.
        logger (logging.Logger): Logger instance for the class.

    Example:
        >>> generator = AIStoryGenerator("A sci-fi adventure", chapter_count=3)
        >>> await generator.generate_pdf("my_story.pdf")
    """

    def __init__(self, prompt: str, chapter_count: int = 3) -> None:
        """
        Initialize the AIStoryGenerator.

        Args:
            prompt (str): The main story prompt/theme.
            chapter_count (int, optional): Number of chapters. Defaults to 3.

        Raises:
            ValueError: If prompt is empty or chapter_count is less than 1.
        """
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty")
        if chapter_count < 1:
            raise ValueError("Chapter count must be at least 1")

        self.prompt = prompt
        self.chapter_count = chapter_count
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.llm = ChatOpenAI(
            model="gpt-4-1106-preview",
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        self.story: Optional[Story] = None
        
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    async def _generate_chapter_content(self, chapter_num: int) -> Optional[Chapter]:
        """
        Generate content for a single chapter using AI.

        Args:
            chapter_num (int): The chapter number to generate.

        Returns:
            Optional[Chapter]: Generated chapter object or None if generation fails.

        Raises:
            ValidationError: If the generated content doesn't match the Chapter model.
        """
        try:
            parser = JsonOutputParser(pydantic_object=Chapter)
            prompt = ChatPromptTemplate.from_template(
                """Create a detailed chapter for a story. Follow this structure:
                Chapter Number: {chapter_num}
                Chapter Title: [insert title]
                Chapter Content: [insert 3-5 paragraphs of content]
                
                Original Prompt: {prompt}
                {format_instructions}"""
            )
            
            chain = prompt | self.llm | parser
            result = await chain.ainvoke({
                "prompt": self.prompt,
                "chapter_num": chapter_num,
                "format_instructions": parser.get_format_instructions()
            })
            
            return Chapter.model_validate(result)
            
        except ValidationError as e:
            self.logger.error(f"Validation error: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Chapter generation failed: {e}")
            return None

    async def _generate_chapter_image(self, chapter: Chapter) -> Optional[str]:
        """
        Generate an illustration for a chapter using DALL-E.

        Args:
            chapter (Chapter): Chapter object containing content to illustrate.

        Returns:
            Optional[str]: Base64 encoded image string or None if generation fails.
        """
        try:
            if not isinstance(chapter, Chapter):
                raise ValueError("Invalid chapter type")
                
            image_prompt = (
                f"Detailed illustration for chapter {chapter.chapter_number}: "
                f"{chapter.chapter_title}. Scene: {chapter.chapter_content[:300]}"
            )
            
            response = await asyncio.to_thread(
                self.client.images.generate,
                model="dall-e-3",
                prompt=image_prompt,
                size="1024x1024",
                quality="hd",
                response_format="b64_json",
                n=1,
            )
            return response.data[0].b64_json
        except Exception as e:
            self.logger.error(f"Image generation failed: {e}")
            return None

    async def generate_story(self) -> Story:
        """
        Generate a complete story with chapters and illustrations.

        Returns:
            Story: Complete story object with chapters and images.

        Raises:
            Exception: If story generation fails.
        """
        chapters: List[Chapter] = []
        for chapter_num in range(1, self.chapter_count + 1):
            if chapter := await self._generate_chapter_content(chapter_num):
                if image_b64 := await self._generate_chapter_image(chapter):
                    chapter.image_base64 = image_b64
                chapters.append(chapter)
        
        summary_chain = ChatPromptTemplate.from_template(
            "Generate a 3-sentence summary for this story: {prompt}"
        ) | self.llm
        
        return Story(
            title=(await summary_chain.ainvoke({"prompt": self.prompt})).content,
            chapters=chapters,
            summary=(await summary_chain.ainvoke({"prompt": self.prompt})).content,
        )

    async def generate_pdf(self, filename: str = "storybook.pdf") -> None:
        """
        Generate a PDF document from the story.

        Args:
            filename (str, optional): Output PDF filename. Defaults to "storybook.pdf".

        Raises:
            Exception: If PDF generation fails.
        """
        if not self.story:
            self.story = await self.generate_story()
        
        pdf = PDFBuilder(filename)
        pdf.add_title_page(self.story.title)
        
        for chapter in self.story.chapters:
            pdf.add_chapter(chapter)
        
        pdf.build()
        self.logger.info(f"PDF generated successfully: {filename}")
