import asyncio
from scripts.story_generator import AIStoryGenerator

async def main() -> None:
    """
    Main entry point for the story generation process.

    Example usage of the AIStoryGenerator class.
    """
    try:
        generator = AIStoryGenerator(
            prompt="A peek into the future where everthing is surprising.",
            chapter_count=4
        )
        
        await generator.generate_pdf("outputs/peeking_into_future.pdf")
    except Exception as e:
        print(f"Fatal error: {e}")

if __name__ == "__main__":
    asyncio.run(main())