"""
Tools and utilities for the SmolAgents
"""
import os
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from PIL import Image
import pymupdf as fitz
from smolagents import tool, CodeAgent
from smolagents.agents import ActionStep

class PDFProcessor:
    """
    A class to handle PDF processing with batched image loading
    """
    def __init__(self, max_batch_size: int = 15):
        """
        Initialize the PDF processor
        
        Args:
            max_batch_size: Maximum number of pages to process in a batch
        """
        self.max_batch_size = max_batch_size
        self.current_pdf_path = None
        self.doc = None
        self.total_pages = 0
        self.current_batch_start = 0
    
    def open_pdf(self, pdf_path: str) -> str:
        """
        Open a PDF file and prepare it for processing
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Information about the PDF
        """
        try:
            # Close any previously open document
            if self.doc:
                self.doc.close()
            
            self.current_pdf_path = pdf_path
            self.doc = fitz.open(pdf_path)
            self.total_pages = len(self.doc)
            self.current_batch_start = 0
            
            return f"PDF opened successfully. Total pages: {self.total_pages}"
        except Exception as e:
            return f"Error opening PDF: {str(e)}"
    
    def get_next_batch_of_images(self) -> Tuple[List[Image.Image], Dict[str, Any]]:
        """
        Get the next batch of images from the PDF
        
        Returns:
            Tuple containing:
            - List of PIL Images
            - Dict with metadata about the batch
        """
        if not self.doc:
            return [], {"error": "No PDF file is currently open"}
        
        if self.current_batch_start >= self.total_pages:
            return [], {"message": "End of PDF reached", "current_batch": None}
        
        batch_end = min(self.current_batch_start + self.max_batch_size, self.total_pages)
        
        images = []
        for page_num in range(self.current_batch_start, batch_end):
            page = self.doc.load_page(page_num)
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better resolution
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(img)
        
        batch_info = {
            "batch_start_page": self.current_batch_start + 1,  # 1-indexed for human readability
            "batch_end_page": batch_end,
            "total_pages": self.total_pages,
            "is_last_batch": batch_end >= self.total_pages
        }
        
        # Update for next batch
        self.current_batch_start = batch_end
        
        return images, batch_info

# Create PDF processor instance to be used by tools
pdf_processor = PDFProcessor()

# Define tools for the agent
@tool
def open_pdf(pdf_path: str) -> str:
    """
    Open a PDF file and prepare it for batch processing.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Information about the PDF
    """
    return pdf_processor.open_pdf(pdf_path)

@tool
def get_next_pdf_batch() -> Dict[str, Any]:
    """
    Get the next batch of pages from the currently open PDF.
    The images will be added to the agent's observations automatically.
    
    Returns:
        Information about the current batch of pages
    """
    images, batch_info = pdf_processor.get_next_batch_of_images()
    
    # Images will be added to the observations via the callback
    # Store them temporarily in the processor
    pdf_processor.current_batch_images = images
    
    return batch_info

@tool
def read_file(file_path: str) -> str:
    """
    Read a file and return its contents.
    
    Args:
        file_path: Path to the file
        
    Returns:
        The contents of the file as a string
    """
    try:
        with open(file_path, 'r') as file:
            return file.read()
    except Exception as e:
        return f"Error reading file: {str(e)}"

@tool
def write_file(content: str, file_path: str) -> str:
    """
    Write content to a file.
    
    Args:
        content: The content to save
        file_path: Path where the file should be saved
        
    Returns:
        Confirmation message
    """
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'w') as file:
            file.write(content)
        return f"Successfully saved content to {file_path}"
    except Exception as e:
        return f"Error saving file: {str(e)}"

@tool
def list_directory(directory_path: str) -> List[str]:
    """
    List all files and directories in the specified path.
    
    Args:
        directory_path: Path to the directory
        
    Returns:
        List of file and directory names
    """
    try:
        return os.listdir(directory_path)
    except Exception as e:
        return f"Error listing directory: {str(e)}"

def add_pdf_images_callback(step_log: ActionStep, agent: CodeAgent) -> None:
    """
    Callback to add PDF images to the agent's observations after certain actions.
    
    Args:
        step_log: The current step log
        agent: The agent instance
    """
    # Check if we have images from a get_next_pdf_batch tool call
    if hasattr(pdf_processor, 'current_batch_images') and pdf_processor.current_batch_images:
        images = pdf_processor.current_batch_images
        if images:
            step_log.observations_images = images
            # Add text to observations about the images
            image_info = f"\nAdded {len(images)} PDF page images to observations (pages {pdf_processor.current_batch_start - len(images) + 1}-{pdf_processor.current_batch_start})"
            step_log.observations = (step_log.observations or "") + image_info
            
            # Clear the images to avoid duplicating them in future steps
            pdf_processor.current_batch_images = []