"""
PDF report generation module for regression modeling.

This module provides a comprehensive report generator for regression models,
supporting rich visualizations, detailed metrics, and profitability analysis.
"""

import os
import tempfile
import json
from datetime import datetime
from typing import List, Dict, Optional, Union, Tuple, Any
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
    PageBreak, Flowable, HRFlowable, ListFlowable, ListItem
)
import matplotlib.pyplot as plt
from PIL import Image as PILImage
import numpy as np
import pandas as pd

# Import style and visualization modules
from .styles import (
    create_styles,
    OAF_GREEN, OAF_BLUE, OAF_LIGHT_GREEN, OAF_LIGHT_BLUE, OAF_DARK_GREEN,
    TUPANDE_BLUE, TUPANDE_RED, TUPANDE_GREEN
)

class RegressionReport:
    """
    Class to generate PDF reports for regression modeling results.
    """
    def __init__(
        self, 
        output_path: str, 
        title: str = "Loan Repayment Rate Regression Model Report",
        subtitle: str = "OneAcre Fund - Tupande Credit Risk Analysis",
        author: str = "Data Science Team",
        logo_path: Optional[str] = None
    ):
        """
        Initialize the report generator.
        
        Args:
            output_path: Path where the PDF report will be saved
            title: Title of the report
            subtitle: Subtitle of the report
            author: Author or team name
            logo_path: Optional path to logo image
        """
        self.output_path = output_path
        self.title = title
        self.subtitle = subtitle
        self.author = author
        self.logo_path = logo_path
        self.elements = []
        self.temp_files = []  # Track temporary files for cleanup
        self.styles = create_styles()
        self.chapter_count = 0
        self.section_count = 0
        
        # Initialize document
        self.doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            rightMargin=0.5*inch,
            leftMargin=0.5*inch,
            topMargin=0.5*inch,
            bottomMargin=0.5*inch,
            title=title,
            author=author
        )
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Add title page
        self._add_title_page()
        
        # Add table of contents placeholder
        self._add_toc_placeholder()

    def _add_title_page(self):
        """Add the title page to the report."""
        # Add significant vertical space to center content
        self.elements.append(Spacer(1, 2*inch))
        
        # Add logo if provided
        if self.logo_path and os.path.exists(self.logo_path):
            img = Image(self.logo_path, width=3*inch, height=1.5*inch)
            self.elements.append(img)
            self.elements.append(Spacer(1, 0.5*inch))
        
        # Create title
        title = Paragraph(self.title, self.styles['ReportTitle'])
        
        # Add subtitle
        subtitle = Paragraph(self.subtitle, self.styles['ReportSubtitle'])
        
        # Add author
        author = Paragraph(f"Prepared by: {self.author}", self.styles['ReportAuthor'])
        
        # Format date with more details
        date_text = Paragraph(
            f"Generated on: {datetime.now().strftime('%B %d, %Y %H:%M:%S')}",
            self.styles['ReportAuthor']
        )
        
        # Add all elements
        self.elements.extend([
            title,
            Spacer(1, 0.25*inch),
            subtitle,
            Spacer(1, 1*inch),
            author,
            Spacer(1, 0.25*inch),
            date_text
        ])
        
        # Add a horizontal rule
        self.elements.append(Spacer(1, 1*inch))
        self.elements.append(HRFlowable(width="80%", thickness=2, color=OAF_LIGHT_GREEN, 
                                         lineCap='square', spaceBefore=15, spaceAfter=15))
        
        # Add a confidentiality notice
        confidentiality = Paragraph(
            "CONFIDENTIAL: This document contains proprietary information and is intended "
            "for internal use at OneAcre Fund and authorized partners only.",
            self.styles['NoteText']
        )
        
        self.elements.append(confidentiality)
        self.elements.append(PageBreak())

    def _add_toc_placeholder(self):
        """Add a table of contents placeholder to be filled later."""
        # This would be properly implemented with a custom ToC solution
        # For now, we're just adding a placeholder heading
        self.elements.append(Paragraph("Table of Contents", self.styles['TOCHeading']))
        self.elements.append(Spacer(1, 0.1*inch))
        
        # Add placeholder text - in a real implementation, this would be dynamically generated
        toc_entries = [
            "1. Executive Summary",
            "2. Data Inspection and Preparation",
            "3. Model Development",
            "4. Model Evaluation",
            "5. Threshold Analysis",
            "6. Margin Analysis",
            "7. Holdout Validation",
            "8. Appendix: Detailed Metrics"
        ]
        
        for entry in toc_entries:
            self.elements.append(Paragraph(entry, self.styles['TOCChapter']))
            
        self.elements.append(Spacer(1, 0.25*inch))
        self.elements.append(PageBreak())

    def add_chapter(self, title: str, description: Optional[str] = None):
        """
        Add a new chapter (major section) to the report.
        
        Args:
            title: Chapter title
            description: Optional chapter description
        """
        self.chapter_count += 1
        self.section_count = 0  # Reset section count for new chapter
        
        # Add chapter heading with number
        chapter_title = f"{self.chapter_count}. {title}"
        self.elements.append(Paragraph(chapter_title, self.styles['Chapter']))
        
        # Add horizontal rule
        self.elements.append(HRFlowable(width="100%", thickness=1, color=OAF_LIGHT_GREEN, 
                                         lineCap='square', spaceBefore=6, spaceAfter=6))
        
        # Add description if provided
        if description:
            self.elements.append(Paragraph(description, self.styles['BodyText']))
            
        self.elements.append(Spacer(1, 0.1*inch))

    def add_section(self, title: str, content: Optional[str] = None):
        """
        Add a new section to the report.
        
        Args:
            title: Section title
            content: Optional text content for the section
        """
        self.section_count += 1
        
        # Add section heading with hierarchical numbering
        section_title = f"{self.chapter_count}.{self.section_count} {title}"
        self.elements.append(Paragraph(section_title, self.styles['Section']))
        
        # Add content if provided
        if content:
            self.elements.append(Paragraph(content, self.styles['BodyText']))
            
        self.elements.append(Spacer(1, 0.1*inch))

    def add_subsection(self, title: str, content: Optional[str] = None):
        """
        Add a new subsection to the report.
        
        Args:
            title: Subsection title
            content: Optional text content for the subsection
        """
        # Add subsection heading
        self.elements.append(Paragraph(title, self.styles['Subsection']))
        
        # Add content if provided
        if content:
            self.elements.append(Paragraph(content, self.styles['BodyText']))
            
        self.elements.append(Spacer(1, 0.05*inch))

    def add_paragraph(self, text: str, style: str = 'BodyText'):
        """
        Add a paragraph of text.
        
        Args:
            text: Paragraph text
            style: Style to apply (default is 'BodyText')
        """
        self.elements.append(Paragraph(text, self.styles[style]))
        self.elements.append(Spacer(1, 0.05*inch))

    def add_code_block(self, code: str, language: str = "python"):
        """
        Add a code block to the report.
        
        Args:
            code: Code content as string
            language: Programming language (for documentation)
        """
        # Format the code text
        code_lines = code.split('\n')
        formatted_code = '<br/>'.join([line.replace(' ', '&nbsp;') for line in code_lines])
        
        # Add a label for the language
        self.elements.append(Paragraph(f"<i>{language}</i>:", self.styles['BodyText']))
        
        # Add the code block
        self.elements.append(Paragraph(formatted_code, self.styles['CodeText']))
        self.elements.append(Spacer(1, 0.1*inch))

    def add_note(self, text: str):
        """
        Add a note or comment.
        
        Args:
            text: Note text
        """
        note_text = f"<b>Note:</b> {text}"
        self.elements.append(Paragraph(note_text, self.styles['NoteText']))
        self.elements.append(Spacer(1, 0.05*inch))

    def add_table(
        self, 
        data: List[List[str]], 
        headers: List[str], 
        title: Optional[str] = None,
        column_widths: Optional[List[float]] = None,
        highlight_max: Optional[List[int]] = None,
        highlight_min: Optional[List[int]] = None,
        percent_cols: Optional[List[int]] = None,
        decimal_places: Optional[Dict[int, int]] = None
    ):
        """
        Add a table to the report.
        
        Args:
            data: Table data as list of lists
            headers: Column headers
            title: Optional table title
            column_widths: Optional list of column widths
            highlight_max: Optional list of column indices to highlight maximum values
            highlight_min: Optional list of column indices to highlight minimum values
            percent_cols: Optional list of column indices to format as percentages
            decimal_places: Optional dict mapping column indices to decimal places for formatting
        """
        if title:
            self.elements.append(Paragraph(title, self.styles['Subsection']))
        
        # Format data based on percent_cols and decimal_places
        formatted_data = []
        for row in data:
            formatted_row = []
            for i, cell in enumerate(row):
                if percent_cols and i in percent_cols and isinstance(cell, (int, float)):
                    # Format as percentage
                    formatted_cell = f"{cell:.1%}"
                elif decimal_places and i in decimal_places and isinstance(cell, (int, float)):
                    # Format with specific decimal places
                    places = decimal_places[i]
                    formatted_cell = f"{cell:.{places}f}"
                elif isinstance(cell, float):
                    # Default float formatting
                    formatted_cell = f"{cell:.4f}"
                else:
                    formatted_cell = str(cell)
                formatted_row.append(formatted_cell)
            formatted_data.append(formatted_row)
        
        # Build table with headers
        table_data = [headers] + formatted_data
        
        # Set column widths if provided, otherwise auto-calculate
        if column_widths:
            table = Table(table_data, colWidths=column_widths)
        else:
            # Auto-calculate column widths based on content
            # For simplicity here, use a fixed width, but this could be improved
            table_width = self.doc.width - inch  # Leave some margin
            col_width = table_width / len(headers)
            table = Table(table_data, colWidths=[col_width] * len(headers))
        
        # Standard style for all tables
        style = [
            # Header row styling
            ('BACKGROUND', (0, 0), (-1, 0), OAF_BLUE),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            
            # Cell styling
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('ALIGN', (0, 1), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
            
            # Grid styling
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('LINEBELOW', (0, 0), (-1, 0), 1, OAF_DARK_GREEN),
            
            # Alternating row colors
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.whitesmoke])
        ]
        
        # Add highlight styles for max/min values if requested
        if highlight_max:
            for col_idx in highlight_max:
                if col_idx < len(headers):
                    # Find the maximum value in this column (skip header row)
                    try:
                        col_values = [float(row[col_idx]) for row in formatted_data]
                        if col_values:
                            max_value = max(col_values)
                            max_row = col_values.index(max_value) + 1  # +1 for header row
                            max_cell_style = ('BACKGROUND', (col_idx, max_row), (col_idx, max_row), 
                                              OAF_LIGHT_GREEN)
                            style.append(max_cell_style)
                    except (ValueError, TypeError):
                        # Skip if column contains non-numeric values
                        pass
        
        if highlight_min:
            for col_idx in highlight_min:
                if col_idx < len(headers):
                    # Find the minimum value in this column (skip header row)
                    try:
                        col_values = [float(row[col_idx]) for row in formatted_data]
                        if col_values:
                            min_value = min(col_values)
                            min_row = col_values.index(min_value) + 1  # +1 for header row
                            min_cell_style = ('BACKGROUND', (col_idx, min_row), (col_idx, min_row), 
                                              OAF_LIGHT_BLUE)
                            style.append(min_cell_style)
                    except (ValueError, TypeError):
                        # Skip if column contains non-numeric values
                        pass
        
        # Apply all styles to the table
        table.setStyle(TableStyle(style))
        
        # Add the table to the report
        self.elements.append(table)
        self.elements.append(Spacer(1, 0.2*inch))

    def _get_temp_path(self, prefix: str = "temp_plot_", suffix: str = ".png") -> str:
        """
        Get a unique temporary file path.
        
        Args:
            prefix: Prefix for the temporary file
            suffix: Suffix (extension) for the temporary file
            
        Returns:
            Path to the temporary file
        """
        # Create a unique filename using timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{prefix}{timestamp}{suffix}"
        
        # Use the same directory as the output PDF
        return os.path.join(os.path.dirname(self.output_path), filename)

    def add_plot(
        self, 
        figure: Optional[plt.Figure] = None, 
        image_path: Optional[str] = None,
        caption: Optional[str] = None,
        width: float = 6*inch,
        height: Optional[float] = None,
        dpi: int = 300
    ):
        """
        Add a matplotlib figure or image to the report.
        
        Args:
            figure: matplotlib Figure object (if providing directly)
            image_path: Path to image file (alternative to figure)
            caption: Optional caption for the image
            width: Width of the image in the report
            height: Optional height (calculated from aspect ratio if None)
            dpi: Resolution for matplotlib figures
        """
        if figure is not None:
            # Save figure to temporary file
            temp_path = self._get_temp_path()
            figure.savefig(temp_path, dpi=dpi, bbox_inches='tight')
            self.temp_files.append(temp_path)
            path_to_use = temp_path
        elif image_path is not None and os.path.exists(image_path):
            path_to_use = image_path
        else:
            print("Warning: No figure or valid image path provided to add_plot()")
            return
        
        # Calculate height if not provided (maintain aspect ratio)
        if height is None:
            # Get aspect ratio from image
            with PILImage.open(path_to_use) as img:
                aspect = img.height / img.width
                height = width * aspect
        
        # Add image to report
        img = Image(path_to_use, width=width, height=height)
        self.elements.append(img)
        
        # Add caption if provided
        if caption:
            self.elements.append(Paragraph(caption, self.styles['CaptionText']))
        
        self.elements.append(Spacer(1, 0.2*inch))

    def add_plots_grid(
        self,
        figures: Optional[List[plt.Figure]] = None,
        image_paths: Optional[List[str]] = None,
        captions: Optional[List[str]] = None,
        title: Optional[str] = None,
        rows: int = 2,
        cols: int = 2,
        figsize: Tuple[int, int] = (7, 5),
        dpi: int = 200
    ):
        """
        Add multiple plots in a grid layout.
        
        Args:
            figures: List of matplotlib figures to include (alternative to image_paths)
            image_paths: List of paths to image files (alternative to figures)
            captions: Optional list of captions for each plot
            title: Optional overall title for the grid
            rows: Number of rows in the grid
            cols: Number of columns in the grid
            figsize: Size for the combined figure in inches
            dpi: Resolution for the output image
        """
        if (figures is None and image_paths is None) or (figures is not None and image_paths is not None):
            print("Warning: Provide either figures or image_paths to add_plots_grid()")
            return
        
        # If we have a section title, add it
        if title:
            self.add_subsection(title)
        
        # Method 1: Combine figures into a single grid figure (better quality)
        if figures:
            # Create a new figure to hold all subplots
            fig, axes = plt.subplots(rows, cols, figsize=figsize, 
                                     squeeze=False, constrained_layout=True)
            
            # Flatten the axes for easier indexing
            axes_flat = axes.flatten()
            
            # Add each figure to a subplot
            for i, (ax, figure) in enumerate(zip(axes_flat, figures)):
                if i < len(figures):
                    # Extract the axis content from the original figure
                    # This is a simplified approach - more complex figures might need
                    # more sophisticated handling
                    for ax_orig in figure.get_axes():
                        # Copy content by saving and re-importing the figure
                        temp_path = self._get_temp_path(prefix=f"temp_subfig_{i}_")
                        figure.savefig(temp_path, dpi=dpi)
                        self.temp_files.append(temp_path)
                        
                        # Import as image and place on axis
                        with PILImage.open(temp_path) as img:
                            # Remove axis elements for the container axes
                            ax.axis('off')
                            # Display figure as image
                            ax.imshow(np.asarray(img), aspect='auto')
                            
                            # Add caption as axis title if available
                            if captions and i < len(captions):
                                ax.set_title(captions[i], fontsize=10)
                else:
                    # Hide unused axes
                    ax.axis('off')
            
            # Add the combined figure to the report
            self.add_plot(figure=fig, dpi=dpi)
            plt.close(fig)
            
        # Method 2: Add individual image files sequentially
        elif image_paths:
            # Calculate grid dimensions based on number of images
            n_images = len(image_paths)
            
            # Calculate image width based on number of columns
            width = (self.doc.width - 0.5*inch) / cols
            
            # Add each image
            for i, img_path in enumerate(image_paths):
                if i < n_images:
                    # Add a new row after each complete row of images
                    if i > 0 and i % cols == 0:
                        self.elements.append(Spacer(1, 0.1*inch))
                    
                    # Add the image
                    if os.path.exists(img_path):
                        img = Image(img_path, width=width)
                        self.elements.append(img)
                        
                        # Add caption if available
                        if captions and i < len(captions):
                            self.elements.append(Paragraph(captions[i], self.styles['CaptionText']))
            
            # Add space after the grid
            self.elements.append(Spacer(1, 0.2*inch))

    def add_page_break(self):
        """Add a page break to the report."""
        self.elements.append(PageBreak())

    def add_horizontal_line(self, width="100%", thickness=1, color=OAF_LIGHT_GREEN):
        """
        Add a horizontal line to the report.
        
        Args:
            width: Width of the line (as percentage or absolute)
            thickness: Line thickness in points
            color: Line color
        """
        self.elements.append(HRFlowable(width=width, thickness=thickness, 
                                         color=color, spaceBefore=10, spaceAfter=10))

    def add_bullet_list(self, items: List[str], style: str = 'BodyText'):
        """
        Add a bullet point list to the report.
        
        Args:
            items: List of text items
            style: Style to apply to list items
        """
        bullet_list = []
        for item in items:
            bullet_item = ListItem(
                Paragraph(item, self.styles[style]),
                leftIndent=20,
                value='bullet'
            )
            bullet_list.append(bullet_item)
            
        self.elements.append(ListFlowable(
            bullet_list,
            bulletType='bullet',
            start='bullet',
            leftIndent=10,
            bulletFontSize=8,
            bulletOffsetY=2
        ))
        self.elements.append(Spacer(1, 0.1*inch))

    def add_numbered_list(self, items: List[str], style: str = 'BodyText'):
        """
        Add a numbered list to the report.
        
        Args:
            items: List of text items
            style: Style to apply to list items
        """
        numbered_list = []
        for i, item in enumerate(items, 1):
            numbered_item = ListItem(
                Paragraph(item, self.styles[style]),
                leftIndent=20,
                value=f"{i}."
            )
            numbered_list.append(numbered_item)
            
        self.elements.append(ListFlowable(
            numbered_list,
            bulletType='bullet',
            start='1',
            leftIndent=10,
            bulletOffsetY=2
        ))
        self.elements.append(Spacer(1, 0.1*inch))

    def add_json_data(self, data: Dict, title: Optional[str] = None):
        """
        Add JSON data in a formatted way.
        
        Args:
            data: Dictionary containing the data
            title: Optional title for the data section
        """
        if title:
            self.elements.append(Paragraph(title, self.styles['Subsection']))
        
        # Format the JSON string with indentation for readability
        json_str = json.dumps(data, indent=2)
        
        # Format for display with line breaks
        formatted_json = json_str.replace('\n', '<br/>').replace(' ', '&nbsp;')
        
        # Add as a code block
        self.elements.append(Paragraph(formatted_json, self.styles['CodeText']))
        self.elements.append(Spacer(1, 0.1*inch))

    def cleanup(self):
        """Clean up temporary files."""
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception as e:
                print(f"Warning: Failed to remove temporary file {temp_file}: {str(e)}")
        self.temp_files = []

    def save(self):
        """Save the report to PDF file."""
        try:
            self.doc.build(self.elements)
            print(f"Report saved to {self.output_path}")
            return self.output_path
        except Exception as e:
            print(f"Error saving report: {str(e)}")
            return None
        finally:
            # Clean up temporary files even if an error occurs
            self.cleanup()
