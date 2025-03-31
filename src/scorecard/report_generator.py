"""
PDF report generation module for scorecard modeling.
"""

import os
import tempfile
from datetime import datetime
from typing import List, Dict, Optional, Union, Tuple
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.platypus import PageBreak
import matplotlib.pyplot as plt
from PIL import Image as PILImage
import numpy as np

class ScorecardReport:
    """
    Class to generate PDF reports for scorecard modeling process.
    """
    def __init__(self, output_path: str, title: str = "Scorecard Model Report"):
        """
        Initialize the report generator.
        
        Args:
            output_path: Path where the PDF report will be saved
            title: Title of the report
        """
        self.output_path = output_path
        self.title = title
        self.elements = []
        self.temp_files = []  # Track temporary files for cleanup
        self.styles = getSampleStyleSheet()
        
        # Create custom styles
        self.styles.add(ParagraphStyle(
            name='SectionTitle',
            parent=self.styles['Heading1'],
            fontSize=16,
            spaceAfter=20
        ))
        self.styles.add(ParagraphStyle(
            name='SubSection',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceAfter=15
        ))
        
        # Initialize document
        self.doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Add title page
        self._add_title_page()

    def _add_title_page(self):
        """Add the title page to the report."""
        # Add significant vertical space to center content
        self.elements.append(Spacer(1, 150))
        
        # Create title with larger font 
        title_style = ParagraphStyle(
            name='ReportTitle',
            parent=self.styles['Title'],
            fontSize=24,
            alignment=1,  # Center alignment
            spaceAfter=20
        )
        title = Paragraph(self.title, title_style)
        
        # Add subtitle
        subtitle = Paragraph(
            "OneAcre Fund Credit Risk Analysis",
            self.styles['Heading2']
        )
        
        # Format date with more details
        date_text = Paragraph(
            f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            self.styles['Normal']
        )
        
        # Add description
        description = Paragraph(
            "This report contains the results of credit scorecard modeling for loan repayment analysis.",
            self.styles['Normal']
        )
        
        self.elements.extend([
            title,
            Spacer(1, 10),
            subtitle,
            Spacer(1, 30),
            date_text,
            Spacer(1, 10),
            description,
            PageBreak()
        ])

    def add_section(self, title: str, content: str):
        """
        Add a new section to the report.
        
        Args:
            title: Section title
            content: Text content for the section
        """
        self.elements.extend([
            Paragraph(title, self.styles['SectionTitle']),
            Paragraph(content, self.styles['Normal']),
            Spacer(1, 12)
        ])

    def add_subsection(self, title: str, content: str):
        """
        Add a new subsection to the report.
        
        Args:
            title: Subsection title
            content: Text content for the subsection
        """
        self.elements.extend([
            Paragraph(title, self.styles['SubSection']),
            Paragraph(content, self.styles['Normal']),
            Spacer(1, 8)
        ])

    def add_table(self, data: List[List[str]], headers: List[str], title: Optional[str] = None):
        """
        Add a table to the report.
        
        Args:
            data: Table data as list of lists
            headers: Column headers
            title: Optional table title
        """
        if title:
            self.elements.append(Paragraph(title, self.styles['SubSection']))
        
        table_data = [headers] + data
        table = Table(table_data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        
        self.elements.extend([
            table,
            Spacer(1, 12)
        ])

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

    def add_plot(self, figure: plt.Figure, caption: Optional[str] = None):
        """
        Add a matplotlib figure to the report.
        
        Args:
            figure: matplotlib Figure object
            caption: Optional caption for the plot
        """
        # Save figure to temporary file
        temp_path = self._get_temp_path()
        figure.savefig(temp_path, dpi=300, bbox_inches='tight')
        self.temp_files.append(temp_path)
        
        # Add to report
        img = Image(temp_path, width=6*inch, height=4*inch)
        self.elements.append(img)
        
        if caption:
            self.elements.append(Paragraph(caption, self.styles['Italic']))
        
        self.elements.append(Spacer(1, 12))

    def add_woe_plots_grid(self, plot_paths: List[str], captions: List[str], 
                          rows: int = 3, cols: int = 2):
        """
        Add multiple WOE plots in a grid layout.
        
        Args:
            plot_paths: List of file paths to WOE plot images
            captions: List of captions for each plot
            rows: Number of rows in the grid
            cols: Number of columns in the grid
        """
        # Check if we have valid plot paths
        if not plot_paths:
            print("Warning: No WOE plot paths provided to add_woe_plots_grid")
            return
            
        plots_per_page = rows * cols
        for i in range(0, len(plot_paths), plots_per_page):
            page_plots = plot_paths[i:i + plots_per_page]
            page_captions = captions[i:i + plots_per_page]
            
            # For each plot on the page
            for plot_path, caption in zip(page_plots, page_captions):
                # Check if the file exists
                if not os.path.exists(plot_path):
                    print(f"Warning: WOE plot file not found: {plot_path}")
                    continue
                    
                # Add directly to document
                img = Image(plot_path, width=5*inch, height=3*inch)
                self.elements.append(img)
                
                # Add caption
                if caption:
                    caption_paragraph = Paragraph(caption, self.styles['Italic'])
                    self.elements.append(caption_paragraph)
                
                self.elements.append(Spacer(1, 15))
            
            # Add page break between plot grids
            if i + plots_per_page < len(plot_paths):
                self.elements.append(PageBreak())
                
    def add_image_file(self, image_path: str, width: float = 6*inch, 
                       height: float = 4*inch, caption: Optional[str] = None):
        """
        Add an image file directly to the report.
        
        Args:
            image_path: Path to the image file
            width: Width of the image in the report
            height: Height of the image in the report
            caption: Optional caption for the image
        """
        if not os.path.exists(image_path):
            print(f"Warning: Image file not found: {image_path}")
            return
            
        img = Image(image_path, width=width, height=height)
        self.elements.append(img)
        
        if caption:
            self.elements.append(Paragraph(caption, self.styles['Italic']))
        
        self.elements.append(Spacer(1, 12))

    def add_model_summary(self, model_info: Dict):
        """
        Add model summary information to the report.
        
        Args:
            model_info: Dictionary containing model information
        """
        self.add_section("Model Summary", "")
        
        # Model configuration
        if 'config' in model_info:
            self.add_subsection("Model Configuration", "")
            config_data = [[k, str(v)] for k, v in model_info['config'].items()]
            self.add_table(config_data, ['Parameter', 'Value'])
        
        # Performance metrics
        if 'metrics' in model_info:
            self.add_subsection("Performance Metrics", "")
            metrics_data = [[k, f"{v:.4f}"] for k, v in model_info['metrics'].items()]
            self.add_table(metrics_data, ['Metric', 'Value'])
        
        # Feature importance
        if 'feature_importance' in model_info:
            self.add_subsection("Feature Importance", "")
            importance_data = [[k, f"{v:.4f}"] for k, v in model_info['feature_importance'].items()]
            self.add_table(importance_data, ['Feature', 'Importance'])

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
        finally:
            # Clean up temporary files even if an error occurs
            self.cleanup()
