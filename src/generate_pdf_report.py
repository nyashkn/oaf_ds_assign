#!/usr/bin/env python
"""
Generate a PDF report from a markdown file using markdown-pdf.

This script uses the markdown-pdf package to convert markdown to PDF
with proper styling and image handling.

Example usage:
    python src/generate_pdf_report.py --markdown docs/regression_modeling_summary.md 
                                     --output docs/loan_repayment_regression_report.pdf
"""

import os
import sys
import argparse
import subprocess
import tempfile
import shutil
from pathlib import Path

def fix_image_paths(markdown_content, markdown_dir):
    """
    Fix image paths to use absolute paths so they're found when converting to PDF.
    
    Args:
        markdown_content: Content of the markdown file
        markdown_dir: Directory containing the markdown file
        
    Returns:
        Updated markdown content with fixed image paths
    """
    lines = markdown_content.split('\n')
    fixed_lines = []
    
    for line in lines:
        # Check if the line contains an image
        if '![' in line and '](' in line and ')' in line:
            start_idx = line.find('](') + 2
            end_idx = line.find(')', start_idx)
            
            if start_idx < end_idx:
                img_path = line[start_idx:end_idx]
                
                # Check if the path is relative
                if img_path.startswith('..') or not (img_path.startswith('/') or img_path.startswith('http')):
                    # Construct absolute path
                    abs_img_path = os.path.abspath(os.path.join(markdown_dir, img_path))
                    
                    # Replace the path
                    line = line[:start_idx] + abs_img_path + line[end_idx:]
        
        fixed_lines.append(line)
    
    return '\n'.join(fixed_lines)

def generate_pdf(markdown_path, output_path, style=None):
    """
    Generate a PDF from a markdown file.
    
    Args:
        markdown_path: Path to markdown file
        output_path: Path for output PDF
        style: Optional CSS style file
        
    Returns:
        Path to generated PDF
    """
    print(f"Generating PDF from {markdown_path} to {output_path}")
    
    # Create a temporary directory for the conversion
    with tempfile.TemporaryDirectory() as temp_dir:
        # Copy the markdown file to the temp directory
        temp_md_path = os.path.join(temp_dir, os.path.basename(markdown_path))
        
        # Get the content and fix image paths
        markdown_dir = os.path.dirname(os.path.abspath(markdown_path))
        with open(markdown_path, 'r') as f:
            md_content = f.read()
        
        fixed_content = fix_image_paths(md_content, markdown_dir)
        
        # Write the fixed content to the temp file
        with open(temp_md_path, 'w') as f:
            f.write(fixed_content)
        
        # Create a CSS file if not provided
        if not style:
            css_content = """
            body {
                font-family: Arial, sans-serif;
                font-size: 12px;
                line-height: 1.6;
                margin: 30px;
            }
            h1 {
                font-size: 24px;
                border-bottom: 1px solid #ddd;
                padding-bottom: 10px;
                margin-top: 30px;
            }
            h2 {
                font-size: 20px;
                border-bottom: 1px solid #eee;
                padding-bottom: 10px;
                margin-top: 25px;
            }
            h3 {
                font-size: 16px;
                margin-top: 20px;
            }
            table {
                border-collapse: collapse;
                width: 100%;
                margin: 20px 0;
            }
            th, td {
                text-align: left;
                padding: 8px;
                border: 1px solid #ddd;
            }
            th {
                background-color: #f2f2f2;
                font-weight: bold;
            }
            tr:nth-child(even) {
                background-color: #f9f9f9;
            }
            img {
                max-width: 90%;
                margin: 20px auto;
                display: block;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            }
            blockquote {
                border-left: 5px solid #eee;
                padding-left: 15px;
                color: #666;
                margin: 20px 0;
            }
            code {
                background-color: #f8f8f8;
                padding: 2px 4px;
                border-radius: 3px;
                font-family: monospace;
            }
            pre {
                background-color: #f8f8f8;
                padding: 10px;
                border-radius: 3px;
                overflow-x: auto;
            }
            """
            
            css_path = os.path.join(temp_dir, "style.css")
            with open(css_path, 'w') as f:
                f.write(css_content)
            
            style = css_path
        
        # Try using different utilities based on availability
        try:
            # Try markdown-pdf if available
            cmd = ["npx", "markdown-pdf", "-s", style, "-o", output_path, temp_md_path]
            print(f"Running conversion command: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
            
            if os.path.exists(output_path):
                print(f"PDF successfully generated: {output_path}")
                return output_path
        except (subprocess.SubprocessError, FileNotFoundError) as e:
            print(f"Error using markdown-pdf: {e}")
        
        try:
            # Try pandoc as a fallback
            cmd = ["pandoc", temp_md_path, "-o", output_path, "--pdf-engine=xelatex", "--css", style]
            print(f"Trying conversion with pandoc: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
            
            if os.path.exists(output_path):
                print(f"PDF successfully generated with pandoc: {output_path}")
                return output_path
        except (subprocess.SubprocessError, FileNotFoundError) as e:
            print(f"Error using pandoc: {e}")
            
        print("Could not generate PDF. Please ensure markdown-pdf or pandoc is installed.")
        print("Install markdown-pdf with: npm install -g markdown-pdf")
        print("Or install pandoc from: https://pandoc.org/installing.html")
        
        return None

def main(markdown_path, output_path, style=None):
    """Main function to generate the PDF report."""
    if not os.path.exists(markdown_path):
        print(f"Error: Markdown file not found: {markdown_path}")
        return False
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Generate the PDF
    result_path = generate_pdf(markdown_path, output_path, style)
    
    if result_path and os.path.exists(result_path):
        print(f"PDF report generated: {result_path}")
        return True
    else:
        print("Failed to generate PDF report")
        return False

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate PDF from markdown report')
    parser.add_argument('--markdown', type=str, required=True,
                       help='Path to markdown file')
    parser.add_argument('--output', type=str, required=True,
                       help='Path for output PDF file')
    parser.add_argument('--style', type=str,
                       help='Path to CSS style file (optional)')
    
    args = parser.parse_args()
    
    # Generate the PDF
    main(args.markdown, args.output, args.style)
