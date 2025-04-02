import os
import datetime
import io
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from mcp import StdioServerParameters
from mcpadapt.core import MCPAdapt
from mcpadapt.smolagents_adapter import SmolAgentsAdapter
# from phoenix.otel import register
from smolagents import CodeAgent, LiteLLMModel, tool
import pymupdf as fitz
import numpy as np
from PIL import Image

import litellm
litellm._turn_on_debug()
# Configure the Phoenix tracer
# tracer_provider = register(
#     project_name="tupande_insights",
#     auto_instrument=True,
# )

# Load environment variables
load_dotenv()

# Define tools for the agent
@tool
def get_pdf_as_images(pdf_path: str) -> List[np.ndarray]:
    """
    Convert a PDF file to a list of images (one per page).
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        A list of numpy arrays representing images of each page
    """
    try:
        # Open the PDF file
        doc = fitz.open(pdf_path)
        images = []
        
        # Iterate through each page
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            
            # Render page to an image
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better resolution
            
            # Convert to PIL Image
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            # Convert to numpy array
            img_array = np.array(img)
            images.append(img_array)
        
        return images
    except Exception as e:
        return f"Error processing PDF: {str(e)}"

@tool
def read_markdown_file(file_path: str) -> str:
    """
    Read a markdown file and return its contents.
    
    Args:
        file_path: Path to the markdown file
        
    Returns:
        The contents of the markdown file as a string
    """
    try:
        with open(file_path, 'r') as file:
            return file.read()
    except Exception as e:
        return f"Error reading file: {str(e)}"

@tool
def save_markdown_file(content: str, file_path: str) -> str:
    """
    Save content to a markdown file.
    
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

def main():
    # Initialize the model
    model = LiteLLMModel(
        model_id="anthropic.claude-3-5-sonnet-20241022-v2:0"
        # Uncomment to use Claude 3.7 Sonnet
        # model_id="anthropic.claude-3-7-sonnet-20250219-v1:0"
    )

    # Define filesystem MCP server parameters
    project_path = Path(__file__).resolve().parents[2]  # Go up three levels from this file
    filesystem_server = StdioServerParameters(
        command="npx",
        args=[
            "-y",
            "@modelcontextprotocol/server-filesystem",
            str(project_path),
        ],
    )

    # Generate timestamp for the output file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"docs/final/02_agent_insights_{timestamp}.md"

    # First, get the PDF images before running the agent
    pdf_path = 'data/processed/scorecard_modelling/v22/modeling_report.pdf'
    pdf_images = get_pdf_as_images(pdf_path)
    
    # Convert numpy arrays to PIL Images if needed
    pil_images = []
    for img_array in pdf_images:
        if isinstance(img_array, np.ndarray):
            pil_images.append(Image.fromarray(img_array))
        else:
            # Handle the error case if get_pdf_as_images returned an error string
            print(f"Error converting PDF to images: {img_array}")
            pil_images = []
            break

    # Keep context active while using the agent
    with MCPAdapt(filesystem_server, SmolAgentsAdapter()) as fs_tools:
        agent = CodeAgent(
            tools=[*fs_tools, get_pdf_as_images, read_markdown_file, save_markdown_file],
            model=model,
            verbosity_level=2,
            planning_interval=3,
            add_base_tools=True,
            provide_run_summary=True,
            additional_authorized_imports=[
                "pandas",
                "numpy",
                "io",
                "matplotlib",
                "plotly",
                "tabulate",
                "json",
                "PIL",
                "fitz",  # PyMuPDF
                "datetime",
                "markdown",
                "re",
                "base64",
            ],
            max_steps=30,  # PDF analysis might require more steps
        )
        
        # Define the task
        task = f"""
        # Tupande Scorecard Model Analysis Task
        
        ## Objective
        Analyze the scorecard modeling report for One Acre Fund's Tupande program to extract valuable business insights that can guide strategic decisions. Focus on identifying patterns, risk factors, and opportunities to improve loan repayment rates.
        
        ## Data Sources
        1. Primary Data: Scorecard modeling report PDF at 'data/processed/scorecard_modelling/v22/modeling_report.pdf'
        2. Feature explanations: 'docs/03_features_explanation.md'
        3. Contextual information:
           - 'docs/external/02_oaf_background_research.md'
           - 'docs/external/02_oaf_impact_studies.md'
           - 'docs/external/02_oaf_product_and_credit.md'
           - 'docs/external/02_tupande_gross_margin_estimate.md'
        
        ## Key Analysis Areas
        1. Variable Importance Analysis:
           - Identify the most significant predictors of loan repayment
           - Analyze how geographic factors (region, area, sales territory) impact repayment
           - Examine the relationship between historical behavior and current repayment patterns
        
        2. Risk Segmentation:
           - Identify key segments with higher/lower risk profiles
           - Analyze WOE (Weight of Evidence) patterns across different variables
           - Determine critical threshold values that significantly change risk levels
        
        3. Model Performance Assessment:
           - Evaluate model metrics (KS, AUC, Gini) for both training and testing datasets
           - Analyze how well the model separates good vs. bad loans
           - Identify potential areas for model improvement
        
        4. Business Application:
           - Suggest concrete business actions based on the findings
           - Recommend strategies to reduce default rates
           - Identify customer segments to focus on or approaches for
           
        ## Technical Approach
        1. First, read any supporting documents to understand context:
           - Use read_markdown_file() to access background information
        
        2. Analyze the data from the provided PDF images:
           - Focus on key metrics, tables, and charts
           - Identify patterns in the WOE binning plots
           - Examine variable importance rankings
        
        3. Generate actionable insights connecting model findings to business context:
           - For each insight, explain: what is the pattern, why it matters, and what action to take
           - Suggest which visualizations should be included to support each insight
           - Clearly indicate which findings should be highlighted in the final report
        
        4. Output your findings in a structured markdown document:
           - Use save_markdown_file() to save to '{output_path}'
           - Format should include clear sections, bulleted lists, and recommendations
           - Flag key visualizations that should be created with notes like [deposit vs repayment heatmap]
        
        ## Output Format
        The final markdown document should include:
        1. Executive summary with key takeaways
        2. Detailed insights organized by business impact
        3. Specific recommendations for:
           - Lending criteria adjustments
           - Customer segment targeting
           - Risk mitigation strategies
        4. Visualization recommendations with descriptions
        5. Appendix with technical details about important variables
        
        Be thorough and insightful in your analysis. The goal is to provide actionable intelligence that can help improve Tupande's lending operations and farmers' success.
        """

        result = agent.run(
            task=task,
            images=pil_images,  # Pass the PDF images directly to the agent
            reset=False,
        )
        return result

if __name__ == "__main__":
    result = main()
    print(result)