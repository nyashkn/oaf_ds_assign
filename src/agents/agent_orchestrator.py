"""
Main orchestration script for running the SmolAgents for insights generation
"""
import os
import datetime
from pathlib import Path
from typing import Dict, Any

from dotenv import load_dotenv
from smolagents import CodeAgent, LiteLLMModel

# Import our custom modules
from agent_tools import (
    open_pdf, 
    get_next_pdf_batch, 
    read_file, 
    write_file, 
    list_directory,
    add_pdf_images_callback
)
from agent_prompts import get_pdf_analysis_prompt, get_feature_engineering_prompt
from human_intervention import human_intervention

# Load environment variables
load_dotenv()

# Optional debug mode
import litellm
# Uncomment to enable debug mode
# litellm._turn_on_debug()

def create_agent(model_id: str, task_type: str) -> CodeAgent:
    """
    Create and configure a CodeAgent with appropriate tools and settings
    
    Args:
        model_id: The LLM model ID to use
        task_type: Type of task the agent will perform, affects imports allowed
        
    Returns:
        Configured CodeAgent instance
    """
    # Initialize the model
    model = LiteLLMModel(model_id=model_id)
    
    # Determine which imports to allow based on task
    allowed_imports = [
        "os", "datetime", "pathlib",  # Basic utilities
        "json", "re", "base64",       # Data handling
    ]
    
    if task_type == "pdf_analysis":
        allowed_imports.extend([
            "pandas", "numpy", "matplotlib", "plotly", 
            "io", "PIL", "fitz", "markdown", "tabulate"
        ])
    elif task_type == "feature_engineering":
        allowed_imports.extend([
            "pandas", "numpy", "sklearn", "scipy", 
            "matplotlib", "category_encoders", "feature_engine"
        ])
    
    # Create the agent with appropriate tools and callbacks
    agent = CodeAgent(
        tools=[
            # File and PDF tools
            open_pdf, get_next_pdf_batch, read_file, write_file, list_directory,
            # Human-in-the-loop tool
            human_intervention
        ],
        model=model,
        verbosity_level=2,  # Detailed output
        planning_interval=3,
        add_base_tools=True,
        provide_run_summary=True,
        additional_authorized_imports=allowed_imports,
        max_steps=40,  # Increased to allow for batch processing
        step_callbacks=[add_pdf_images_callback],  # Add our custom callback
    )
    
    return agent

def run_pdf_analysis_task(
    model_id: str = "anthropic.claude-3-5-sonnet-20241022-v2:0",
    pdf_path: str = "data/processed/scorecard_modelling/v22/modeling_report.pdf",
    output_dir: str = "docs/final",
    debug: bool = False,
) -> str:
    """
    Run the PDF analysis task to extract insights
    
    Args:
        model_id: The LLM model ID to use
        pdf_path: Path to the PDF file to analyze
        output_dir: Directory where output should be saved
        debug: Whether to enable debug mode
        
    Returns:
        Path to the output file
    """
    # Enable debug mode if requested
    if debug:
        import litellm
        litellm._turn_on_debug()
        print(f"Debug mode enabled for LiteLLM")
    
    # Generate timestamp for the output file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"{output_dir}/02_agent_insights_{timestamp}.md"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Log the starting parameters
    print(f"Starting PDF analysis task:")
    print(f"- Model: {model_id}")
    print(f"- PDF Path: {pdf_path}")
    print(f"- Output Path: {output_path}")
    
    # Get the task prompt
    task = get_pdf_analysis_prompt(output_path=output_path)
    
    # Create and run the agent
    agent = create_agent(model_id, task_type="pdf_analysis")
    
    print(f"Agent created with {len(agent.tools)} tools")
    print(f"Running agent with max_steps={agent.max_steps}")
    
    result = agent.run(task=task, reset=False)
    
    print(f"Agent task completed")
    
    return result

def run_feature_engineering_task(
    model_id: str = "anthropic.claude-3-5-sonnet-20241022-v2:0",
    dataset_path: str = "data/raw/customer_data.csv",
    output_dir: str = "src/features",
) -> str:
    """
    Run the feature engineering task to create feature transformation code
    
    Args:
        model_id: The LLM model ID to use
        dataset_path: Path to the dataset
        output_dir: Directory where output should be saved
        
    Returns:
        Path to the output file
    """
    # Generate output file path
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"{output_dir}/build_features_{timestamp}.py"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get the task prompt
    task = get_feature_engineering_prompt(output_path=output_path, dataset_path=dataset_path)
    
    # Create and run the agent
    agent = create_agent(model_id, task_type="feature_engineering")
    result = agent.run(task=task, reset=False)
    
    return result

def main(task_type: str = "pdf_analysis", debug: bool = False) -> str:
    """
    Main function to run the appropriate agent task
    
    Args:
        task_type: Type of task to run (pdf_analysis or feature_engineering)
        debug: Whether to enable debug mode
        
    Returns:
        Result from the agent run
    """
    if task_type == "pdf_analysis":
        return run_pdf_analysis_task(debug=debug)
    elif task_type == "feature_engineering":
        return run_feature_engineering_task()
    else:
        raise ValueError(f"Unknown task type: {task_type}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run SmolAgent tasks")
    parser.add_argument(
        "--task", 
        type=str, 
        default="pdf_analysis",
        choices=["pdf_analysis", "feature_engineering"],
        help="Type of task to run"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode for LiteLLM"
    )
    parser.add_argument(
        "--pdf",
        type=str,
        default="data/processed/scorecard_modelling/v22/modeling_report.pdf",
        help="Path to the PDF file (for pdf_analysis task)"
    )
    
    args = parser.parse_args()
    
    result = main(task_type=args.task, debug=args.debug)
    print(result)