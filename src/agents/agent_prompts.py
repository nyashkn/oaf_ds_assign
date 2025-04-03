"""
Prompts for SmolAgents to perform different tasks
"""
from typing import Dict, Any

def get_pdf_analysis_prompt(output_path: str, context: Dict[str, Any] = None) -> str:
    """
    Generate a prompt for PDF analysis task
    
    Args:
        output_path: Path where the output should be saved
        context: Additional context parameters
        
    Returns:
        Formatted prompt string
    """
    return f"""
    # Tupande Scorecard Model Analysis Task
    
    ## Objective
    Analyze the scorecard modeling report for One Acre Fund's Tupande program to extract valuable business insights that can guide strategic decisions. Focus on identifying patterns, risk factors, and opportunities to improve loan repayment rates.
    
    ## Data Sources
    1. Primary Data: Scorecard modeling report PDF that will be provided in batches
    2. Feature explanations: 'docs/03_features_explanation.md'
    3. Contextual information:
       - 'docs/external/02_oaf_background_research.md'
       - 'docs/external/02_oaf_impact_studies.md'
       - 'docs/external/02_oaf_product_and_credit.md'
       - 'docs/external/02_tupande_gross_margin_estimate.md'
    
    ## Approach
    1. First, read the PDF in batches using the provided tools:
       - Use `open_pdf('data/processed/scorecard_modelling/v22/modeling_report.pdf')` to open the PDF
       - Use `get_next_pdf_batch()` to get batches of pages as you process the document
       - The PDF images will be automatically added to your observations
    
    2. Read supporting documents to understand context:
       - Use `read_file()` to access background information
    
    3. Analyze PDF content batch by batch:
       - Take notes on important information from each batch
       - Focus on key metrics, tables, and charts
       - Pay special attention to variable importance rankings and WOE binning plots
       
    4. Use the human_intervention tool when you need help:
       - For clarification: `human_intervention("clarification", "Your question here")`
       - For approval: `human_intervention("approval", "Confirm this action?")`
       - For choices: `human_intervention("multiple_choice", "Which option?", ["Option 1", "Option 2"])`
    
    5. After reviewing all batches, synthesize your findings:
       - Identify the most significant predictors of loan repayment
       - Analyze geographic factors impact
       - Examine historical behavior patterns
       - Identify risk segments
       - Assess model performance
    
    6. Generate actionable insights connecting model findings to business context:
       - For each insight, explain: what is the pattern, why it matters, and what action to take
       - Suggest visualizations to support each insight
    
    7. Output your findings in a structured markdown document:
       - Use `write_file()` to save to '{output_path}'
       - Include clear sections, bulleted lists, and recommendations
    
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

def get_feature_engineering_prompt(output_path: str, dataset_path: str, context: Dict[str, Any] = None) -> str:
    """
    Generate a prompt for feature engineering task
    
    Args:
        output_path: Path where the output code should be saved
        dataset_path: Path to the dataset
        context: Additional context parameters
        
    Returns:
        Formatted prompt string
    """
    return f"""
    # Feature Engineering Task for Tupande Program
    
    ## Objective
    Create advanced features for the Tupande program's loan repayment prediction model using the provided dataset. 
    Generate Python code that transforms raw data into meaningful features that can improve model performance.
    
    ## Data Source
    The dataset is located at: '{dataset_path}'
    
    ## Requirements
    1. Read and analyze the dataset to understand its structure:
       - Use `read_file()` to examine data documentation if available
       - Analyze column distributions and relationships
    
    2. If you encounter any questions or need clarification, use:
       - `human_intervention("clarification", "Your specific question here")` 
       - For choices between approaches: `human_intervention("multiple_choice", "Which approach?", ["Approach 1", "Approach 2"])`
       - For approval: `human_intervention("approval", "Should I proceed with this transformation?")`
    
    3. Create a comprehensive feature engineering script that:
       - Handles missing values appropriately
       - Creates features based on domain knowledge about agricultural lending
       - Implements temporal features from any time-related columns
       - Generates interaction features between relevant variables
       - Normalizes/standardizes features as needed
       - Implements one-hot encoding for categorical variables
       - Creates features that capture geographical patterns
    
    4. Produce code that:
       - Is well-documented with comments explaining the rationale for each feature
       - Is modular and reusable
       - Includes validation to ensure feature quality
       - Outputs a clean feature dataset ready for modeling
    
    5. Write the complete feature engineering script to '{output_path}'
    
    ## Output
    Your output should be a complete Python script that:
    1. Loads the raw data
    2. Implements all feature transformations
    3. Includes proper error handling
    4. Saves the engineered feature set
    5. Includes clear documentation about each feature created
    
    Focus on creating features that will meaningfully improve loan repayment prediction accuracy based on agricultural lending domain knowledge.
    """