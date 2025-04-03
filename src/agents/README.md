# Agent Orchestrator for Tupande Program

This project implements an intelligent agent system for analyzing PDF reports and generating business insights for the One Acre Fund's Tupande program.

## Features

- **PDF Batch Processing**: Processes large PDFs in manageable batches to work within Claude's image limits
- **Human-in-the-Loop**: Interactive tools for human collaboration, clarification, and approval
- **Modular Architecture**: Separate tools, prompts, and orchestration components
- **Multiple Task Support**: Framework for both PDF analysis and feature engineering tasks

## Files

- **agent_orchestrator.py**: Main script to run the agent system
- **agent_tools.py**: Tools for file operations, PDF handling, etc.
- **agent_prompts.py**: Task prompts and instructions
- **human_intervention.py**: Interactive human-in-the-loop tools

## Usage

```bash
# Run PDF analysis task (default)
python agent_orchestrator.py

# Run with debug mode
python agent_orchestrator.py --debug

# Specify a different PDF path
python agent_orchestrator.py --pdf path/to/your/file.pdf

# Run feature engineering task
python agent_orchestrator.py --task feature_engineering
```

## Human Intervention

The system supports three types of human intervention:

1. **Clarification**: Get open-ended text input from the user
   ```python
   human_intervention("clarification", "What columns should I focus on?")
   ```

2. **Approval**: Get YES/NO confirmation
   ```python
   human_intervention("approval", "Should I proceed with this analysis?")
   ```

3. **Multiple Choice**: Present options and get selection
   ```python
   human_intervention("multiple_choice", "Which method should I use?", 
                     ["Random Forest", "Logistic Regression", "Neural Network"])
   ```

## PDF Processing

PDFs are processed in batches of 10 pages, with image cleanup to prevent memory issues:

1. `open_pdf()` - Opens and prepares the PDF for batch processing
2. `get_next_pdf_batch()` - Retrieves the next batch of pages as images

## Extending the System

To add new task types:
1. Create a new prompt function in `agent_prompts.py`
2. Add a new task function in `agent_orchestrator.py`
3. Update the CLI argument parser to include the new task type

## Requirements

- Python 3.8+
- smolagents
- litellm
- pymupdf
- dotenv
- PIL
