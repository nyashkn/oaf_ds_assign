"""
Human-in-the-loop tools for SmolAgents
"""
from typing import List, Optional, Dict, Any
from smolagents import tool

@tool
def human_intervention(
    scenario: str, 
    message_for_human: str, 
    choices: Optional[List[str]] = None
) -> str:
    """
    A universal human-in-the-loop tool that enables interactive agent-human collaboration.
    
    Args:
        scenario: One of 'clarification', 'approval', or 'multiple_choice'
            - 'clarification': Ask an open-ended question and get text input
            - 'approval': Ask for YES/NO confirmation 
            - 'multiple_choice': Present options and get selection
        message_for_human: The question or information to present to the user
        choices: List of options (required for multiple_choice scenario)
    
    Returns:
        User's response as a string
    """
    if scenario not in ["clarification", "approval", "multiple_choice"]:
        return "Error: scenario must be 'clarification', 'approval', or 'multiple_choice'."

    print("\n" + "="*50)
    print("[HUMAN INTERVENTION NEEDED]")
    print("="*50)
    print(f"Agent says: {message_for_human}")

    if scenario == "clarification":
        print("\nPlease provide additional information:")
        user_input = input("> ").strip()
        print("="*50)
        return user_input

    elif scenario == "approval":
        while True:
            print("\nType 'YES' or 'NO' to proceed:")
            user_input = input("> ").strip().upper()
            if user_input in ["YES", "NO"]:
                print("="*50)
                return user_input
            print("Invalid input. Please type 'YES' or 'NO'.")

    elif scenario == "multiple_choice":
        if not choices or not isinstance(choices, list) or len(choices) == 0:
            return "Error: No choices were provided for multiple_choice scenario."
        
        print("\nAvailable options:")
        for i, choice in enumerate(choices, start=1):
            print(f"{i}. {choice}")
        
        while True:
            try:
                print("\nEnter the number of your chosen option:")
                user_input = input("> ").strip()
                choice_num = int(user_input)
                
                if 1 <= choice_num <= len(choices):
                    selected_choice = choices[choice_num - 1]
                    print(f"You selected: {selected_choice}")
                    print("="*50)
                    # Return both the index and the text of the choice
                    return f"Option {choice_num}: {selected_choice}"
                else:
                    print(f"Please enter a number between 1 and {len(choices)}.")
            except ValueError:
                print("Please enter a valid number.")