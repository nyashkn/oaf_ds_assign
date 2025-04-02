"""
Paragraph style definitions for the regression reporting module.

This module provides style definitions for various paragraph types used in the reports.
"""

from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib import colors

from .colors import (
    OAF_GREEN, OAF_BLUE, OAF_LIGHT_GREEN, OAF_LIGHT_BLUE, OAF_DARK_GREEN,
    TUPANDE_BLUE, TUPANDE_RED, TUPANDE_GREEN, TUPANDE_DARK, TUPANDE_LIGHT
)

# Title page style constants
TITLE_STYLES = {
    'ReportTitle': {
        'fontSize': 24,
        'leading': 30,
        'textColor': OAF_DARK_GREEN,
        'alignment': TA_CENTER,
        'spaceAfter': 36
    },
    'ReportSubtitle': {
        'fontSize': 18,
        'leading': 24,
        'textColor': OAF_BLUE,
        'alignment': TA_CENTER,
        'spaceAfter': 24
    },
    'ReportAuthor': {
        'fontSize': 12,
        'leading': 16,
        'textColor': colors.black,
        'alignment': TA_CENTER,
        'spaceAfter': 12
    }
}

# Heading style constants
HEADING_STYLES = {
    'Chapter': {
        'fontSize': 18,
        'leading': 24,
        'textColor': OAF_GREEN,
        'spaceBefore': 24,
        'spaceAfter': 12
    },
    'Section': {
        'fontSize': 14,
        'leading': 20,
        'textColor': OAF_BLUE,
        'spaceBefore': 14,
        'spaceAfter': 8
    },
    'Subsection': {
        'fontSize': 12,
        'leading': 16,
        'textColor': OAF_DARK_GREEN,
        'spaceBefore': 10,
        'spaceAfter': 6
    }
}

# Text style constants
TEXT_STYLES = {
    'BodyText': {
        'fontSize': 10,
        'leading': 14,
        'spaceBefore': 6,
        'spaceAfter': 6
    },
    'CaptionText': {
        'fontSize': 9,
        'leading': 12,
        'alignment': TA_CENTER,
        'spaceBefore': 2,
        'spaceAfter': 6,
        'fontName': 'Helvetica-Oblique'
    },
    'CodeText': {
        'fontSize': 9,
        'leading': 12,
        'fontName': 'Courier',
        'backColor': colors.whitesmoke,
        'borderWidth': 1,
        'borderColor': colors.lightgrey,
        'borderPadding': 5,
        'spaceBefore': 6,
        'spaceAfter': 6
    },
    'NoteText': {
        'fontSize': 9,
        'leading': 12,
        'textColor': colors.darkgrey,
        'spaceBefore': 4,
        'spaceAfter': 4,
        'fontName': 'Helvetica-Oblique'
    }
}

# Table style constants
TABLE_STYLES = {
    'TableHeader': {
        'fontSize': 10,
        'leading': 14,
        'alignment': TA_CENTER,
        'textColor': colors.white
    },
    'TableCell': {
        'fontSize': 9,
        'leading': 12,
        'alignment': TA_CENTER
    }
}

# TOC style constants
TOC_STYLES = {
    'TOCHeading': {
        'fontSize': 16,
        'leading': 20,
        'textColor': OAF_DARK_GREEN,
        'spaceBefore': 12,
        'spaceAfter': 6
    },
    'TOCChapter': {
        'fontSize': 12,
        'leading': 16,
        'spaceBefore': 6,
        'spaceAfter': 2
    },
    'TOCSection': {
        'fontSize': 10,
        'leading': 14,
        'leftIndent': 20,
        'spaceBefore': 2,
        'spaceAfter': 2
    }
}

def create_styles() -> dict:
    """
    Create style objects for the report based on the defined constants.
    
    Returns:
        Dictionary of ParagraphStyle objects
    """
    # Get the standard sample stylesheet
    styles = getSampleStyleSheet()
    
    # Helper function to safely add styles
    def add_style_safely(name, parent_style, props):
        # If style already exists, modify it instead of adding
        if name in styles:
            for prop, value in props.items():
                setattr(styles[name], prop, value)
        else:
            # Add new style
            styles.add(ParagraphStyle(
                name=name,
                parent=styles[parent_style],
                **props
            ))
    
    # Add title styles
    for name, style_props in TITLE_STYLES.items():
        add_style_safely(name, 'Title', style_props)
    
    # Add heading styles
    for name, style_props in HEADING_STYLES.items():
        parent_style = 'Heading1' if name == 'Chapter' else 'Heading2' if name == 'Section' else 'Heading3'
        add_style_safely(name, parent_style, style_props)
    
    # Add text styles
    for name, style_props in TEXT_STYLES.items():
        add_style_safely(name, 'Normal', style_props)
    
    # Add table styles
    for name, style_props in TABLE_STYLES.items():
        add_style_safely(name, 'Normal', style_props)
    
    # Add TOC styles
    for name, style_props in TOC_STYLES.items():
        parent_style = 'Heading1' if name == 'TOCHeading' else 'Normal'
        add_style_safely(name, parent_style, style_props)
    
    return styles
