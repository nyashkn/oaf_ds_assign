"""
Styles package for the regression reporting module.

This package contains style definitions, colors, and formatting for report generation.
"""

from .colors import (
    OAF_GREEN, OAF_BLUE, OAF_LIGHT_GREEN, OAF_LIGHT_BLUE, OAF_DARK_GREEN,
    TUPANDE_BLUE, TUPANDE_RED, TUPANDE_GREEN, TUPANDE_DARK, TUPANDE_LIGHT,
    POSITIVE_COLOR, NEGATIVE_COLOR, NEUTRAL_COLOR,
    TABLE_HEADER_COLOR, TABLE_ACCENT_COLOR, TABLE_ALT_ROW_COLOR
)
from .paragraph_styles import (
    create_styles, 
    TITLE_STYLES, TEXT_STYLES, HEADING_STYLES, TABLE_STYLES, TOC_STYLES
)

__all__ = [
    # Colors
    'OAF_GREEN', 'OAF_BLUE', 'OAF_LIGHT_GREEN', 'OAF_LIGHT_BLUE', 'OAF_DARK_GREEN',
    'TUPANDE_BLUE', 'TUPANDE_RED', 'TUPANDE_GREEN', 'TUPANDE_DARK', 'TUPANDE_LIGHT',
    'POSITIVE_COLOR', 'NEGATIVE_COLOR', 'NEUTRAL_COLOR',
    'TABLE_HEADER_COLOR', 'TABLE_ACCENT_COLOR', 'TABLE_ALT_ROW_COLOR',
    
    # Style functions
    'create_styles',
    
    # Style dictionaries
    'TITLE_STYLES', 'TEXT_STYLES', 'HEADING_STYLES', 'TABLE_STYLES', 'TOC_STYLES'
]
