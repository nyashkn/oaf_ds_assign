"""
Color definitions for the regression reporting module.

This module defines color schemes for reports, including OAF and Tupande branding colors.
"""

from reportlab.lib import colors

# Define OneAcre Fund and Tupande theme colors
OAF_GREEN = colors.HexColor('#32a852')  # Main green for titles
OAF_BLUE = colors.HexColor('#3d85c6')   # Blue for headings
OAF_LIGHT_GREEN = colors.HexColor('#b6d7a8')  # Light green for accents
OAF_LIGHT_BLUE = colors.HexColor('#a4c2f4')   # Light blue for tables
OAF_DARK_GREEN = colors.HexColor('#274e13')   # Dark green for text

# Tupande brand colors
TUPANDE_BLUE = colors.HexColor('#007bff')
TUPANDE_RED = colors.HexColor('#dc3545')
TUPANDE_GREEN = colors.HexColor('#28a745')
TUPANDE_DARK = colors.HexColor('#333333')
TUPANDE_LIGHT = colors.HexColor('#f8f9fa')

# Analysis color schemes
POSITIVE_COLOR = OAF_GREEN
NEGATIVE_COLOR = TUPANDE_RED
NEUTRAL_COLOR = OAF_BLUE

# Table color schemes
TABLE_HEADER_COLOR = OAF_BLUE
TABLE_ACCENT_COLOR = OAF_LIGHT_GREEN
TABLE_ALT_ROW_COLOR = colors.whitesmoke
