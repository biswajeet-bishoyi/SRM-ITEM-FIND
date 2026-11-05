"""
Text Preprocessing and Cleaning Module for SRM CampusFind.
Handles normalization, noise removal, tokenization, and metadata extraction.
"""

import re

# SRM Campus Known Locations Mapping
SRM_LOCATIONS = [
    "central library", "tech park", "university building", "ub", 
    "java canteen", "mtp canteen", "abode valley", "biotech block", 
    "architecture block", "srm hospital", "clock tower", "main gate",
    "estates block", "oat", "open air theatre", "sports complex"
]

COLOR_KEYWORDS = [
    "black", "blue", "navy", "red", "white", "silver", "grey", "gray", 
    "gold", "pink", "green", "yellow", "brown", "purple"
]

CATEGORY_KEYWORDS = {
    "electronics": ["earbuds", "earphones", "laptop", "phone", "iphone", "samsung", "charger", "airpods", "ipad", "powerbank", "watch", "smartwatch"],
    "bags": ["backpack", "bag", "purse", "wallet", "pouch", "duffel", "handbag"],
    "documents": ["id card", "identity card", "passport", "notebook", "book", "drive license", "hall ticket"],
    "accessories": ["bottle", "water bottle", "keys", "keychain", "glasses", "spectacles", "sunglasses", "umbrella", "calculator"]
}

def clean_text(text: str) -> str:
    """Normalize and clean input text description."""
    if not text:
        return ""
    text = text.lower()
    # Remove special characters except spaces and hyphens
    text = re.sub(r'[^a-z0-9\s\-]', ' ', text)
    # Collapse multiple whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_structured_attributes(text: str) -> dict:
    """Extract candidate categories, colors, and locations directly from raw description."""
    cleaned = clean_text(text)
    found_colors = [color for color in COLOR_KEYWORDS if color in cleaned]
    found_locations = [loc for loc in SRM_LOCATIONS if loc in cleaned]
    
    found_category = "other"
    for category, keywords in CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw in cleaned:
                found_category = category
                break
        if found_category != "other":
            break
            
    return {
        "category": found_category,
        "colors": found_colors,
        "detected_locations": found_locations,
        "cleaned_text": cleaned
    }
