
from datetime import datetime
from typing import Optional


def format_dob_for_ec(date_str: Optional[str]) -> Optional[str]:
    """
    Converts a date string like '01 Feb 1982' to '1982-02-01'.
    - If the year starts with 4 or 7, replaces the first digit with 1 (e.g., 4982 -> 1982).
    - Validates that the year is >= 1900 and not in the future.
    Returns None if invalid.
    """
    if not date_str:
        return None

    try:
        date_obj = datetime.strptime(date_str.strip(), "%d %b %Y")
        year_str = str(date_obj.year)
        
        # Fix the year if it starts with 4 or 7
        if year_str.startswith(("4", "7")):
            corrected_year = int("1" + year_str[1:])
            date_obj = date_obj.replace(year=corrected_year)

        # Validate corrected date
        if date_obj.year < 1900 or date_obj > datetime.now():
            return None

        return date_obj.strftime("%Y-%m-%d")
    except ValueError:
        return None
    