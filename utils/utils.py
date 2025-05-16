import re

def parse_text(text):
    text = text.strip()
    
    sections = re.split(r"<begin_of_point>", text)
    prefix = sections[0].strip() if sections[0] else None
    
    points = []
    conclusion = None
    
    for section in sections[1:]:
        parts = section.split("<end_of_point>", 1)
        if len(parts) == 2:
            points.append(parts[0].strip())
            conclusion = parts[1].strip() if parts[1] else None
        else:
            points.append(parts[0].strip())
    
    return prefix, points, conclusion


def extract_content_by_regex(text, start_marker, end_marker):
    """
    extract content between start marker and end marker
    """
    pattern = re.escape(start_marker) + r"(.*?)" + re.escape(end_marker)

    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def get_boxed_content(text):
    """
    extract content from \boxed{}
    """
    content =  re.findall(r"\\boxed\{(.*?)\}", text)
    
    if content:
        return content[0]
    
    return None