def to_json_string(obj, indent=0, indent_str='  '):
    """
    Convert a nested list or dictionary into a JSON-like string.
    
    Args:
        obj: List, dict, or primitive value to convert
        indent: Current indentation level
        indent_str: String to use for each indentation level
    
    Returns:
        JSON-like string representation
    """
    current_indent = indent_str * indent
    next_indent = indent_str * (indent + 1)
    
    if isinstance(obj, dict):
        if not obj:
            return '{}'
        
        lines = ['{']
        items = list(obj.items())
        for i, (key, value) in enumerate(items):
            comma = ',' if i < len(items) - 1 else ''
            value_str = to_json_string(value, indent + 1, indent_str)
            
            # Check if value is multiline
            if '\n' in value_str:
                lines.append(f'{next_indent}"{key}": {value_str}{comma}')
            else:
                lines.append(f'{next_indent}"{key}": {value_str}{comma}')
        
        lines.append(current_indent + '}')
        return '\n'.join(lines)
    
    elif isinstance(obj, list):
        if not obj:
            return '[]'
        
        lines = ['[']
        for i, item in enumerate(obj):
            comma = ',' if i < len(obj) - 1 else ''
            item_str = to_json_string(item, indent + 1, indent_str)
            
            # Check if item is multiline
            if '\n' in item_str:
                lines.append(f'{next_indent}{item_str}{comma}')
            else:
                lines.append(f'{next_indent}{item_str}{comma}')
        
        lines.append(current_indent + ']')
        return '\n'.join(lines)
    
    elif isinstance(obj, str):
        return f'"{obj}"'
    
    elif obj is None:
        return 'null'
    
    elif isinstance(obj, bool):
        return 'true' if obj else 'false'
    
    else:
        return str(obj)