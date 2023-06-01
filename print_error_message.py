import traceback

def print_error_message():
    """
    Prints an error message with the function name, line and nature of the error
    that occurred. Useful in combination with try/except blocks. See example below.
    
    Example
    -------
    try:
        x = [3, 4, 5] + [3, 4]
    except:
        print_error_message()
    
    Created on 2016-01-18 by Sam NH
    """
    error_message = traceback.format_exc()
    print(error_message)

if __name__ == '__main__':
        
    try:
        x = [3, 4, 5] + [3, 4]
    except:
        print_error_message()

    try:
        x = [3, 4, 5] + 3
    except:
        print_error_message()