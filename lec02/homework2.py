'''
This homework defines one method, called "arithmetic".
that method, type `help homework2.arithmetic`.
'''

def arithmetic(x, y):
  
    if isinstance(y, str) and isinstance(x, str):
        return x + y
    
    elif isinstance(y, str) and isinstance(x, float):
        return str(x) + y

    elif isinstance(y, float) and isinstance(x, str):
        return x * int(y)

    elif isinstance(y, float) and isinstance(x, float):
        return x * y

    else:
        raise TypeError("Unsupported combination of types for x and y")

