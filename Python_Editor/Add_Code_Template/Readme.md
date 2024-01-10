## How to add Class and Function Template

1. Go to file, setting, editor and click on file and code templates.
2. Click on plus add icon and then add class, function and main code.
- Class
```
# -*- coding: utf-8 -*-
"""
Author: Your Name
Date: 2023-11-18
Version: 1.0
"""
class Person:
    def __init__(self, name, age) -> object:
        """

        :rtype: object
        :param name:
        :param age:
        """
        self.name = name
        self.age = age

    def __str__(self):
        """

        :rtype: object
        """
        return f"{self.name}({self.age})"


```

- Function
```
# -*- coding: utf-8 -*-
"""
Author: Your Name
Date: 2023-11-18
Version: 1.0
"""


def func(a: object) -> object:
    """
    :rtype: object

    """
    return  a


def func1(b: object) -> object:
    """
    :rtype: object

    """
    return  b

```
- Main Code
```
# -*- coding: utf-8 -*-
"""
Author: Your Name
Date: 2023-11-18
Version: 1.0
"""


def main():
    """

    :rtype: object
    """
    print("Hello, World!")

    # More code...


if __name__ == "__main__":
    main()

```
