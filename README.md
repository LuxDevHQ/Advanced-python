# Advanced Concepts Guide for Beginners

This comprehensive guide covers advanced Python concepts with beginner-friendly explanations, practical examples, and real-world applications.

---

## 1. File Operations & Data Handling

### What are Files?
Files are containers that store data on your computer. Python can read from files (like opening a book) and write to files (like writing in a notebook).

### Basic Concepts
- **Reading**: Getting data from a file into your program
- **Writing**: Putting data from your program into a file
- **File Modes**: Different ways to open files
  - `'r'` - Read only (default)
  - `'w'` - Write only (overwrites existing content)
  - `'a'` - Append (adds to the end)
  - `'rb'`, `'wb'` - Binary modes for images, videos, etc.

### Why Use `with` Statement?
The `with` statement automatically closes files, even if an error occurs. It's like having a helper that always remembers to close the door behind you.

### Examples

#### Basic File Reading
```python
# Method 1: Simple reading
with open('story.txt', 'r') as file:
    content = file.read()
    print(content)

# Method 2: Reading line by line (memory efficient)
with open('story.txt', 'r') as file:
    for line in file:
        print(line.strip())  # strip() removes newline characters

# Method 3: Reading all lines into a list
with open('story.txt', 'r') as file:
    lines = file.readlines()
    for line in lines:
        print(line.strip())
```

#### Writing to Files
```python
# Writing simple text
with open('my_diary.txt', 'w') as file:
    file.write("Today I learned Python!\n")
    file.write("It's really fun!")

# Writing multiple lines
lines = [
    "First line\n",
    "Second line\n", 
    "Third line\n"
]
with open('multiple_lines.txt', 'w') as file:
    file.writelines(lines)

# Appending to existing file
with open('my_diary.txt', 'a') as file:
    file.write("\nTomorrow I'll learn more!")
```

#### Practical Example: Student Grade Manager
```python
def save_grades(student_grades):
    """Save student grades to a file"""
    with open('grades.txt', 'w') as file:
        for student, grade in student_grades.items():
            file.write(f"{student}: {grade}\n")

def load_grades():
    """Load student grades from a file"""
    grades = {}
    try:
        with open('grades.txt', 'r') as file:
            for line in file:
                name, grade = line.strip().split(': ')
                grades[name] = int(grade)
    except FileNotFoundError:
        print("No grades file found. Starting fresh!")
    return grades

# Usage
grades = {'Alice': 85, 'Bob': 92, 'Charlie': 78}
save_grades(grades)
loaded_grades = load_grades()
print(loaded_grades)
```

#### Error Handling with Files
```python
def safe_file_read(filename):
    """Safely read a file with error handling"""
    try:
        with open(filename, 'r') as file:
            return file.read()
    except FileNotFoundError:
        return f"Error: {filename} not found!"
    except PermissionError:
        return f"Error: No permission to read {filename}!"
    except Exception as e:
        return f"Unexpected error: {e}"

# Test it
content = safe_file_read('nonexistent.txt')
print(content)
```

---

## 2. Parallel Processing

### What is Parallel Processing?
Imagine you have 4 friends helping you solve math problems. Instead of solving them one by one, each friend can work on a different problem at the same time. That's parallel processing!

### When to Use Parallel Processing
- Processing large amounts of data
- Performing calculations that can be split up
- Tasks that don't depend on each other

### Key Concepts
- **Process**: A running program
- **Thread**: A lightweight process that shares memory
- **Pool**: A collection of worker processes
- **CPU-bound**: Tasks that use lots of processing power
- **I/O-bound**: Tasks that wait for input/output (like reading files)

### Examples

#### Basic Multiprocessing
```python
import multiprocessing
import time

def square_number(n):
    """Square a number (simulate some work)"""
    time.sleep(0.1)  # Simulate work
    return n * n

def without_multiprocessing():
    """Process numbers sequentially"""
    numbers = list(range(10))
    start_time = time.time()
    
    results = []
    for num in numbers:
        results.append(square_number(num))
    
    end_time = time.time()
    print(f"Sequential: {end_time - start_time:.2f} seconds")
    return results

def with_multiprocessing():
    """Process numbers in parallel"""
    numbers = list(range(10))
    start_time = time.time()
    
    with multiprocessing.Pool() as pool:
        results = pool.map(square_number, numbers)
    
    end_time = time.time()
    print(f"Parallel: {end_time - start_time:.2f} seconds")
    return results

if __name__ == "__main__":
    # Compare speeds
    seq_results = without_multiprocessing()
    par_results = with_multiprocessing()
    print(f"Results match: {seq_results == par_results}")
```

#### Real-world Example: Image Processing
```python
import multiprocessing
from pathlib import Path

def process_image(image_path):
    """Simulate image processing"""
    # In real life, this might resize, filter, or convert images
    print(f"Processing {image_path}")
    # Simulate processing time
    import time
    time.sleep(0.5)
    return f"Processed {image_path}"

def batch_process_images(image_paths):
    """Process multiple images in parallel"""
    print(f"Processing {len(image_paths)} images...")
    
    with multiprocessing.Pool(processes=4) as pool:
        results = pool.map(process_image, image_paths)
    
    return results

# Example usage
if __name__ == "__main__":
    # Simulate a list of image files
    image_files = [f"image_{i}.jpg" for i in range(8)]
    results = batch_process_images(image_files)
    for result in results:
        print(result)
```

#### Using concurrent.futures (Easier Alternative)
```python
import concurrent.futures
import time

def download_file(url):
    """Simulate downloading a file"""
    print(f"Downloading {url}")
    time.sleep(1)  # Simulate download time
    return f"Downloaded {url}"

def download_multiple_files(urls):
    """Download multiple files concurrently"""
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        # Submit all tasks
        future_to_url = {executor.submit(download_file, url): url for url in urls}
        
        # Collect results as they complete
        results = []
        for future in concurrent.futures.as_completed(future_to_url):
            url = future_to_url[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Error downloading {url}: {e}")
        
        return results

# Example usage
urls = [
    "http://example.com/file1.txt",
    "http://example.com/file2.txt", 
    "http://example.com/file3.txt",
    "http://example.com/file4.txt"
]

results = download_multiple_files(urls)
for result in results:
    print(result)
```

---

## 3. Decorators

### What are Decorators?
A decorator is like a gift wrapper for functions. It takes a function, adds some extra functionality, and gives you back an enhanced version.

### Why Use Decorators?
- Add functionality without changing the original function
- Keep code clean and organized
- Reuse common functionality across multiple functions

### Basic Concepts
- Decorators are functions that take other functions as arguments
- They return a new function with added behavior
- The `@` symbol is syntactic sugar for applying decorators

### Examples

#### Simple Decorator
```python
def my_decorator(func):
    """A simple decorator that adds greeting messages"""
    def wrapper():
        print("Hello! About to run the function...")
        result = func()
        print("Function finished running!")
        return result
    return wrapper

# Method 1: Using @ symbol
@my_decorator
def say_goodbye():
    print("Goodbye!")

# Method 2: Manual decoration (same as above)
def say_hello():
    print("Hello!")

say_hello = my_decorator(say_hello)

# Test them
say_goodbye()
print("---")
say_hello()
```

#### Decorator with Arguments
```python
def repeat(times):
    """Decorator that repeats function execution"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            results = []
            for i in range(times):
                print(f"Execution {i + 1}:")
                result = func(*args, **kwargs)
                results.append(result)
            return results
        return wrapper
    return decorator

@repeat(3)
def greet(name):
    message = f"Hello, {name}!"
    print(message)
    return message

# Test it
results = greet("Alice")
print(f"All results: {results}")
```

#### Practical Decorators

##### Timing Decorator
```python
import time
import functools

def timer(func):
    """Measure how long a function takes to run"""
    @functools.wraps(func)  # Preserves original function's metadata
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.4f} seconds")
        return result
    return wrapper

@timer
def slow_function():
    """A function that takes some time"""
    time.sleep(1)
    return "Done!"

@timer
def calculate_sum(n):
    """Calculate sum of numbers from 1 to n"""
    return sum(range(1, n + 1))

# Test them
result1 = slow_function()
result2 = calculate_sum(1000000)
```

##### Logging Decorator
```python
import functools
from datetime import datetime

def log_calls(func):
    """Log function calls with timestamp"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] Calling {func.__name__} with args={args}, kwargs={kwargs}")
        
        try:
            result = func(*args, **kwargs)
            print(f"[{timestamp}] {func.__name__} returned: {result}")
            return result
        except Exception as e:
            print(f"[{timestamp}] {func.__name__} raised exception: {e}")
            raise
    
    return wrapper

@log_calls
def divide(a, b):
    """Divide two numbers"""
    return a / b

@log_calls  
def greet_user(name, greeting="Hello"):
    """Greet a user"""
    return f"{greeting}, {name}!"

# Test them
result1 = divide(10, 2)
result2 = greet_user("Bob", greeting="Hi")
try:
    result3 = divide(10, 0)  # This will cause an error
except ZeroDivisionError:
    print("Handled division by zero!")
```

##### Caching Decorator (Memoization)
```python
import functools

def cache_results(func):
    """Cache function results to avoid redundant calculations"""
    cache = {}
    
    @functools.wraps(func)
    def wrapper(*args):
        if args in cache:
            print(f"Cache hit for {func.__name__}{args}")
            return cache[args]
        
        print(f"Calculating {func.__name__}{args}")
        result = func(*args)
        cache[args] = result
        return result
    
    return wrapper

@cache_results
def fibonacci(n):
    """Calculate fibonacci number (slow recursive version)"""
    if n < 2:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

# Test it - notice how caching makes it much faster
print(f"fib(10) = {fibonacci(10)}")
print(f"fib(15) = {fibonacci(15)}")  # This reuses previous calculations
```

---

## 4. Lambda Functions

### What are Lambda Functions?
Lambda functions are like nicknames for simple functions. Instead of writing a full function definition, you can create a small, anonymous function in one line.

### When to Use Lambda Functions
- Simple operations that fit in one line
- With functions like `map()`, `filter()`, `sort()`
- When you need a function temporarily

### Basic Syntax
```python
lambda arguments: expression
```

### Examples

#### Basic Lambda Functions
```python
# Regular function
def add_regular(x, y):
    return x + y

# Lambda function (equivalent)
add_lambda = lambda x, y: x + y

# Test both
print(add_regular(3, 5))  # 8
print(add_lambda(3, 5))   # 8

# More examples
square = lambda x: x ** 2
is_even = lambda x: x % 2 == 0
full_name = lambda first, last: f"{first} {last}"

print(square(4))              # 16
print(is_even(7))             # False
print(full_name("John", "Doe"))  # John Doe
```

#### Lambda with Built-in Functions

##### Using with map()
```python
# map() applies a function to every item in a list
numbers = [1, 2, 3, 4, 5]

# Without lambda (using regular function)
def square_func(x):
    return x ** 2

squared_regular = list(map(square_func, numbers))

# With lambda (more concise)
squared_lambda = list(map(lambda x: x ** 2, numbers))

print(squared_regular)  # [1, 4, 9, 16, 25]
print(squared_lambda)   # [1, 4, 9, 16, 25]

# More examples
names = ["alice", "bob", "charlie"]
capitalized = list(map(lambda name: name.capitalize(), names))
print(capitalized)  # ['Alice', 'Bob', 'Charlie']

temperatures_c = [0, 20, 30, 40]
temperatures_f = list(map(lambda c: (c * 9/5) + 32, temperatures_c))
print(temperatures_f)  # [32.0, 68.0, 86.0, 104.0]
```

##### Using with filter()
```python
# filter() keeps only items that meet a condition
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Get even numbers
even_numbers = list(filter(lambda x: x % 2 == 0, numbers))
print(even_numbers)  # [2, 4, 6, 8, 10]

# Get numbers greater than 5
big_numbers = list(filter(lambda x: x > 5, numbers))
print(big_numbers)  # [6, 7, 8, 9, 10]

# Filter names by length
names = ["Al", "Bob", "Charlie", "David", "Eve"]
long_names = list(filter(lambda name: len(name) > 3, names))
print(long_names)  # ['Charlie', 'David']

# Filter positive numbers
mixed_numbers = [-3, -1, 0, 2, 5, -8, 10]
positive = list(filter(lambda x: x > 0, mixed_numbers))
print(positive)  # [2, 5, 10]
```

##### Using with sorted()
```python
# Sort with custom logic
students = [
    ("Alice", 85),
    ("Bob", 90), 
    ("Charlie", 78),
    ("Diana", 92)
]

# Sort by grade (second element)
by_grade = sorted(students, key=lambda student: student[1])
print(by_grade)  # [('Charlie', 78), ('Alice', 85), ('Bob', 90), ('Diana', 92)]

# Sort by name length
by_name_length = sorted(students, key=lambda student: len(student[0]))
print(by_name_length)  # [('Bob', 90), ('Alice', 85), ('Diana', 92), ('Charlie', 78)]

# Sort strings by last character
words = ["hello", "world", "python", "code"]
by_last_char = sorted(words, key=lambda word: word[-1])
print(by_last_char)  # ['code', 'hello', 'python', 'world']
```

#### Practical Examples

##### Data Processing
```python
# Process employee data
employees = [
    {"name": "Alice", "salary": 50000, "department": "Engineering"},
    {"name": "Bob", "salary": 45000, "department": "Marketing"},
    {"name": "Charlie", "salary": 60000, "department": "Engineering"},
    {"name": "Diana", "salary": 55000, "department": "Sales"}
]

# Get all salaries
salaries = list(map(lambda emp: emp["salary"], employees))
print(f"Salaries: {salaries}")

# Get high earners (salary > 50000)
high_earners = list(filter(lambda emp: emp["salary"] > 50000, employees))
print(f"High earners: {[emp['name'] for emp in high_earners]}")

# Sort by salary (highest first)
by_salary = sorted(employees, key=lambda emp: emp["salary"], reverse=True)
print(f"Sorted by salary: {[emp['name'] for emp in by_salary]}")
```

##### Mathematical Operations
```python
# Create a list of mathematical operations
operations = [
    ("add", lambda x, y: x + y),
    ("subtract", lambda x, y: x - y),
    ("multiply", lambda x, y: x * y),
    ("divide", lambda x, y: x / y if y != 0 else "Cannot divide by zero")
]

# Use the operations
a, b = 10, 3
for name, operation in operations:
    result = operation(a, b)
    print(f"{name}({a}, {b}) = {result}")
```

#### Lambda Limitations
```python
# Things you CAN'T do with lambda:

# 1. Multiple statements (use regular function instead)
# This won't work:
# bad_lambda = lambda x: print(x); return x * 2

# Use regular function:
def print_and_double(x):
    print(x)
    return x * 2

# 2. Complex logic (use regular function instead)
# This is hard to read:
# complex_lambda = lambda x: "positive" if x > 0 else "negative" if x < 0 else "zero"

# Better as regular function:
def describe_number(x):
    if x > 0:
        return "positive"
    elif x < 0:
        return "negative"
    else:
        return "zero"
```

---

## 5. List Comprehensions

### What are List Comprehensions?
List comprehensions are a concise way to create lists. They're like a recipe that tells Python exactly how to build a list step by step.

### Basic Syntax
```python
[expression for item in iterable]
[expression for item in iterable if condition]
```

### Why Use List Comprehensions?
- More readable than loops
- Often faster than traditional loops
- More "Pythonic" (follows Python style guidelines)

### Examples

#### Basic List Comprehensions
```python
# Traditional way with loops
squares_traditional = []
for x in range(10):
    squares_traditional.append(x ** 2)

# List comprehension way (more concise)
squares_comprehension = [x ** 2 for x in range(10)]

print(squares_traditional)    # [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
print(squares_comprehension)  # [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]

# More examples
numbers = [1, 2, 3, 4, 5]
doubled = [x * 2 for x in numbers]
print(doubled)  # [2, 4, 6, 8, 10]

words = ["hello", "world", "python"]
lengths = [len(word) for word in words]
print(lengths)  # [5, 5, 6]

names = ["alice", "bob", "charlie"]
capitalized = [name.capitalize() for name in names]
print(capitalized)  # ['Alice', 'Bob', 'Charlie']
```

#### List Comprehensions with Conditions
```python
# Get even numbers
numbers = range(20)
evens = [x for x in numbers if x % 2 == 0]
print(evens)  # [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]

# Get positive numbers
mixed_numbers = [-5, -2, 0, 3, 7, -1, 8]
positives = [x for x in mixed_numbers if x > 0]
print(positives)  # [3, 7, 8]

# Get long words
words = ["cat", "elephant", "dog", "hippopotamus", "ant"]
long_words = [word for word in words if len(word) > 3]
print(long_words)  # ['elephant', 'hippopotamus']

# Transform and filter at the same time
numbers = range(10)
even_squares = [x ** 2 for x in numbers if x % 2 == 0]
print(even_squares)  # [0, 4, 16, 36, 64]
```

#### Nested List Comprehensions
```python
# Create a multiplication table
multiplication_table = [[i * j for j in range(1, 4)] for i in range(1, 4)]
print(multiplication_table)  # [[1, 2, 3], [2, 4, 6], [3, 6, 9]]

# Flatten a nested list
nested = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
flattened = [num for sublist in nested for num in sublist]
print(flattened)  # [1, 2, 3, 4, 5, 6, 7, 8, 9]

# Get all combinations
colors = ["red", "blue"]
sizes = ["small", "large"]
combinations = [f"{color}-{size}" for color in colors for size in sizes]
print(combinations)  # ['red-small', 'red-large', 'blue-small', 'blue-large']
```

#### Practical Examples

##### Data Processing
```python
# Process student grades
students = [
    {"name": "Alice", "grade": 85},
    {"name": "Bob", "grade": 92},
    {"name": "Charlie", "grade": 78},
    {"name": "Diana", "grade": 96},
    {"name": "Eve", "grade": 73}
]

# Get names of students who passed (grade >= 80)
passed_students = [student["name"] for student in students if student["grade"] >= 80]
print(f"Passed: {passed_students}")  # ['Alice', 'Bob', 'Diana']

# Convert grades to letter grades
def grade_to_letter(grade):
    if grade >= 90: return 'A'
    elif grade >= 80: return 'B' 
    elif grade >= 70: return 'C'
    else: return 'F'

letter_grades = [grade_to_letter(student["grade"]) for student in students]
print(f"Letter grades: {letter_grades}")  # ['B', 'A', 'C', 'A', 'C']
```

##### Text Processing
```python
# Process a sentence
sentence = "The quick brown fox jumps over the lazy dog"
words = sentence.split()

# Get words longer than 4 characters
long_words = [word for word in words if len(word) > 4]
print(long_words)  # ['quick', 'brown', 'jumps']

# Get first letter of each word
initials = [word[0].upper() for word in words]
print(''.join(initials))  # TQBFJOTLD

# Remove vowels from each word
consonants_only = [''.join([char for char in word if char.lower() not in 'aeiou']) 
                   for word in words]
print(consonants_only)  # ['Th', 'qck', 'brwn', 'fx', 'jmps', 'vr', 'th', 'lzy', 'dg']
```

##### Mathematical Operations
```python
# Generate fibonacci sequence
def generate_fibonacci(n):
    fib = [0, 1]
    [fib.append(fib[-1] + fib[-2]) for _ in range(n - 2)]
    return fib[:n]

print(generate_fibonacci(10))  # [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]

# Prime numbers (simple approach)
def is_prime(n):
    if n < 2:
        return False
    return all(n % i != 0 for i in range(2, int(n ** 0.5) + 1))

primes = [x for x in range(2, 50) if is_prime(x)]
print(primes)  # [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
```

#### When NOT to Use List Comprehensions
```python
# Too complex - use regular loop instead
# Bad (hard to read):
complex_bad = [x if x > 0 else -x if x < 0 else 0 for x in range(-5, 6) if x != 0]

# Better (more readable):
def process_number(x):
    if x > 0:
        return x
    elif x < 0:
        return -x
    else:
        return 0

complex_good = []
for x in range(-5, 6):
    if x != 0:
        complex_good.append(process_number(x))

print(complex_good)  # [5, 4, 3, 2, 1, 1, 2, 3, 4, 5]
```

---

## 6. Dictionary Comprehensions

### What are Dictionary Comprehensions?
Dictionary comprehensions are like list comprehensions, but they create dictionaries instead of lists. They're a concise way to build dictionaries with key-value pairs.

### Basic Syntax
```python
{key_expression: value_expression for item in iterable}
{key_expression: value_expression for item in iterable if condition}
```

### Examples

#### Basic Dictionary Comprehensions
```python
# Create a dictionary of squares
numbers = range(5)
squares_dict = {x: x ** 2 for x in numbers}
print(squares_dict)  # {0: 0, 1: 1, 2: 4, 3: 9, 4: 16}

# Create dictionary from two lists
names = ["Alice", "Bob", "Charlie"]
ages = [25, 30, 35]
name_age_dict = {name: age for name, age in zip(names, ages)}
print(name_age_dict)  # {'Alice': 25, 'Bob': 30, 'Charlie': 35}

# String lengths
words = ["cat", "elephant", "dog", "hippopotamus"]
word_lengths = {word: len(word) for word in words}
print(word_lengths)  # {'cat': 3, 'elephant': 8, 'dog': 3, 'hippopotamus': 12}
```

#### Dictionary Comprehensions with Conditions
```python
# Only include even numbers
numbers = range(10)
even_squares = {x: x ** 2 for x in numbers if x % 2 == 0}
print(even_squares)  # {0: 0, 2: 4, 4: 16, 6: 36, 8: 64}

# Only include long words
words = ["cat", "elephant", "dog", "hippopotamus", "ant"]
long_word_lengths = {word: len(word) for word in words if len(word) > 3}
print(long_word_lengths)  # {'elephant': 8, 'hippopotamus': 12}

# Grade classifications
students_grades = {"Alice": 85, "Bob": 92, "Charlie": 78, "Diana": 96}
high_performers = {name: grade for name, grade in students_grades.items() if grade >= 90}
print(high_performers)  # {'Bob': 92, 'Diana': 96}
```

#### Transforming Existing Dictionaries
```python
# Original dictionary
original_prices = {"apple": 1.0, "banana": 0.5, "orange": 0.8, "grape": 2.0}

# Apply discount
discounted_prices = {item: price * 0.9 for item, price in original_prices.items()}
print(discounted_prices)  # {'apple': 0.9, 'banana': 0.45, 'orange': 0.72, 'grape': 1.8}

# Convert to uppercase keys
upper_prices = {item.upper(): price for item, price in original_prices.items()}
print(upper_prices)  # {'APPLE': 1.0, 'BANANA': 0.5, 'ORANGE': 0.8, 'GRAPE': 2.0}

# Swap keys and values (be careful - values must be unique!)
swapped = {price: item for item, price in original_prices.items()}
print(swapped)  # {1.0: 'apple', 0.5: 'banana', 0.8: 'orange', 2.0: 'grape'}
```

#### Practical Examples

##### Data Analysis
```python
# Sales data processing
sales_data = [
    {"product": "laptop", "quantity": 5, "price": 1000},
    {"product": "mouse", "quantity": 20, "price": 25},
    {"product": "keyboard", "quantity": 15, "price": 75},
    {"product": "monitor", "quantity": 8, "price": 300}
]

# Calculate total revenue per product
revenue_per_product = {
    item["product"]: item["quantity"] * item["price"] 
    for item in sales_data
}
print(revenue_per_product)  
# {'laptop': 5000, 'mouse': 500, 'keyboard': 1125, 'monitor': 2400}

# High-value products (revenue > 1000)
high_value_products = {
    item["product"]: item["quantity"] * item["price"] 
    for item in sales_data 
    if item["quantity"] * item["price"] > 1000
}
print(high_value_products)  # {'laptop': 5000, 'keyboard': 1125, 'monitor': 2400}
