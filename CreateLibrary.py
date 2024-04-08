import os

# Configuration
library_name = "your_library_name"
library_dirs = ["", "/your_library_name", "/tests"]
library_files = {
    "/your_library_name/__init__.py": "",
    "/your_library_name/dataframe_ops.py": """import pandas as pd

class DataFrameOps:
    @staticmethod
    def save_df_to_csv(df, file_name):
        df.to_csv(file_name, index=False)
        print(f"DataFrame saved to {file_name}")

    @staticmethod
    def csv_to_df(file_name):
        return pd.read_csv(file_name)

    @staticmethod
    def df_to_excel(df, file_name):
        df.to_excel(file_name, index=False)
        print(f"DataFrame saved to {file_name}")
""",
    "/your_library_name/file_ops.py": """def write_text_to_file(file_name, text):
    with open(file_name, 'w') as file:
        file.write(text)
""",
    "/tests/__init__.py": "",
    "/README.md": f"# {library_name}\n\nThis is a Python library for data manipulation and file operations.",
    "/setup.py": f"""from setuptools import setup, find_packages

setup(
    name='{library_name}',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='A Python library for data manipulation and file operations',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/{library_name}',
    packages=find_packages(),
    install_requires=[
        'pandas',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
""",
}

# Create directories
for dir_path in library_dirs:
    os.makedirs(library_name + dir_path, exist_ok=True)

# Create files
for file_path, content in library_files.items():
    with open(library_name + file_path, 'w') as f:
        f.write(content)

print(f"{library_name} library structure created successfully.")
