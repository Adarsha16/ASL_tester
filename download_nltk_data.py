import nltk

print("Downloading/Verifying NLTK data packages...")

# A list of all required packages
packages = [
    "punkt",  # For tokenizing sentences
    "wordnet",  # For lemmatization
    "omw-1.4",  # For lemmatization
    "averaged_perceptron_tagger",  # For Part-of-Speech tagging (this is the missing one)
]

for package in packages:
    print(f"--> Checking for {package}")
    nltk.download(package, quiet=True)

print("\nAll required NLTK packages are present.")
