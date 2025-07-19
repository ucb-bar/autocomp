import os

# Loop over levels 1, 2, 3, 4
for level in [1, 2, 3, 4]:
    directory = f"kb-level{level}"
    
    # Skip if the directory does not exist
    if not os.path.isdir(directory):
        print(f"Directory {directory} does not exist. Skipping.")
        continue

    print(f"Processing {directory}")

    # Iterate through files in the directory
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)

        # Only process regular files
        if os.path.isfile(filepath):
            with open(filepath, 'r') as f:
                content = f.read()

            # Replace target string
            new_content = content.replace("super(Model", "super(ModelNew")

            # Overwrite file only if changes were made
            if new_content != content:
                with open(filepath, 'w') as f:
                    f.write(new_content)
                print(f"Updated {filepath}")
            else:
                print(f"No change in {filepath}")