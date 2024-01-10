import re
import shutil
import os

with open("../README.md", "r", encoding="utf-8") as f:
    lines = f.readlines()


for line in lines:
    if ".png" in line:
        match = re.search(r"\!\[([^\]]+)\]\((.+)\)", line)
        if match:
            image_name = match.group(1).strip()
            image_path = match.group(2).strip()
            print(f"Image Name: {image_name}")
            print(f"Image Path: {image_path}")
            current_directory = os.getcwd()
            image_name_with_extension = os.path.basename(image_path)
            destination_path = os.path.join(
                current_directory, image_name_with_extension
            )
            try:
              shutil.copy2(image_path, destination_path)
            except:
              pass
        else:
            print("No match found")
