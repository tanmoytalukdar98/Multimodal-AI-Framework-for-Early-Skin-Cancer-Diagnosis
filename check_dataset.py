import os
import pandas as pd

data_dir="data"
metadata=os.path.join(data_dir,"HAM10000_metadata.csv")
part1=os.path.join(data_dir,"HAM10000_images_part_1")
part2=os.path.join(data_dir,"HAM10000_images_part_2")

print("Metadata exists:",os.path.exists(metadata))
print("Part 1 exists:",os.path.exists(part1))
print("Part 2 exists:",os.path.exists(part2))

df=pd.read_csv(metadata)

print("\nMetadata rows:",len(df))
print("Columns:",list(df.columns))
print("\nClass distribution:")
print(df["dx"].value_counts())

print("\nPart 1 images:",len([f for f in os.listdir(part1) if f.lower().endswith(".jpg")]))
print("Part 2 images:",len([f for f in os.listdir(part2) if f.lower().endswith(".jpg")]))