from PIL import Image
import os

# Configuration
rows = 5
cols = 4
image_width = 1080  # adjust based on your actual image size
image_height = 1920
output_path = '/mnt/d/University/university_notes/summer_semester_24/Thesis/latex/pictures/expr/images/image_grid.jpg'
path = '/mnt/d/University/university_notes/summer_semester_24/Thesis/latex/pictures/expr/images'  # specify the path to your images

# Create a blank white canvas
grid_width = cols * image_width
grid_height = rows * image_height
grid_img = Image.new('RGB', (grid_width, grid_height), 'white')

# Paste each image into the grid
for row in range(1, rows + 1):
    for col in range(1, cols + 1):
        img_name = f'{path}/{row}_{col}.png'
        if not os.path.exists(img_name):
            print(f'Warning: {img_name} not found. Skipping.')
            continue
        img = Image.open(img_name).resize((image_width, image_height))
        x = (col - 1) * image_width
        y = (row - 1) * image_height
        grid_img.paste(img, (x, y))

# Save the final grid
grid_img.save(output_path)
print(f"Saved image grid to '{output_path}'")
