from PIL import Image

# Open and ensure in RGB mode - in case image is palettised
im = Image.open('aa.jpeg').convert('RGB')

# Crude conversion to black and white using 20% red, 50% green and 30% blue
matrix = (0.2, 0.5, 0.3, 0.0, 0.2, 0.5, 0.3, 0.0, 0.2, 0.5, 0.3, 0.0)

result = im.convert('RGB',matrix)

result.save('result.png')
