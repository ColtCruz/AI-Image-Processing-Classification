from PIL import Image, ImageFilter
import matplotlib.pyplot as plt

def apply_blur_filter(image_path):
    try:
        img = Image.open(image_path)
        img_resized = img.resize((128, 128))
        
        # 1. Gaussian Blur
        img_blurred = img_resized.filter(ImageFilter.GaussianBlur(radius=2))
        img_blurred.save("filter_blur.png")

        # 2. Edge Detection
        img_edges = img_resized.filter(ImageFilter.FIND_EDGES)
        img_edges.save("filter_edges.png")

        # 3. Sharpen
        img_sharpened = img_resized.filter(ImageFilter.SHARPEN)
        img_sharpened.save("filter_sharpen.png")

        # 4. Emboss
        img_emboss = img_resized.filter(ImageFilter.EMBOSS)
        img_emboss.save("filter_emboss.png")

        print("Filters applied and images saved: blur, edges, sharpen, emboss.")


    except Exception as e:
        print(f"Error processing image: {e}")

if __name__ == "__main__":
    image_path = "Crowned_Jellyfish.jpg"  # Replace with the path to your image file
    apply_blur_filter(image_path)