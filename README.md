ColtCruz-Image-Classification-and-Processing
Part 1: Basic Classifier and Grad-CAM
Classifier Code Explanation
I asked the AI to explain each line of the base_classifier.py program. It walked through everything from importing TensorFlow and loading the image, to how the model processes the image and outputs top predictions. Since I don't have much Python experience, I was curious if the explanations would make sense. They actually did. The AI described things clearly enough that I could follow what was happening step by step. I understood how the image was preprocessed, passed through the model, and how predictions were decoded and printed. Overall, the explanation helped me gain confidence in working with unfamiliar code.

Classifier Results
I ran the base_classifier.py program on an image of a Crowned Jellyfish. Here are the top-3 predictions:

jellyfish (0.77)

umbrella (0.03)

volcano (0.02)

These results are accurate. The classifier correctly identified the jellyfish and was mostly confident about it.

Grad-CAM Integration
I asked the AI how to integrate Grad-CAM into my classifier. It generated code that creates a heatmap overlay showing where the model focused most when making its prediction. I added the code to the same program and generated a visualization called gradcam_output.jpg. The heatmap showed the model mainly focused on the center of the jellyfish, especially the body and surrounding water contrast. That part of the image seems to be the most important feature for classification.

What I Learned
Grad-CAM helped me better understand what parts of the image influence the prediction. I’ve seen AI described as a black box, but this process made it feel a lot more transparent. I was surprised by how precise the model could be and how useful it is to visualize its attention.

Part 2: Occlusion Testing
Occlusion Ideas
I prompted the AI to suggest three ways to obscure the important parts of the image identified by Grad-CAM. It suggested black box occlusion, blur occlusion, and noise occlusion. A black box completely blocks the region with a solid rectangle. A blur smooths out the region to reduce clarity. Noise replaces the region with random pixels to disrupt the pattern the model relies on.

Occlusion Implementation
I added these occlusion types directly into my base_classifier.py file. Each one targets the region highlighted in the Grad-CAM heatmap. It generates three new images:

occlusion_black_box.jpg

occlusion_blur.jpg

occlusion_noise.jpg

Occlusion Results
After testing the classifier again on each occluded image, I observed the following. The black box caused the classifier to misclassify or drop confidence significantly. The blur had a mild effect. It still predicted "jellyfish" but with slightly less confidence. The noise confused the model the most, leading to erratic predictions and low confidence.

Analysis
The classifier definitely struggled more with the noise and black box occlusions. These techniques removed or distorted the critical features used by the model. This showed how dependent the model is on very specific image regions. Occlusion helped me test the model’s weaknesses and how fragile classification can be when those areas are modified.

Part 3: Image Filtering
Code Walkthrough
I used the basic_filter.py script to apply a blur to the image. I asked the AI to explain the code line by line, and even though I’m still new to Python, I was able to understand how the script opens the image, resizes it, applies a filter, and saves the result. It clarified how each module like ImageFilter and matplotlib contributed to the final output.

Additional Filters
I asked the AI to suggest three other filters. It recommended edge detection, sharpening, and emboss. Edge detection highlights outlines and structures in the image. Sharpening enhances fine detail and makes the image more defined. Emboss adds a relief effect that makes the image look engraved or 3D.

I implemented all three using the PIL library’s built-in filters. Here are the saved output images:

filter_blur.png

filter_edges.png

filter_sharpen.png

filter_emboss.png

Final Thoughts on Filters
Each filter created a noticeably different artistic style. Edge detection gave the image a sketched outline look. Sharpening made the jellyfish pop out more clearly. Embossing added a cool metallic texture. I didn’t go overboard with exaggerated styles, but I see the potential to build out more stylized effects with more time.

Final Reflection
This project pushed me out of my comfort zone with Python, image processing, and neural networks. I leaned heavily on the AI to explain unfamiliar code and guide implementation, and it worked. I now understand how an image classifier makes predictions, how Grad-CAM helps visualize those decisions, and how modifying input can affect outcomes. The filters section gave me room to be a little creative too. Overall, this was a hands-on, challenging, and rewarding experience. I would absolutely use AI again to assist in future development work like this.
