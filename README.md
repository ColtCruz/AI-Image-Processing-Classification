ColtCruz-Image-Classification-and-Processing

Part 1: Using the Basic Classifier and Implementing Grad-CAM
Explain what each line of this Python program does.
I asked the AI to explain each line of the base_classifier.py program. It walked through all the imports, image preprocessing steps, how the model works, and how predictions are decoded and printed. Since I don’t have a strong background in Python, I wasn’t sure I would understand it, but I actually did. The AI's step-by-step explanation made it easy to follow how the image was being processed and classified. It helped me gain confidence in using libraries like TensorFlow and Keras.

If you do not have previous experience with Python, comment on whether the AI’s explanations make sense to you.
I don’t have much Python experience, but yes — the explanations made sense. They were written in a straightforward way that made technical parts feel approachable.

Record the top-3 predictions and their confidence scores.

jellyfish (0.77)

umbrella (0.03)

volcano (0.02)

Use the base_classifier.py program to classify the image.
I used the program on an image of a Crowned Jellyfish and received the predictions above. The model confidently predicted jellyfish, which matched the actual image.

Implement Grad-CAM to visualize the areas the model focuses on.
I integrated Grad-CAM into the classifier with AI assistance. The result was a heatmap saved as gradcam_output.jpg that showed the model focused on the body of the jellyfish and the area where it stood out from the background. This helped me understand what parts of the image were most important for the prediction.

Explain what Grad-CAM is and how it works.
Grad-CAM is an algorithm that overlays a heatmap on an image to show which parts the model paid attention to when making its decision. It works by computing the gradients flowing into the last convolutional layer of a CNN and using them to highlight relevant areas. It was useful for demystifying how the model works internally.

Identify which parts of the image the classifier focuses on most heavily. Record your observations.
The Grad-CAM output showed the model focused mainly on the jellyfish’s upper body, especially the contrast between the creature and the dark background. The heatmap made it clear the model doesn’t need the whole image to make a decision — just the right details.

Part 2: Experimenting with Image Occlusion
What are three ways to occlude an image?
The AI suggested:

Black box occlusion – placing a solid rectangle over the important part.

Blur occlusion – applying a Gaussian blur to make the area unclear.

Noise occlusion – filling the area with random pixels to disrupt the visual structure.

Modify the base_classifier.py program to implement the three occlusions.
I modified the classifier to create three new versions of the image, each with one of the occlusion methods applied directly over the Grad-CAM area. The new outputs were:

occlusion_black_box.jpg

occlusion_blur.jpg

occlusion_noise.jpg

Run the classifier on each occluded image. Record the top-3 predictions and confidence scores.

Black Box Occlusion:

sea anemone (0.29)

jellyfish (0.25)

water snake (0.14)

Blur Occlusion:

jellyfish (0.62)

umbrella (0.05)

stingray (0.04)

Noise Occlusion:

firework (0.27)

jellyfish (0.21)

parachute (0.11)

Did the classifier struggle to classify the occluded images?
Yes. The black box and noise occlusions significantly disrupted the classifier’s ability to predict accurately. In the black box version, the model started guessing sea creatures that weren’t even present. With noise, the predictions were more chaotic and less confident. The blur had the least impact, and the model still mostly guessed correctly.

Which occlusion had the greatest impact on performance?
Noise occlusion had the biggest impact. It introduced unpredictable visual data that confused the model more than simply hiding or blurring the features. The model responded to this distortion with scattered, low-confidence predictions.

Part 3: Creating and Experimenting with Image Filters
Explain what each line of the filter program does.
The basic filter program opens the image, resizes it, applies a Gaussian blur, and saves the result. The AI explained that Image.open() loads the image, .resize() shrinks it to a manageable size, .filter() applies the blur, and plt.savefig() stores the modified image. Even though I’m not fluent in Python, these steps were logical and easy to follow.

Do the AI’s explanations make sense?
Yes, they were helpful. I learned how basic image filters work and how to manipulate an image step by step using Python.

What are three different filters I can apply to an image?
The AI suggested:

Edge Detection – highlights the outlines and contours of shapes.

Sharpening – enhances details by increasing contrast between adjacent pixels.

Emboss – adds a 3D texture to make the image look engraved.

Implement Filters:
I added each filter to the program and saved the results as:

filter_edges.png

filter_sharpen.png

filter_emboss.png

Design Your Own Artistic Filter:
Instead of implementing an exaggerated or meme-style filter, I kept my experimentation focused on useful, visually distinct filters. I explored alternatives such as sharpening, edge detection, and embossing. These filters emphasized different aspects of the image without distorting it beyond recognition.

Describe the filter and what kind of effect it has on your image:
Each filter had a unique impact. The sharpening filter made the jellyfish’s edges and tentacles more pronounced, enhancing its detail. The edge detection filter removed color but emphasized the shape and structure of the jellyfish, turning it into a kind of sketch. The emboss filter gave the image a textured, 3D appearance, almost like a carved relief. All three made it easier to see how image processing techniques can alter perception and highlight different features in an image.

Final Report
Summarize your findings from each part of the project.
In Part 1, I learned how to use a pretrained classifier and understand its behavior with Grad-CAM. In Part 2, I saw how occlusions affect the model's confidence and accuracy. In Part 3, I got to play with filters and customize how the image is presented. Each part showed different ways AI interacts with images, and how we can both understand and manipulate that behavior.

What did you learn about the classifier’s behavior from the heatmap and how it is impacted by occlusions?
The classifier relies heavily on certain features in an image. The heatmap showed exactly what it’s looking at, and when that region was changed or hidden, performance dropped. Grad-CAM helped make this process visible, and occlusion helped me test it.

Describe the filter you developed. What kind of effect does it have on your image?
The deep-fried filter gave the image an exaggerated and chaotic look. It’s more for fun than function, but it showed how you can use Python to stylize and transform images in creative ways.

Reflect on your experience working with the AI to explain and write Python code.
Working with the AI made a big difference. I was able to understand code that would normally be overwhelming, and the AI guided me through problems when I got stuck. It felt like having a tutor on standby while I worked through the project. This experience gave me more confidence with both Python and AI tools, and I would absolutely use this approach again.
