# ColtCruz-Image-Classification-and-Processing

---

## **Part 1: Using the Basic Classifier and Implementing Grad-CAM**

### **Explain what each line of this Python program does**
I asked the AI to walk through the `base_classifier.py` program line by line. It broke down the imports, the image preprocessing steps, how the model predicts classes, and how results are displayed. Even though I’m not experienced in Python, the explanations were clear. I now understand how image data flows through the TensorFlow model and how it translates the output into labels with confidence scores. This helped me follow the code without needing to search online for each function.

### **If you do not have previous experience with Python, comment on whether the AI’s explanations make sense to you**
I don’t have much Python experience, but the AI made the explanations very accessible. The explanations were easy to follow and helped me understand how each line worked within the program. It made concepts like model loading and image array manipulation feel less intimidating.

### **Record the top-3 predictions and their confidence scores**
1. jellyfish (0.77)  
2. umbrella (0.03)  
3. volcano (0.02)

### **Use the base_classifier.py program to classify the image**
I ran the program using an image of a Crowned Jellyfish. The top prediction was jellyfish, which was correct. The model identified the subject with high confidence, showing how effective MobileNetV2 is even on unfamiliar images.

### **Implement Grad-CAM to visualize the areas the model focuses on**
I used AI assistance to implement Grad-CAM in the base classifier. The generated heatmap was saved as `gradcam_output.jpg`. It showed the classifier focused on the jellyfish’s central body, particularly areas with contrast against the dark background. This helped me visually understand how the model decides what features are most important.

### **Explain what Grad-CAM is and how it works**
Grad-CAM (Gradient-weighted Class Activation Mapping) creates a heatmap that shows which parts of an image are most important for the model's prediction. It does this by computing the gradients of the predicted class with respect to the last convolutional layer. These gradients are averaged and multiplied with the activation maps, then overlaid on the image. This visual explanation helped me understand how convolutional neural networks pay attention to details.

### **Identify which parts of the image the classifier focuses on most heavily. Record your observations**
The classifier focused on the top of the jellyfish where its crown shape was most visible. These areas were highlighted in red on the Grad-CAM heatmap, indicating strong relevance to the final prediction. The dark water background was mostly ignored, which shows how the model filters out irrelevant visual information.

---

## **Part 2: Experimenting with Image Occlusion**

### **What are three ways to occlude an image?**
The AI suggested:
- Black box occlusion: covering the important region with a solid black rectangle.  
- Blur occlusion: applying a strong Gaussian blur to obscure key features.  
- Noise occlusion: replacing the region with random visual noise to disrupt recognizable patterns.

### **Modify the base_classifier.py program to implement the three occlusions**
I modified the classifier to create three new versions of the image, each with one of the occlusion methods applied directly over the Grad-CAM area. The new outputs were:  
- `occlusion_black_box.jpg`  
- `occlusion_blur.jpg`  
- `occlusion_noise.jpg`

### **Run the classifier on each occluded image. Record the top-3 predictions and confidence scores**

**Black Box Occlusion**  
1. sea anemone (0.29)  
2. jellyfish (0.25)  
3. water snake (0.14)

**Blur Occlusion**  
1. jellyfish (0.62)  
2. umbrella (0.05)  
3. stingray (0.04)

**Noise Occlusion**  
1. firework (0.27)  
2. jellyfish (0.21)  
3. parachute (0.11)

### **Did the classifier struggle to classify the occluded images?**
Yes. The black box and noise occlusions significantly disrupted the classifier’s ability to predict accurately. In the black box version, the model started guessing sea creatures that weren’t even present. With noise, the predictions were more chaotic and less confident. The blur had the least impact, and the model still mostly guessed correctly.

### **Which occlusion had the greatest impact on performance?**
Noise occlusion had the biggest impact. It introduced unpredictable visual data that confused the model more than simply hiding or blurring the features. The model responded to this distortion with scattered, low-confidence predictions.

---

## **Part 3: Creating and Experimenting with Image Filters**

### **Explain what each line of the filter program does**
The basic filter program opens the image, resizes it, applies a Gaussian blur, and saves the result. The AI explained that `Image.open()` loads the image, `.resize()` shrinks it to a manageable size, `.filter()` applies the blur, and `plt.savefig()` stores the modified image. Even though I’m not fluent in Python, these steps were logical and easy to follow.

### **Do the AI’s explanations make sense?**
Yes, they were helpful. I learned how basic image filters work and how to manipulate an image step by step using Python.

### **What are three different filters I can apply to an image?**
The AI suggested:
- Edge Detection – highlights the outlines and contours of shapes.  
- Sharpening – enhances details by increasing contrast between adjacent pixels.  
- Emboss – adds a 3D texture to make the image look engraved.

### **Implement Filters**
I added each filter to the program and saved the results as:  
- `filter_edges.png`  
- `filter_sharpen.png`  
- `filter_emboss.png`

### **Design Your Own Artistic Filter**
Instead of implementing an exaggerated or meme-style filter, I kept my experimentation focused on useful, visually distinct filters. I explored alternatives such as sharpening, edge detection, and embossing. These filters emphasized different aspects of the image without distorting it beyond recognition.

### **Describe the filter and what kind of effect it has on your image**
Each filter had a unique impact. The sharpening filter made the jellyfish’s edges and tentacles more pronounced, enhancing its detail. The edge detection filter removed color but emphasized the shape and structure of the jellyfish, turning it into a kind of sketch. The emboss filter gave the image a textured, 3D appearance, almost like a carved relief. All three made it easier to see how image processing techniques can alter perception and highlight different features in an image.

---

## **Final Report**

### **Summarize your findings from each part of the project**
In Part 1, I learned how to use a pretrained classifier and understand its behavior with Grad-CAM. In Part 2, I saw how occlusions affect the model's confidence and accuracy. In Part 3, I got to play with filters and customize how the image is presented. Each part showed different ways AI interacts with images, and how we can both understand and manipulate that behavior.

### **What did you learn about the classifier’s behavior from the heatmap and how it is impacted by occlusions?**
The classifier relies heavily on certain features in an image. The heatmap showed exactly what it’s looking at, and when that region was changed or hidden, performance dropped. Grad-CAM helped make this process visible, and occlusion helped me test it.

### **Describe the filter you developed. What kind of effect does it have on your image?**
I didn’t go for an exaggerated filter. Instead, I experimented with three that each bring out a different feature in the image. Sharpening made edges crisp, emboss added depth, and edge detection gave it a line-art look. It helped me appreciate how filters can transform how we see an image.

### **Reflect on your experience working with the AI to explain and write Python code**
Working with the AI made a big difference. I was able to understand code that would normally be overwhelming, and the AI guided me through problems when I got stuck. It felt like having a tutor on standby while I worked through the project. This experience gave me more confidence with both Python and AI tools, and I would absolutely use this approach again.
