import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from keras.applications import MobileNetV2
from keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import cv2 

model = MobileNetV2(weights="imagenet")

def classify_image(image_path):
    """Classify an image and display the predictions."""
    try:
        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)
        decoded_predictions = decode_predictions(predictions, top=3)[0]

        print("Top-3 Predictions:")
        for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
            print(f"{i + 1}: {label} ({score:.2f})")

        # -------- Grad-CAM START --------
        grad_model = tf.keras.models.Model(
            [model.inputs], [model.get_layer("Conv_1").output, model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            top_pred_index = tf.argmax(predictions[0])
            top_class_channel = predictions[:, top_pred_index]

        grads = tape.gradient(top_class_channel, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

        img = cv2.imread(image_path)
        img = cv2.resize(img, (224, 224))
        heatmap = cv2.resize(heatmap.numpy(), (img.shape[1], img.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
        output_path = "gradcam_output.jpg"
        cv2.imwrite(output_path, superimposed_img)
        print(f"\nGrad-CAM heatmap saved as {output_path}")

        # -------- Grad-CAM END --------

    # -------- Occlusion Variants START --------
        # Reuse the heatmap to target occlusion area
        mask = heatmap > 150  # high-importance areas
        mask = mask.astype(np.uint8) 

        # 1. Black Box Occlusion
        black_box = img.copy()
        black_box[mask == 1] = 0
        cv2.imwrite("occlusion_black_box.jpg", black_box)

        # 2. Blur Occlusion
        blur_occluded = img.copy()
        blur = cv2.GaussianBlur(blur_occluded, (11, 11), 0)
        blur_occluded[mask == 1] = blur[mask == 1]
        cv2.imwrite("occlusion_blur.jpg", blur_occluded)

        # 3. Noise Occlusion
        noise_occluded = img.copy()
        noise = np.random.randint(0, 256, img.shape, dtype=np.uint8)
        noise_occluded[mask == 1] = noise[mask == 1] 
        cv2.imwrite("occlusion_noise.jpg", noise_occluded)
        print("Occluded images saved: black box, blur, noise.")
        # -------- Occlusion Variants END --------

    except Exception as e:
        print(f"Error processing image: {e}")


if __name__ == "__main__":
    image_path = "Crowned_Jellyfish.jpg"  
    classify_image(image_path)