import cv2
import numpy as np

# Load the image
img = cv2.imread('lena.tif', cv2.IMREAD_COLOR)
cv2.imshow('Lena', img)

while True:
    key = cv2.waitKey(0) & 0xFF

    # Simple Averaging (Average all color channels)
    if key == ord('1'):
        cv2.destroyAllWindows()
        cv2.imshow('Lena', img)

        def calculate_simple_averaging(image):
            # Simple average across all color channels
            return (image[:, :, 2] / 3 + image[:, :, 1] / 3 + image[:, :, 0] / 3).astype(np.uint8)

        simple_averaging = calculate_simple_averaging(img)
        cv2.imshow("simple_averaging", simple_averaging)
        cv2.imwrite('simple_averaging.jpg', simple_averaging)

    # Weighted Average (Luminance formula for grayscale conversion)
    if key == ord('2'):
        cv2.destroyAllWindows()
        cv2.imshow('Lena', img)

        def calculate_weighted_average(image):
            # Apply luminance formula for weighted average (grayscale)
            return (image[:, :, 2] * 0.299 + image[:, :, 1] * 0.587 + image[:, :, 0] * 0.114).astype(np.uint8)

        weighted_average = calculate_weighted_average(img)
        cv2.imshow("weighted_average", weighted_average)
        cv2.imwrite('weighted_average.jpg', weighted_average)

    # Desaturation (Average max and min values of each pixel)
    if key == ord('3'):
        cv2.destroyAllWindows()
        cv2.imshow('Lena', img)

        def calculate_desaturation(image):
            # Desaturation method, average the max and min values
            max_channel = np.maximum.reduce([image[:, :, 0], image[:, :, 1], image[:, :, 2]])
            min_channel = np.minimum.reduce([image[:, :, 0], image[:, :, 1], image[:, :, 2]])
            return (max_channel / 2 + min_channel / 2).astype(np.uint8)

        desaturation = calculate_desaturation(img)
        cv2.imshow("desaturation", desaturation)
        cv2.imwrite('desaturation.jpg', desaturation)

    # Decomposition (Max and Min channel values)
    if key == ord('4'):
        cv2.destroyAllWindows()
        cv2.imshow('Lena', img)

        def calculate_decomposition_max(image):
            # Max of the three channels
            return np.maximum(np.maximum(image[:, :, 2], image[:, :, 1]), image[:, :, 0])

        decomposition_max = calculate_decomposition_max(img)
        cv2.imshow("decomposition_max", decomposition_max)
        cv2.imwrite('decomposition_max.jpg', decomposition_max)

        def calculate_decomposition_min(image):
            # Min of the three channels
            return np.minimum(np.minimum(image[:, :, 2], image[:, :, 1]), image[:, :, 0])

        decomposition_min = calculate_decomposition_min(img)
        cv2.imshow("decomposition_min", decomposition_min)
        cv2.imwrite('decomposition_min.jpg', decomposition_min)

    # Single Channel Extraction (Red, Green, Blue)
    if key == ord('5'):
        cv2.destroyAllWindows()
        cv2.imshow('Lena', img)

        def calculate_single_channel(image, n):
            # Extract the specified channel (0 = Blue, 1 = Green, 2 = Red)
            return image[:, :, n]

        single_channel = calculate_single_channel(img, 0)
        cv2.imshow("single_channel_1", single_channel)  # 0 - blue, 1 - green, 2 - red

        single_channel2 = calculate_single_channel(img, 1)
        cv2.imshow("single_channel_2", single_channel2)

        single_channel3 = calculate_single_channel(img, 2)
        cv2.imshow("single_channel_3", single_channel3)

    # Custom Gray Shades (Custom grayscale with a specific number of shades)
    if key == ord('6'):
        cv2.destroyAllWindows()
        cv2.imshow('Lena', img)

        def custom_gray_shades(image, n_shades):
            # Custom grayscale with specified shades
            if n_shades < 1 or n_shades > 255:
                print("Error - n is not in interval [1,255]")
                return image

            gray_image = (image[:, :, 2] * 0.299 + image[:, :, 1] * 0.587 + image[:, :, 0] * 0.114).astype(np.uint8)
            interval_size = 256 // n_shades
            custom_image = np.zeros_like(gray_image)

            for i in range(n_shades):
                # Apply custom shades by averaging pixels within specific intervals
                true_if_interval = (gray_image >= i * interval_size) & (gray_image <= ((i + 1) * interval_size - 1))
                if np.any(true_if_interval):
                    avg_value = np.mean(gray_image[true_if_interval]).astype(np.uint8)
                    custom_image[true_if_interval] = avg_value

            return custom_image

        n_shades = 4
        custom_shaded_image = custom_gray_shades(img, n_shades)
        cv2.imshow("custom_gray_shades", custom_shaded_image)

    # Floyd-Steinberg Dithering (Error diffusion dithering)
    if key == ord('7'):
        cv2.destroyAllWindows()
        cv2.imshow('Lena', img)

        def nearest_color(old_pixel):
            # Round pixel values to nearest color (either black or white)
            return np.round(old_pixel / 255) * 255

        def floyd_steinberg_dithering(image):
            # Floyd-Steinberg dithering algorithm
            gray_image = (image[:, :, 2] * 0.299 + image[:, :, 1] * 0.587 + image[:, :, 0] * 0.114).astype(np.uint8)
            image = gray_image.astype(np.float32)
            height, width = image.shape[:2]
            for y in range(height):
                for x in range(width):
                    old_pixel = image[y, x]
                    new_pixel = nearest_color(old_pixel)
                    image[y, x] = new_pixel
                    quant_error = old_pixel - new_pixel
                    if x < width - 1:
                        image[y, x + 1] += quant_error * 7 / 16
                    if y < height - 1:
                        if x > 0:
                            image[y + 1, x - 1] += quant_error * 3 / 16
                        image[y + 1, x] += quant_error * 5 / 16
                        if x < width - 1:
                            image[y + 1, x + 1] += quant_error * 1 / 16
            return image.astype(np.uint8)

        floyd_steinberg_dithered = floyd_steinberg_dithering(img)
        cv2.imshow("floyd_steinberg_dithered", floyd_steinberg_dithered)
        cv2.imwrite('floyd_steinberg_dithered.jpg', floyd_steinberg_dithered)

        # Stucki Dithering (Another dithering method)
        def stucki_dithering(image):
            # Stucki dithering algorithm
            gray_image = (image[:, :, 2] * 0.299 + image[:, :, 1] * 0.587 + image[:, :, 0] * 0.114).astype(np.uint8)
            image = gray_image.astype(np.float32)
            height, width = image.shape[:2]
            for y in range(height):
                for x in range(width):
                    old_pixel = image[y, x]
                    new_pixel = nearest_color(old_pixel)
                    image[y, x] = new_pixel
                    quant_error = old_pixel - new_pixel
                    if x < width - 1:
                        image[y, x + 1] += quant_error * 8 / 42
                    if x < width - 2:
                        image[y, x + 2] += quant_error * 4 / 42
                    if y < height - 1:
                        if x > 1:
                            image[y + 1, x - 2] += quant_error * 2 / 42
                        if x > 0:
                            image[y + 1, x - 1] += quant_error * 4 / 42
                        image[y + 1, x] += quant_error * 8 / 42
                        if x < width - 1:
                            image[y + 1, x + 1] += quant_error * 4 / 42
                        if x < width - 2:
                            image[y + 1, x + 2] += quant_error * 2 / 42
                    if y < height - 2:
                        if x > 1:
                            image[y + 2, x - 2] += quant_error * 1 / 42
                        if x > 0:
                            image[y + 2, x - 1] += quant_error * 2 / 42
                        image[y + 2, x] += quant_error * 4 / 42
                        if x < width - 1:
                            image[y + 2, x + 1] += quant_error * 2 / 42
                        if x < width - 2:
                            image[y + 2, x + 2] += quant_error * 1 / 42
            return image.astype(np.uint8)

        stucki_dithered = stucki_dithering(img)
        cv2.imshow("stucki_dithered", stucki_dithered)
        cv2.imwrite('stucki_dithered.jpg', stucki_dithered)

    elif key == ord('q'):
        break  # Exit the loop if 'q' is pressed

    elif key == ord('d'):
        cv2.destroyAllWindows()
        cv2.imshow('Lena', img)

cv2.destroyAllWindows()  # Close all windows
