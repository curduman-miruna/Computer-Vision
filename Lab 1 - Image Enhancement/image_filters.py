import cv2
import numpy as np

img = cv2.imread('lena.tif', cv2.IMREAD_COLOR)
print('Size of image:', img.shape)

cv2.imshow('Lena', img)
cv2.imwrite('lena.jpg', img)

while True:
    key = cv2.waitKey(0) & 0xFF
    if key == ord('2'):
        cv2.destroyAllWindows()
        cv2.imshow('Lena', img)

        blurred_img = cv2.GaussianBlur(img, (5, 5), 0)
        cv2.imshow('Blurred Lena', blurred_img)
        cv2.imwrite('blurred_lena.jpg', blurred_img)

        blurred_img2 = cv2.GaussianBlur(img, (9, 9), 0)
        cv2.imshow('Blurred Lena 2', blurred_img2)
        cv2.imwrite('blurred_lena2.jpg', blurred_img2)

    # Sharpen the image
    elif key == ord('3'):
        cv2.destroyAllWindows()
        cv2.imshow('Lena', img)

        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharpened_img = cv2.filter2D(img, -1, kernel)
        cv2.imshow('Sharpened Lena', sharpened_img)
        cv2.imwrite('sharpened_lena.jpg', sharpened_img)

        sharpened_img2 = cv2.filter2D(img, -1, kernel * 2)
        cv2.imshow('Sharpened Lena 2', sharpened_img2)
        cv2.imwrite('sharpened_lena2.jpg', sharpened_img2)

    # Apply a custom filter
    elif key == ord('4'):
        cv2.destroyAllWindows()
        cv2.imshow('Lena', img)
        kernel2 = np.array([[0, -2, 0], [-2, 8, -2], [0, -2, 0]]) #Laplacian kernel
        filtered_img = cv2.filter2D(img, -1, kernel2)
        cv2.imshow('Filtered Lena', filtered_img)
        cv2.imwrite('filtered_lena.jpg', filtered_img)

    # Rotate the image
    elif key == ord('5'):
        cv2.destroyAllWindows()
        cv2.imshow('Lena', img)
        rotated_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        cv2.imshow('Rotated Lena 90-clock', rotated_img)
        cv2.imwrite('rotated_lena_90_clock.jpg', rotated_img)

        rotated_img2 = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        cv2.imshow('Rotated Lena 90-count', rotated_img2)
        cv2.imwrite('rotated_lena_90_count.jpg', rotated_img2)

        rotated_img3 = cv2.rotate(img, cv2.ROTATE_180)
        cv2.imshow('Rotated Lena 180', rotated_img3)
        cv2.imwrite('rotated_lena_180.jpg', rotated_img3)

        rows, cols = img.shape[:2]
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 60, 1)
        rotated_img4 = cv2.warpAffine(img, M, (cols, rows))
        cv2.imshow('Rotated Lena 60-clock', rotated_img4)
        cv2.imwrite('rotated_lena_60_clock.jpg', rotated_img4)

    # Crop the image
    elif key == ord('6'):
        cv2.destroyAllWindows()
        cv2.imshow('Lena', img)
        def crop_image(image, x, y, width, height):
            if x<0 or y<0 or x+width>image.shape[1] or y+height>image.shape[0]:
                print("Coordinates out of the image")
            return image[y:y + height, x:x + width]

        cropped_img = crop_image(img, 50, 50, 200, 200)
        cv2.imshow('Cropped Lena', cropped_img)
        cv2.imwrite('cropped_lena.jpg', cropped_img)

    # Exit on 'q'
    elif key == ord('q'):
        break

    # Destroy all windows on 'd'
    elif key == ord('d'):
        cv2.destroyAllWindows()
        cv2.imshow('Lena', img)

cv2.destroyAllWindows()
