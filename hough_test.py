import cv2
import numpy as np

def find_biggest_lines(image_path, num_lines=3):
    # Read the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply Canny edge detection
    edges = cv2.Canny(image, 50, 150, apertureSize=3)

    # Find lines using Hough transform
    lines = cv2.HoughLines(edges, 1, np.pi/180, 200)

    if lines is not None:
        lines = lines[:, 0, :]  # reshape to (n, 2), drop third dimension

        # Sort lines by their lengths
        lines = sorted(lines, key=lambda line: np.linalg.norm(line))

        # Get the biggest three lines
        biggest_lines = lines[-num_lines:]

        # Draw the lines on the original image
        image_with_lines = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        for rho, theta in biggest_lines:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(image_with_lines, (x1, y1), (x2, y2), (0, 0, 255), 2)

        return image_with_lines
    else:
        return None

# Example usage
image_path = "example_image.jpg"
biggest_lines_image = find_biggest_lines(image_path)
if biggest_lines_image is not None:
    cv2.imshow("Biggest Lines", biggest_lines_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No lines found in the image.")
