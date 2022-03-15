import os
import cv2
import numpy as np

# brightness gradient: high to low
gradient = 'N@#W&9876543210?!abc;:+=-,_'
# pixel gradient: low to high
pixel_grad = np.linspace(0, 255, len(gradient))


def find_value(val):
    for i in range(0, len(pixel_grad)-1):
        if pixel_grad[i] <= val <= pixel_grad[i+1]:
            return gradient[i]


def generate_ascii_image(image_path: str, final_size: int = 100):
    if not os.path.exists(image_path):
        raise Exception(f'Path {image_path} doesn\'nt exist!')
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ar = gray.shape[1] / gray.shape[0]
    ratio = final_size / gray.shape[1]
    final_shape = (int(gray.shape[0] * ar * ratio), int(gray.shape[1] / ar * ratio))
    gray = cv2.resize(gray, final_shape)

    image = []
    for i in range(gray.shape[1]):
        row = []
        for j in range(gray.shape[0]):
            val = find_value(gray[j, i])
            row.append(val)
        image.append(row)

    img = np.array(image, dtype="str")
    # print_image(img)
    return img


def write_image(img, file_path: str):
    with open(file_path, 'w') as f:
        for i in range(img.shape[1]):
            f.write(" ".join(img[:, i]))
            f.write("\n")


def print_image(img):
    for i in range(img.shape[1]):
        for j in range(img.shape[0]):
            print(img[j, i], end=" ")
        print()


def print_to_image(img, file_path: str, resolution: int = 100):
    font_scale = 0.5
    thickness = 2
    x, y = 10, 10
    (wd, ht), bs = cv2.getTextSize(text="N", fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, thickness=thickness)
    ht *= 1.5
    wd *= 1.5
    grid = int(ht) if ht > wd else int(wd)
    shape = img.shape
    ar = shape[0] / shape[1]
    w = int(grid * resolution)
    h = int(grid * resolution * ar)
    print(wd, ht)
    print(img.shape)
    print(w, h)
    canvas = np.ones((w, h, 3), dtype=np.uint8) * 255

    for i in range(img.shape[1]):
        x = 10
        for val in list(img[:, i]):
            cv2.putText(canvas, val, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness=thickness)
            x += grid
        y += grid
    cv2.imwrite(filename=file_path, img=canvas)


if __name__ == '__main__':
    img = generate_ascii_image("./dog.jpg", final_size=120)
    write_image(img, "image.txt")
    print_to_image(img, "output.jpg", resolution=120)

