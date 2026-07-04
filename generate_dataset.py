import os
import random
import string
from captcha.image import ImageCaptcha

CHARSET = string.ascii_uppercase + string.digits
CAPTCHA_LENGTH = 4
NUM_SAMPLES = 3000
OUTPUT_DIR = "dataset"


def generate():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    image_gen = ImageCaptcha(width=160, height=60)
    labels = []

    print(f"Generating {NUM_SAMPLES} synthetic CAPTCHA images...")
    for i in range(NUM_SAMPLES):
        text = ''.join(random.choices(CHARSET, k=CAPTCHA_LENGTH))
        filename = f"{i:05d}_{text}.png"
        path = os.path.join(OUTPUT_DIR, filename)
        image_gen.write(text, path)
        labels.append((filename, text))
        if (i + 1) % 500 == 0:
            print(f"  Generated {i + 1}/{NUM_SAMPLES}")

    with open(os.path.join(OUTPUT_DIR, "labels.csv"), "w") as f:
        f.write("filename,label\n")
        for fn, lab in labels:
            f.write(f"{fn},{lab}\n")

    print("Dataset generation complete.")
    print(f"Images saved in: {OUTPUT_DIR}/")
    print(f"Labels saved in: {OUTPUT_DIR}/labels.csv")


if __name__ == "__main__":
    generate()
