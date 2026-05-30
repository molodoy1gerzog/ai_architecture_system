from PIL import Image


def validate_image(file_path):

    try:
        image = Image.open(file_path)
        image.verify()

        return True
    
    except Exception:
        return False