import PIL
import PIL.Image

from flower_model import classify, get_training_flowers

"""[recognition of flower type with DL model a]
"""
def get_flower_classification(picture):
    print('Here is some DL happening.')
    answer = classify(picture)
    return answer

if __name__ == '__main__':
    path = 'static/images/sunflower_14.jpeg'
    example_flower = PIL.Image.open(path)
    print(get_flower_classification(example_flower))