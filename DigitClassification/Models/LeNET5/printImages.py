import matplotlib.pyplot as plt
import random


#
# Helper function to show a list of images with their relating titles
#
def showImagesLables(images, imageLable):
    cols = 5
    rows = int(len(images) / cols) + 1
    plt.figure(figsize=(30, 20))
    index = 1
    for x in zip(images, imageLable):
        image = x[0]
        title_text = x[1]
        plt.subplot(rows, cols, index)
        plt.imshow(image, cmap=plt.cm.gray)
        plt.axis("off")
        if title_text != "":
            plt.title(title_text, fontsize=15)
        index += 1

    plt.show()  # Show plot without blocking


def showImages(images):
    cols = 5
    rows = int(len(images) / cols) + 1
    plt.figure(figsize=(30, 20))
    index = 1
    for x in images:
        image = x
        plt.subplot(rows, cols, index)
        plt.imshow(image, cmap=plt.cm.gray)
        plt.axis("off")
        index += 1

    plt.show()  # Show plot without blocking


def showRandomImage(images, imageLables: None, num=10):
    ch = 1
    while ch != 0 :
        images_2_show = []
        titles_2_show = []
        for i in range(0, num):
            r = random.randint(1, len(images)-1)
            images_2_show.append(images[r])
            if imageLables is not None:
                titles_2_show.append(f"{r} : {imageLables[r]}")

        if imageLables is None:
            showImages(images_2_show)
        else:
            showImagesLables(images_2_show, titles_2_show)
        
        ch = int(input("If you want to see more Images press any Number else press 0 : "))
