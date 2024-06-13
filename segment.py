import numpy as np
import torch
from segment_anything import SamPredictor, sam_model_registry
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import argparse
import logging as log


log.getLogger().setLevel(log.INFO)
sam = sam_model_registry["vit_h"](checkpoint="weights/sam_vit_h_4b8939.pth")
sam.to(device='cuda')
predictor = SamPredictor(sam)


def show_mask(mask, ax, mask_name, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([255, 255, 255, 1])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    cv2.imwrite(mask_name, mask_image)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))   


def parse_cloth(image_path):
    log.info('Parsing...')
    garment_locations = []
    # Load the image
    img = mpimg.imread(image_path)  # Replace 'your_image.jpg' with the path to your image file

    # Function to be called when a mouse click event occurs
    def onclick(event):
        x, y = event.xdata, event.ydata
        print(f'Coordinates: ({x:.2f}, {y:.2f})')
        garment_locations.append([round(x), round(y)])

        # Disconnect the event and close the figure
        if len(garment_locations) == 3:
            fig.canvas.mpl_disconnect(cid)
            plt.close(fig)

    # Create a figure and display the image
    fig, ax = plt.subplots()
    ax.imshow(img)

    # Connect the click event to the onclick function
    cid = fig.canvas.mpl_connect('button_press_event', onclick)

    # Show the image with the interactive plot
    plt.show()

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image)

    # print(garment_locations)
    # input_point = np.array([garment_locations[0]])
    # input_label = np.array([1])
    input_point = np.array([garment_locations[0], garment_locations[1], garment_locations[2]])
    input_label = np.array([1, 1, 1])


    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )

    mask_input = logits[np.argmax(scores), :, :]

    masks, _, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        mask_input=mask_input[None, :, :],
        multimask_output=False,
    )

    log.info('Generating masks...')
    mask_path = image_path.split('.')[0] + '_mask.jpg'
    plt.figure(figsize=(10,10))
    plt.imshow(image)
    show_mask(masks, plt.gca(), mask_path, random_color=False)
    show_points(input_point, input_label, plt.gca())
    plt.axis('off')
    plt.show()
    log.info('Done.')
    log.info('Mask saved: ' + mask_path)

    # plt.figure(figsize=(10,10))
    # plt.imshow(image)
    # show_mask(masks[0], plt.gca(), random_color=True)
    # show_points(input_point, input_label, plt.gca())
    # plt.title(f"Mask 0, Score: {scores[0]:.3f}", fontsize=18)
    # plt.axis('off')
    # plt.show()


def main():
    # Set up the argument parser
    parser = argparse.ArgumentParser(description='Segment images to generate clothing mask')
    parser.add_argument('--input', type=str, required=True, help='Input image')

    # Parse the arguments
    args, unknown = parser.parse_known_args()
    parse_cloth(args.input)
    # parse_cloth('reference_images/crop_top.png')
    # parse_cloth('reference_images/dress.png')
    # parse_cloth('reference_images/jumpsuit.png')


if __name__ == "__main__":
    main()
