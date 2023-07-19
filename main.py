import os
import cv2
from paddleocr import PPStructure
import random
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from paddleocr.tools.infer.utility import draw_ocr_box_txt
from paddleocr.ppstructure.predict_system import save_structure_res
import math


def create_font(txt, sz, font_path="./doc/fonts/simfang.ttf"):
    font_size = int(sz[1] * 0.99)
    font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
    # length = font.getsize(txt)[0]
    # if length > sz[0]:
    #     font_size = 10  # int(font_size * sz[0] / length)
    #     font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
    return font


def draw_box_txt_fine(img_size, box, txt, font_path="./doc/fonts/simfang.ttf"):
    box_height = int(
        math.sqrt((box[0][0] - box[3][0]) ** 2 + (box[0][1] - box[3][1]) ** 2)
    )
    box_width = int(
        math.sqrt((box[0][0] - box[1][0]) ** 2 + (box[0][1] - box[1][1]) ** 2)
    )

    if box_height > 2 * box_width and box_height > 30:
        img_text = Image.new("RGB", (box_height, box_width), (255, 255, 255))
        draw_text = ImageDraw.Draw(img_text)
        if txt:
            font = create_font(txt, (box_height, box_width), font_path)
            draw_text.text([0, 0], txt, fill=(0, 0, 0), font=font)
        img_text = img_text.transpose(Image.ROTATE_270)
    else:
        img_text = Image.new("RGB", (box_width, box_height), (255, 255, 255))
        draw_text = ImageDraw.Draw(img_text)
        if txt:
            font = create_font(txt, (box_width, box_height), font_path)
            draw_text.text([0, 0], txt, fill=(0, 0, 0), font=font)

    pts1 = np.float32(
        [[0, 0], [box_width, 0], [box_width, box_height], [0, box_height]]
    )
    pts2 = np.array(box, dtype=np.float32)
    M = cv2.getPerspectiveTransform(pts1, pts2)

    img_text = np.array(img_text, dtype=np.uint8)
    img_right_text = cv2.warpPerspective(
        img_text,
        M,
        img_size,
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    return img_right_text


def draw_ocr_box_txt(
    image,
    boxes,
    txts=None,
    scores=None,
    drop_score=0.5,
    font_path="./doc/fonts/simfang.ttf",
):
    h, w = image.height, image.width
    img_left = image.copy()
    img_right = np.ones((h, w, 3), dtype=np.uint8) * 255
    random.seed(0)

    draw_left = ImageDraw.Draw(img_left)
    if txts is None or len(txts) != len(boxes):
        txts = [None] * len(boxes)
    for idx, (box, txt) in enumerate(zip(boxes, txts)):
        if scores is not None and scores[idx] < drop_score:
            continue
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        draw_left.polygon(box, fill=color)
        img_right_text = draw_box_txt_fine((w, h), box, txt, font_path)
        pts = np.array(box, np.int32).reshape((-1, 1, 2))
        cv2.polylines(img_right_text, [pts], True, color, 1)
        img_right = cv2.bitwise_and(img_right, img_right_text)
    img_left = Image.blend(image, img_left, 0.5)
    img_show = Image.new("RGB", (w * 2, h), (255, 255, 255))
    img_show.paste(img_left, (0, 0, w, h))
    img_show.paste(Image.fromarray(img_right), (w, 0, w * 2, h))
    return np.array(img_show)


def draw_structure_result(image, result, font_path):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    boxes, txts, scores = [], [], []

    img_layout = image.copy()
    draw_layout = ImageDraw.Draw(img_layout)
    text_color = (255, 255, 255)
    text_background_color = (80, 127, 255)
    catid2color = {}
    font_size = 15
    font = ImageFont.truetype(font_path, font_size, encoding="utf-8")

    for region in result:
        if region["type"] not in catid2color:
            box_color = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255),
            )
            catid2color[region["type"]] = box_color
        else:
            box_color = catid2color[region["type"]]
        box_layout = region["bbox"]
        draw_layout.rectangle(
            [(box_layout[0], box_layout[1]), (box_layout[2], box_layout[3])],
            outline=box_color,
            width=3,
        )
        # text_w, text_h = font.getsize(region["type"])
        # draw_layout.rectangle(
        #     [
        #         (box_layout[0], box_layout[1]),
        #         (box_layout[0] + text_w, box_layout[1] + text_h),
        #     ],
        #     fill=text_background_color,
        # )
        draw_layout.text(
            (box_layout[0], box_layout[1]), region["type"], fill=text_color, font=font
        )

        if region["type"] == "table":
            pass
        else:
            for text_result in region["res"]:
                boxes.append(np.array(text_result["text_region"]))
                txts.append(text_result["text"])
                scores.append(text_result["confidence"])

    im_show = draw_ocr_box_txt(
        img_layout, boxes, txts, scores, font_path=font_path, drop_score=0
    )
    return im_show


table_engine = PPStructure(show_log=False, image_orientation=False)

save_folder = "./output"
img_path = "input/5.png"
img = cv2.imread(img_path)
result = table_engine(img)
save_structure_res(result, save_folder, os.path.basename(img_path).split(".")[0])

for line in result:
    line.pop("img")
    print(line)

from PIL import Image

font_path = "doc/fonts/simfang.ttf"  # PaddleOCR下提供字体包
image = Image.open(img_path).convert("RGB")
im_show = draw_structure_result(image, result, font_path=font_path)
im_show = Image.fromarray(im_show)
im_show.save(
    os.path.join(save_folder, os.path.basename(img_path).split(".")[0], "result.jpg")
)
