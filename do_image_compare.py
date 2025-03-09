from skimage.metrics import structural_similarity as compare_ssim
import imutils
import logging
import numpy as np
import cv2
print(cv2.__version__)  # 应输出类似 4.9.0

class ImageCompare():
    def __init__(self):
        pass

    # def match(self, ui_case):
    #     baseline_img = ui_case.baseline_image
    #     width, height, _ = baseline_img.shape
    #
    #     target_img = ui_case.test_image
    #     target_width, target_height, _ = target_img.shape
    #
    #     # target_img should be same size with baseline_img
    #     if width != target_width or height != target_height:
    #         target_img = cv2.resize(target_img, (height, width), interpolation=cv2.INTER_CUBIC)
    #
    #     ui_case.compare_result, ui_case.baseline_image_result, ui_case.test_image_result = self.do_full_pic_match(
    #         baseline_img, target_img)
    #     return ui_case

    def do_full_pic_match(self, baseline_img, target_img):
        """
        :param baseline_img:
        :param target_img
        :return:
        """
        # 灰度转换
        grayA = cv2.cvtColor(baseline_img, cv2.COLOR_BGR2GRAY)
        grayB = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)

        # 进行对比前，首先进行高斯降噪
        grayA_gauss = cv2.GaussianBlur(grayA, (5, 5), 0)
        grayB_gauss = cv2.GaussianBlur(grayB, (5, 5), 0)

        # compute the Structural Similarity Index (SSIM) between the two
        # images, ensuring that the difference image is returned
        (score, diff) = compare_ssim(grayA_gauss, grayB_gauss, full=True)

        diff = (diff * 255).astype("uint8")
        logging.info("SSIM: {}".format(score))

        # obtain the regions of the two input images that differ
        # 图像二值化
        thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

        # 图像形态学去除噪点
        kernel = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(5, 5))
        thresh = cv2.morphologyEx(src=thresh, op=cv2.MORPH_CLOSE, kernel=kernel)
        thresh = cv2.morphologyEx(src=thresh, op=cv2.MORPH_OPEN, kernel=kernel)

        # 寻找轮廓
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        # 绘制结果图片
        result_image_baseline = baseline_img.copy()
        result_image_target = target_img.copy()
        for c in cnts:
            # compute the bounding box of the contour and then draw the
            # bounding box on both input images to represent where the two
            # images differ
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(result_image_baseline, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.rectangle(result_image_target, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # 如果能找到轮廓则说明两个图片有不一样的地方，对比失败
        return len(cnts) == 0, result_image_baseline, result_image_target

    def draw_ui_case(slef, baseline_image_result, test_image_result, timeout=100000):
        """
        展示结果图片
        :param ui_case: 用于展示的case
        :param timeout: 展示图片的默认时间，过去后图片自动关闭
        :return:
        """
        cv2.imshow("original result", baseline_image_result)
        cv2.imshow("current result", test_image_result)
        cv2.waitKey(timeout)


if __name__ == '__main__':
    compare = ImageCompare()
    baseline_img = cv2.imread('./demo_img/searchEngine-align.jpg')
    target_img = cv2.imread('./demo_img/searchEngine-align-wrong.jpg')

    # baseline_img = ui_case.baseline_image
    # width, height, _ = baseline_img.shape
    #
    # target_img = ui_case.test_image
    # target_width, target_height, _ = target_img.shape
    #
    # # target_img should be same size with baseline_img
    # if width != target_width or height != target_height:
    #     target_img = cv2.resize(target_img, (height, width), interpolation=cv2.INTER_CUBIC)

    result, result_image_baseline, result_image_target = compare.do_full_pic_match(baseline_img, target_img)
    print(result)

    compare.draw_ui_case(result_image_baseline, result_image_target)
