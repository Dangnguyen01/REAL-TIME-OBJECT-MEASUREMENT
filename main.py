import cv2
import numpy as np
import imutils
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
from scipy.spatial import distance as dist


def read_and_preprocess(
    filename, canny_low=50, canny_high=100, blur_kernel=9, d_e_kernel=3
):
    # Đọc file ảnh
    img = cv2.imread(filename)
    cv2.imshow("image", img)

    # Chuyển thành ảnh xám
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Làm mờ ảnh
    grayBlur = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)

    # Áp dụng Canny tìm cạnh
    edged = cv2.Canny(grayBlur, canny_low, canny_high)
    edged = cv2.dilate(edged, (d_e_kernel, d_e_kernel), iterations=1)
    edged = cv2.erode(edged, (d_e_kernel, d_e_kernel), iterations=1)

    return img, edged


filename = "input2.JPG"
img, edged = read_and_preprocess(filename)
cv2.imshow("edged", edged)
cv2.waitKey()

def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) / 2, (ptA[1] + ptB[1]) / 2)

def get_distance_in_pixels(orig, c):

    # Lấy minRect
    box = cv2.minAreaRect(c)

    # Lấy tọa độ các đỉnh của MinRect
    box = cv2.boxPoints(box)
    box = np.array(box, dtype='int')

    # Sắp xếp các đỉnh theo trình tự
    cv2.drawContours(orig, [box.astype('int')], -1, (0, 255, 0), 2)

    # Tính toán 4 trung điểm của các cạnh
    (tl, tr, br, bl) = box
    (tltrX, tltrY) = midpoint(tl, tr)
    (blbrX, blbrY) = midpoint(bl, br)
    (tlblX, tlblY) = midpoint(tl, bl)
    (trbrX, trbrY) = midpoint(tr, br)

    # Tính độ dài 2 chiều
    dc_W = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
    dc_H = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

    return dc_W, dc_H, tltrX, tltrY, trbrX, trbrY


def find_object_in_pixel(orig, edge, area_threshold=3000):
    # Tìm các Contour trong ảnh
    contours = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    # Sắp xếp các contours từ trái qua phải
    # Sắp xếp các contours dựa trên tiêu chí cụ thể (ví dụ: diện tích)
    contours = sorted(contours, key=cv2.contourArea, reverse=False)
    P = None


    # Duyệt các Contours
    for c in contours:
        # Nếu contours quá nhỏ -> bỏ qua
        if cv2.contourArea(c) < area_threshold:
            continue

        # Tính toán 2 chiếu bằng Pixel
        dc_W, dc_H, tltrX, tltrY, trbrX, trbrY = get_distance_in_pixels(orig, c)

        # Nếu là đồng xu
        if P is None:
            # Cập nhật số P
            P = 20 / dc_H
            # Gán luôn kích thước thật bằng số đã biết
            dr_W = 20
            dr_H = 20
        else: # Nếu là các vật khác
            # Tính toán kích thước thật dựa vào kích thước pixel và số P
            dr_W = dc_W * P
            dr_H = dc_H * P

        # Ve kich thuoc len hinh
        cv2.putText(orig, "{:.1f} mm".format(dr_H), (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 255), 2)
        cv2.putText(orig, "{:.1f} mm".format(dr_W), (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 255), 2)

    return orig


image = find_object_in_pixel(img, edged)
cv2.imshow("A", img)
cv2.waitKey()
cv2.destroyAllWindows()