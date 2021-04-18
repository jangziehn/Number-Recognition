import os
import sys
from random import *
import cv2
import json
import numpy as np

present_window = False
mode = 0

input_file = sys.argv[1]

if input_file is None:
    print("Invalid argument")
    sys.exit()

folders = ["견고딕", "굴림", "HY신명조"]

w = 72
lines = []

for i in range(2, w+1):
    if w % i == 0:
        lines.append(i)

lines = lines[0:10]


def getRectContour(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 경계값 적용 cv.adaptiveThreshold(이미지, 최대값, 경계값 계산 알고리즘, 경계화 타입, 블록 크기(홀수), 보정상수)
    thr = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 1)

    # 외곽선 찾기
    contours, hierarchy = cv2.findContours(
        thr,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # 외곽선 위치
    x, y, w, h = cv2.boundingRect(contours[0])

    return {
        "x": x,
        "y": y,
        "w": w,
        "h": h,
        "from": (x, y),
        "to": (x+w, y+h),
        "contours": contours
    }


def loadImage(font_name, i):
    img_path = os.path.join(os.getcwd(), 'fonts', font_name, str(i) + '.png')

    img_array = np.fromfile(img_path, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    return img


def load_image_by_path(path):
    img_path = path

    img_array = np.fromfile(img_path, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    return img


def analyzeImage(img, line_cnt, show_window=False):

    rect = getRectContour(img)  # 외곽선 위치 값 찾기

    cropped = img[rect["from"][1]:rect["to"][1],
                  rect["from"][0]:rect["to"][0]]  # 외곽선 위치로 이미지 자르기

    # 자른 이미지 w 값 크기로 리사이징
    canvas = cv2.resize(cropped, (w, w), interpolation=cv2.INTER_AREA)
    canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)  # 탐색할 이미지는 GRAYSCALE로 변환

    # 이미지 리사이징 하면서 색이 흐려지는 픽셀이 생기는데 Threshold 적용해서 B&W 이미지로 변환
    (_, canvas) = cv2.threshold(canvas, 127, 255, cv2.THRESH_BINARY)

    preview = canvas.copy()  # 미리보기용 이미지
    preview = cv2.cvtColor(preview, cv2.COLOR_GRAY2BGR)

    each = int(w/line_cnt)  # 탐지선 간격

    v_cnt = 0  # 탐지 횟수 (가로)
    h_cnt = 0  # 탐지 횟수 (세로)

    # 왼쪽위부터 시작되는 대각선
    x1_cnt = 0
    # 오른쪽위부터 시작되는 대각선
    x2_cnt = 0

    for j in range(line_cnt + 1):
        v = each * j

        if j == 0 or j == line_cnt:
            continue

        # 미리보기 이미지에 탐지선 긋기
        cv2.line(preview, (v, 0), (v, w), (100, 255, 100), 1)
        cv2.line(preview, (0, v), (w, v), (0, 0, 255), 1)
        # 대각선 긋기
        cv2.line(preview, (int(w/2), 0), (w, int(w/2)), (255, 0, 0), 1)
        cv2.line(preview, (0, int(w/3)), (int(2*(w/3)), w), (255, 0, 0), 1)

        # 왼쪽 위부터 시작되는 대각선 검사
        started = False
        prev_filled = False
        for x, y in zip(range(int(w/2), w), range(0, int(w/2))):

            if canvas[y, x] == 0:
                if started is False:
                    started = True
                    cv2.circle(preview, (x, y), 1, (0, 0, 255), -1)
                    x1_cnt = x1_cnt + 1

                prev_filled = True
            else:
                if started is True and prev_filled is True:
                    cv2.circle(preview, (x, y), 1, (0, 0, 255), -1)
                    started = False

                prev_filled = False

        # 오른쪽 위부터 시작되는 대각선 검사
        # 를 왼쪽하단 역슬래시모양 대각선
        started = False
        prev_filled = False
        for x, y in zip(range(int(2*(w/3))), range(int(w/3), w)):

            if canvas[y, x] == 0:
                if started is False:
                    started = True
                    cv2.circle(preview, (x, y), 1, (0, 0, 255), -1)
                    x2_cnt = x2_cnt + 1

                prev_filled = True
            else:
                if started is True and prev_filled is True:
                    cv2.circle(preview, (x, y), 1, (0, 0, 255), -1)
                    started = False

                prev_filled = False

        started = False
        prev_filled = False
        for x in range(w):
            y = v

            if canvas[y, x] == 0:
                if started is False:
                    started = True
                    cv2.circle(preview, (x, y), 1, (0, 255, 0), -1)
                    v_cnt = v_cnt + 1

                prev_filled = True
            else:
                if started is True and prev_filled is True:
                    cv2.circle(preview, (x, y), 1, (255, 0, 0), -1)
                    started = False

                prev_filled = False

        started = False
        prev_filled = False
        for y in range(w):
            x = v

            if canvas[y, x] == 0:
                if started is False:
                    started = True
                    cv2.circle(preview, (x, y), 1, (0, 255, 0), -1)
                    h_cnt = h_cnt + 1

                prev_filled = True
            else:
                if started is True and prev_filled is True:
                    cv2.circle(preview, (x, y), 1, (255, 0, 0), -1)
                    started = False

                prev_filled = False

    if show_window:
        font_name = str(randint(1, 100000))
        cv2.namedWindow(font_name + str(i) + str(line_cnt), cv2.WINDOW_NORMAL)
        cv2.imshow(font_name + str(i) + str(line_cnt), preview)

    return [v_cnt, h_cnt, x1_cnt, x2_cnt]

# 23찾는 함수


def finding23(img, line_cnt, show_window=False):

    rect = getRectContour(img)  # 외곽선 위치 값 찾기

    cropped = img[rect["from"][1]:rect["to"][1],
                  rect["from"][0]:rect["to"][0]]  # 외곽선 위치로 이미지 자르기

    # 자른 이미지 w 값 크기로 리사이징
    canvas = cv2.resize(cropped, (w, w), interpolation=cv2.INTER_AREA)
    canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)  # 탐색할 이미지는 GRAYSCALE로 변환

    # 이미지 리사이징 하면서 색이 흐려지는 픽셀이 생기는데 Threshold 적용해서 B&W 이미지로 변환
    (_, canvas) = cv2.threshold(canvas, 127, 255, cv2.THRESH_BINARY)

    preview = canvas.copy()  # 미리보기용 이미지
    preview = cv2.cvtColor(preview, cv2.COLOR_GRAY2BGR)

    each = int(w/line_cnt)  # 탐지선 간격

    x2_cnt = 0

    started = False
    prev_filled = False
    for x, y in zip(range(int(2*(w/3))), range(int(w/3), w)):

        if canvas[y, x] == 0:
            if started is False:
                started = True
                cv2.circle(preview, (x, y), 1, (0, 125, 255), -1)
                x2_cnt = x2_cnt + 1

            prev_filled = True
        else:
            if started is True and prev_filled is True:
                cv2.circle(preview, (x, y), 1, (0, 125, 255), -1)
                started = False

            prev_filled = False

    return [x2_cnt]


def finding69(img, line_cnt, show_window=False):

    rect = getRectContour(img)  # 외곽선 위치 값 찾기

    cropped = img[rect["from"][1]:rect["to"][1],
                  rect["from"][0]:rect["to"][0]]  # 외곽선 위치로 이미지 자르기

    # 자른 이미지 w 값 크기로 리사이징
    canvas = cv2.resize(cropped, (w, w), interpolation=cv2.INTER_AREA)
    canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)  # 탐색할 이미지는 GRAYSCALE로 변환

    # 이미지 리사이징 하면서 색이 흐려지는 픽셀이 생기는데 Threshold 적용해서 B&W 이미지로 변환
    (_, canvas) = cv2.threshold(canvas, 127, 255, cv2.THRESH_BINARY)

    preview = canvas.copy()  # 미리보기용 이미지
    preview = cv2.cvtColor(preview, cv2.COLOR_GRAY2BGR)

    each = int(w/line_cnt)  # 탐지선 간격

    v_cnt = 0  # 탐지 횟수 (가로)
    h_cnt = 0  # 탐지 횟수 (세로)

    for j in range(line_cnt + 1):
        v = each * j

        if j == 0 or j == line_cnt:
            continue

        # 미리보기 이미지에 탐지선 긋기
        # cv2.line(preview, (v, 0), (v, w), (0, 0, 255), 1)
        cv2.line(preview, (0, v), (w, v), (0, 0, 255), 1)

        started = False
        prev_filled = False
        for x in range(w):
            y = v

            if y != 48:
                continue

            if canvas[y, x] == 0:
                if started is False:
                    started = True
                    cv2.circle(preview, (x, y), 2, (0, 255, 0), -1)
                    v_cnt = v_cnt + 1

                prev_filled = True
            else:
                if started is True and prev_filled is True:
                    cv2.circle(preview, (x, y), 2, (255, 0, 0), -1)
                    started = False

                prev_filled = False

        started = False
        prev_filled = False
        for y in range(w):
            x = v

            if canvas[y, x] == 0:
                if started is False:
                    started = True
                    cv2.circle(preview, (x, y), 2, (0, 255, 0), -1)
                    h_cnt = h_cnt + 1

                prev_filled = True
            else:
                if started is True and prev_filled is True:
                    cv2.circle(preview, (x, y), 2, (255, 0, 0), -1)
                    started = False

                prev_filled = False

    if show_window:
        font_name = str(randint(1, 100000))
        cv2.namedWindow(font_name + str(i) + str(line_cnt), cv2.WINDOW_NORMAL)
        cv2.imshow(font_name + str(i) + str(line_cnt), preview)

    return [v_cnt, h_cnt]


num_map = {}
isSixNine = {}

if input_file == 'make':

    for i in range(10):
        print("Number: " + str(i))

        arr_line = []
        for line_cnt in lines:
            arr_font = []
            for k in range(3):
                font_name = folders[k]
                str_row = font_name + ' -'

                img = loadImage(font_name, i)  # 이미지 로딩

                [v_cnt, h_cnt, x1_cnt, x2_cnt] = analyzeImage(
                    img, line_cnt, show_window=False)
                H = h_cnt
                V = v_cnt
                X1 = x1_cnt
                X2 = x2_cnt
                # !!!!!!!!! 여기가 곱하고 더하고 어쩌구~ !!!!!!!!!!!!
                arr_font.append(H*V)

            arr_line.append(arr_font)

        print(arr_line)
        print("===========================")

        num_map[i] = arr_line

    # 6 9 판별기
    for i in range(10):
        arr_line = []
        for line_cnt in lines:
            arr_font = []
            for k in range(3):
                font_name = folders[k]
                str_row = font_name + ' -'

                img = loadImage(font_name, i)  # 이미지 로딩

                [v_cnt, h_cnt, x1_cnt, x2_cnt] = analyzeImage(
                    img, line_cnt, show_window=False)
                arr_font.append(v_cnt)

            arr_line.append(arr_font)

        # print(arr_line)
        # print("===========================")

        isSixNine[i] = arr_line

    with open('./data.json', 'w', encoding='utf-8') as f:
        json.dump(num_map, f)
    with open('./data_69.json', 'w', encoding='utf-8') as f:
        json.dump(isSixNine, f)

    print('generated. check the data.json file')


else:
    with open('./data.json', 'r') as f:
        num_map = json.load(f)
    with open('./data_69.json', 'r') as f:
        isSixNine = json.load(f)
    # 테스트셋 판별기
    check_map = {}

    img = load_image_by_path(input_file)

    checking = []

    for line_cnt in lines:

        [v_cnt, h_cnt, x1_cnt, x2_cnt] = analyzeImage(
            img, line_cnt, show_window=False)
        H = h_cnt
        V = v_cnt
        X1 = x1_cnt
        X2 = x2_cnt
        # !!!!!!!!! 여기가 곱하고 더하고 어쩌구~ !!!!!!!!!!!!
        checking.append(H*V)

    cnt_list = []
    for N in num_map.values():
        cnt = 0
        for C, answer in zip(checking, N):
            # print(C,answer)
            if C in answer:
                cnt += 1
        cnt_list.append(cnt)

    predict = cnt_list.index(max(cnt_list))
# 6 9 예외처리기
    if predict == 6 or predict == 9:
        checking = []
        for line_cnt in lines:
            [v_cnt, h_cnt] = finding69(img, line_cnt, show_window=False)
            checking.append(v_cnt)

        cnt_list = []
        for N in isSixNine.values():
            cnt = 0
            for C, answer in zip(checking, N):
                # print(C,answer)
                if C in answer:
                    cnt += 1
            cnt_list.append(cnt)

        if checking[1] == 1:
            predict = 9
        if checking[1] == 2:
            predict = 6
    else:
        predict = cnt_list.index(max(cnt_list))

    checking23 = []
    # 2 3 판별
    # 3을 2로 읽는 경우가 많으므로, 예상치가 2일때 답이 3이라면 2가 아님을 알려줘야 함
    if predict == 2 or predict == 3:
        iCheck = 0

        for line_cnt in lines:
            x2_cnt_23 = finding23(img, line_cnt, show_window=False)
            XX22 = x2_cnt_23

            checking23.append(XX22)

        # 답이 2일 경우 checking23의 2번째 항목은 짝수, 3일 경우 홀수임을 확인

        if (int(checking23[0][0]) % 2) == 1:
            predict = 3
        elif (int(checking23[0][0]) % 2) == 0:
            predict = 2

    print()
    print("Recognition Result : " + str(predict))


if present_window:
    cv2.waitKey(0)
    cv2.destroyAllWindows()
