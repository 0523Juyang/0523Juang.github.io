import os
# 設定環境變數，允許多個 OpenMP 執行環境共存，避免程式因為重複載入而出錯
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 引入所需的套件，包括 YOLO 模型、OpenCV、時間處理、Shapely 幾何和 NumPy
from ultralytics import YOLO
import cv2, time
from shapely.geometry import Polygon
import numpy as np

# 建立一個名為 'YOLOv8' 的 OpenCV 窗口，允許使用者調整視窗大小
cv2.namedWindow('YOLOv8', cv2.WINDOW_NORMAL)

# 加載 YOLOv8 模型，使用中等大小的預訓練權重 ('yolov8m.pt')
model = YOLO("yolov8m.pt")
# 獲取模型的分類名稱，例如人、車、動物等物件名稱
names = model.names

# 目標影片來源，可以是本地文件或網絡流媒體
# target = 0  # 如果目標是攝影機
# target = "city.mp4"  # 使用本地影片文件
# target = "https://cctv-ss04.thb.gov.tw/T17-264K+800"  # 網絡攝影機串流

target = "Counter.mp4"
cap = cv2.VideoCapture(target)  # 讀取影片來源

# 定義檢測區域，這裡是一個矩形區域（四個點的座標）
area = [
    [[1148, 33], [1184, 148], [992, 159], [970, 44]],   # 區域1
    [[1479, 172], [1676, 161], [1635, 25], [1436, 37]] # 區域2
    ]

# 用於追蹤的車輛列表
persontrackList = [[],[]]

# 取得相交比例的函數
# 用於計算物件和區域之間的重疊百分比
def inarea(object, area):
    
    # 根據物件邊界值計算其四個頂點，形成矩形
    b = [[object[0], object[1]], [object[2], object[1]], [object[2], object[3]], [object[0], object[3]]]
    for i in range(len(area)):
        poly1 = Polygon(b)  # 創建物件的多邊形
        poly2 = Polygon(area[i])  # 創建區域的多邊形
        intersection_area = poly1.intersection(poly2).area  # 計算相交的面積
        poly1Area = poly1.area  # 計算物件的總面積
        overlap_percent = (intersection_area / poly1Area) * 100  # 計算相交的百分比
        if overlap_percent >40:
            return i

    return -1

# 繪製區域的函數
# 用於在影像上標註定義好的區域
def drawArea(f, area, color, th):
    for a in area:
        v = np.array(a, np.int32)  # 將區域頂點轉換為 NumPy 整數數組
        cv2.polylines(f, [v], isClosed=True, color=color, thickness=th)  # 繪製多邊形
    return f

# 主循環，持續處理每一幀
while True:
    try:
        st = time.time()  # 記錄開始時間，用於計算 FPS
        r, frame = cap.read()  # 讀取影片的每一幀
        # 使用 YOLO 模型進行物件追蹤
        results = model.track(frame, persist=True, verbose=False)


        # 解析模型的檢測結果
        result_list = results[0].boxes.data.tolist()
        for data in result_list:
            if results[0].boxes.id is not None:  # 檢查是否有檢測到物件分數
                if int(data[6]) in [0]:  # 如果檢測到的物件是person, object ID = 0
                    x1, y1, x2, y2 = int(data[0]), int(data[1]), int(data[2]), int(data[3])  # 取得person的邊界框座標                  
                    personIn = inarea(data, area) # 計算person在哪一區域
                    if personIn >= 0:  # 如果 person 在區域內
                        if int(data[4]) not in persontrackList[personIn]:
                            persontrackList[personIn].append(int(data[4]))  # 將 person_track id  加入追蹤列表
                            
                        # 在影像上繪製person的邊界框和 ID
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, "ID =" + str(int(data[4])), (x1, y1), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2, cv2.LINE_AA)


        # 計算並顯示每秒幀數 (FPS)
        et = time.time()
        FPS = round((1 / (et - st)), 1)
        cv2.putText(frame, 'FPS=' + str(FPS), (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2, cv2.LINE_AA)
        # 顯示目前追蹤的person數量
        cv2.putText(frame, 'Line_1=' + str(len(persontrackList[0])), (900, 80), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, 'Line_2=' + str(len(persontrackList[1])), (900, 110), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 3, cv2.LINE_AA)
        frame = drawArea(frame, area, (0, 0, 255), 3)  # 在影像上繪製指定區域
        # 顯示處理後的影像
        cv2.imshow('YOLOv8', frame)
        key = cv2.waitKey(1)  # 等待按鍵輸入
        if key == 27:  # 按下 'ESC' 鍵退出
            break
    except Exception as e:
        print(e)
        break

# 釋放影片資源並關閉窗口
cap.release()
cv2.destroyAllWindows()
