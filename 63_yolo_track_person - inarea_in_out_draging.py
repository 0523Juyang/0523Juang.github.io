import os
# 設定環境變數，允許多個 OpenMP 執行環境共存，避免程式因為重複載入而出錯
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 引入所需的套件，包括 YOLO 模型、OpenCV、時間處理、Shapely 幾何和 NumPy
from ultralytics import YOLO
import cv2, time
from shapely.geometry import Point, Polygon
import numpy as np
from collections import defaultdict

# 用於追蹤的列表，key 是物件 ID
#persontrackList = defaultdict(lambda: []) # Store tracking ID history, key is the object ID
in_countList = defaultdict(lambda: [])  # 進入計數
out_countList = defaultdict(lambda: [])  # 離開計數
pointTrackList = defaultdict(lambda: []) # Store tracking location history, key is the object ID

line_position = []  # List to store the start and end points of the line
area=[[]] # List to store the area of the line
is_drawing = False  # Flag to check if the mouse is dragging
start_point = None  # Variable to store the start point of the line

# 檔案名稱
filename = "area.txt"


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

# 檢查檔案是否存在
if os.path.exists(filename):
    # 讀取檔案內容
    with open(filename, "r") as file:
        content = file.read()
        area[0] = eval(content)  # 轉換檔案內容為 Python 結構
    print("檔案已讀取，內容為：", area)






'''
# 定義檢測區域，這裡是一個矩形區域（四個點的座標）
area = [
    [[1035, 163], [1193, 159], [1269, 526], [1056, 528]],   # 區域1
    [[1460, 148], [1672, 134], [1836, 413], [1627, 438]]    # 區域2
    ]

'''
def mousePoints(event, x, y, flags, params):
    """
    Handles mouse events for drawing a line by dragging.
    """
    global is_drawing, start_point, line_position, frame
    if len(line_position) < 5:
        if event == cv2.EVENT_LBUTTONDOWN:  # 按下左鍵開始拖拉
            is_drawing = True
            start_point = (x, y)  # 設置起點
            print(f"Line start at {start_point}")

        elif event == cv2.EVENT_MOUSEMOVE and is_drawing:  # 拖拉時更新畫面
            temp_frame = frame.copy()  # 使用臨時畫面顯示拖拉效果
            cv2.line(temp_frame, start_point, (x, y), (255, 255, 180), 2)
            cv2.imshow('YOLOv8', temp_frame)


        elif event == cv2.EVENT_LBUTTONUP:  # 放開左鍵確定終點
            is_drawing = False
            end_point = (x, y)  # 設置終點
            line_position.append([start_point, end_point])  # 保存線條起點和終點
            area[0].extend([start_point])

            if len(area[0]) ==4:
                 # 將資料寫入檔案
                    with open(filename, "w") as file:
                        file.write(str(area[0]))  # 將資料轉成字串寫入
                        print(f"資料已成功儲存到 {filename}")

            print(f"Line drawn from {start_point} to {end_point}")
            cv2.line(frame, start_point, end_point, (255, 255, 180), 2)  # 畫出最終線條

# 取得相交比例的函數
# 用於計算物件和區域之間的重疊百分比
def inArea(center_object, area):
    polygon = []
    for a in area:
        polygon.append(Polygon(a)) 
    point = center_object


    point = Point(point)
    for i,location in enumerate(polygon):
        
        if location.contains(point): # 如果區域包含物件,返回區域編號
            cv2.putText(frame, str(point) +"In Area" +str(i+1), (30, (100+i*50)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
            return i
    return -1
    

# 繪製區域的函數
# 用於在影像上標註定義好的區域
def drawArea(f, area, color, th):
    for a in area:
        v = np.array(a, np.int32)  # 將區域頂點轉換為 NumPy 整數數組
        cv2.polylines(f, [v], isClosed=True, color=color, thickness=th)  # 繪製多邊形
    return f
#
def plotTrack(f, trackList):
                    frame = f                
                    points = np.array(trackList, dtype=np.int32).reshape((-1, 1, 2))
                    #現在位置
                    cv2.circle(frame, (trackList[-1]), 7, (0,0,255), -1)
                    #移動路徑
                    cv2.polylines(frame, [points], isClosed=False, color=(0,0,255), thickness=2)
                    return frame



# 主循環，持續處理每一幀
while True:
    try:
        st = time.time()  # 記錄開始時間，用於計算 FPS
        r, frame = cap.read()  # 讀取影片的每一幀
        if len(area[0]) < 4:  # 如果已經定義了區域
        # 繪製已完成的線條
            for line in line_position:
                cv2.line(frame, line[0], line[1], (255, 255, 180), 2)

        if len(area[0]) == 4: 


            # 使用 YOLO 模型進行物件追蹤q
            results = model.track(frame, persist=True, verbose=False)


            # 解析模型的檢測結果
            result_list = results[0].boxes.data.tolist()
            for data in result_list:
                
                if results[0].boxes.id is not None:  # 檢查是否有檢測到物件分數
                    if int(data[6]) in [0]:  # 如果檢測到的物件是person, object ID = 0
                        #print(f"Detect ID: {str(data[4])}")
                        
                        x1, y1, x2, y2 = int(data[0]), int(data[1]), int(data[2]), int(data[3])  # 取得person的邊界框座標                  
                        person_center = ((int((x1 + x2) / 2), int((y1 + y2) / 2)))
                        personIn = inArea(person_center, area) # 計算person在哪一區域
                        #trackId = persontrackList[personIn]# 取得該區域ID的追蹤列表
                        trackList = pointTrackList[int(data[4])]# 取得該區域物件座標的追蹤列表
                        

                        if personIn >= 0:  # 如果 person 在區域內
    
                        # 將person的中心點加入追蹤列表
                            trackList.append(person_center)
                        # 繪製追蹤路徑
                            frame = plotTrack(frame, trackList)
                            # 保留最近30個位置
                            if len(trackList) > 30:
                                trackList.pop(0)

                        #    if int(data[4]) not in trackId:
                        #       trackId.append(int(data[4]))  # 將 person_track id  加入追蹤列表    

                            if len(pointTrackList[int(data[4])]) > 10:
                                # 確認移動方向
                                prev_y = pointTrackList[int(data[4])][-9][1] # 取得前第5個座標的y座標
                                curr_y = pointTrackList[int(data[4])][-1][1] # 取得目前座標的y座標
                                if prev_y>= area[personIn][0][1] and ((curr_y -prev_y) >10):
                                    if (int(data[4]) not in out_countList[personIn]) and (int(data[4]) not in in_countList[personIn]):
                                        out_countList[personIn].append(int(data[4]))  # 儲存離開ID
                                elif prev_y <= area[personIn][3][1] and ((prev_y - curr_y) >10) :
                                    if (int(data[4]) not in out_countList[personIn]) and (int(data[4]) not in in_countList[personIn]):
                                        in_countList[personIn].append(int(data[4]))  # 儲存進入ID
                                # 在影像上繪製person的邊界框和 ID
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame, "ID =" + str(int(data[4])), (x1, y1), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2, cv2.LINE_AA)
            frame = drawArea(frame, area, (0, 0, 255), 3)  # 在影像上繪製指定區域

        # 計算並顯示每秒幀數 (FPS)
        et = time.time()
        FPS = round((1 / (et - st)), 1)
        cv2.putText(frame, 'FPS=' + str(FPS), (30, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2, cv2.LINE_AA)
        # 顯示目前追蹤的person數量
        cv2.putText(frame, 'Line_1 IN:' + str(len(in_countList[0]))+' OUT:'+str(len(out_countList[0])), (30, 200), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 3, cv2.LINE_AA)
        cv2.putText(frame, 'Line_2 IN:' + str(len(in_countList[1]))+' OUT:'+str(len(out_countList[1])), (30, 230), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 3, cv2.LINE_AA)

        # 顯示處理後的影像
        cv2.imshow('YOLOv8', frame)
        if len(area[0]) == 0:
            cv2.setMouseCallback('YOLOv8', mousePoints)  # 設置滑鼠回調函數q
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # 按下 q 鍵離開迴圈
            break
        elif key == ord('r'):  # 按下 r 鍵清除所有線條
            line_position = []  # 清空線條列表
            area = [[]]
            in_countList = defaultdict(lambda: [])  # 進入計數
            out_countList = defaultdict(lambda: [])  # 離開計數
            print("All lines cleared!")
    except Exception as e:
        print(e)
        break

# 釋放影片資源並關閉窗口
cap.release()
cv2.destroyAllWindows()
