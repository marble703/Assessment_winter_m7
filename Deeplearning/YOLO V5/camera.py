import cv2

# 创建一个 VideoCapture 对象，用于捕获视频流
cap = cv2.VideoCapture(0)  # 参数 0 表示打开默认摄像头，如果有多个摄像头可以尝试其他编号

# 检查摄像头是否成功打开
if not cap.isOpened():
    print("Error: Failed to open camera.")
    exit()

# 循环读取并显示摄像头捕获的视频流
while True:
    # 读取一帧视频
    ret, frame = cap.read()

    # 检查是否成功读取视频帧
    if not ret:
        print("Error: Failed to read frame.")
        break

    # 显示视频帧
    cv2.imshow('Camera', frame)

    # 检查用户是否按下了 'q' 键，如果是则退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头资源并关闭窗口
cap.release()
cv2.destroyAllWindows()