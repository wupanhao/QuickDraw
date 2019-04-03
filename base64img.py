
# coding: utf-8
import base64
import numpy as np
import cv2
 
# img_file = open(r'00.JPG','rb')   # 二进制打开图片文件
# img_b64encode = base64.b64encode(img_file.read())  # base64编码
# img_file.close()  # 文件关闭

# img_b64encode = "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAFA3PEY8MlBGQUZaVVBfeMiCeG5uePWvuZHI////////////////////////////////////////////////////2wBDAVVaWnhpeOuCguv/////////////////////////////////////////////////////////////////////////wAARCAEYARgDASIAAhEBAxEB/8QAGQABAAMBAQAAAAAAAAAAAAAAAAMEBQEC/8QANBABAAIBAgQCBwcEAwAAAAAAAAECAwQREiExQVFhEyIycaHB8CMzQlKRsdEFcoHhFFOS/8QAFAEBAAAAAAAAAAAAAAAAAAAAAP/EABQRAQAAAAAAAAAAAAAAAAAAAAD/2gAMAwEAAhEDEQA/AKQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA9xiyTG8Y7THjEHocv/Xf/AMyDwO2ras7WrMT5w4AAAAAAAAAOxEzO0RvMrGPRZL7TbakefUFZ2tbWnatZmfKGjXTYcUTNoifO/wBbFtXhpG0Tvty2rH1AKtdHmt1iK++f4SxoJ255Ofu/25fXz+CkR5yinV55n29v8QCadBO3LJz93+1XLithvw22368lvR575L2red+W8I9f9/H9vzkFUAAAAAAAAAAABLj02XJ0rtHjPJ70cYpy7ZI3mfZ36fXgs59XXHM1rHFaP0gDHo8dPa9efPo9zmwYuUWrXyrH8KGXUZMvtTtHhHREDRnXYonaItPnEOf87F+W/wCkfyzwGpkrXU6fl3jePKfrqzLVmlpraNpjssaXUzinhvzpPwWNZg9JTirHrV8O8AzgAAAAdrWb2itY3mewOLOHSXyc770r8ZWNPpq4Y48m02679oeM+tiN64uc/mn5Am2w6aN/Vr+8/NXya6Z3jHXbzlUtab2m1p3me7gPV8l8k73tM+95AAAFrQVn0trdojZzXTvn90Qhx5LYrcVJ2l5mZmZmZ3mQcAAAAAAAAAAAAAAAAAAaGhzcVPRzPrV6e5nveK848lbx2kEmrxeiyztHq25wgaepp6bT715/ijzZgAJMOG2a/DXp3nwBzHjtlvFaxz/Zo48ePTY9525dbT9fB37PS4vCI/WZZ+bNfNbe07R2r4A9ajU2zb1jlTw8fegAAAAAAAAAAAAAAAAAAAAAAAAAAAAAGlobcWniNvZmY+fzZ+SsVyWrHSJmFr+n32tek78+cPOrxWnUxw/j6AiwYbZskRHSOs+C/kvj02KNo28Kx3Kxj0mHnPLvPjLOy5LZck2t36eQGTJfLbivO/l2h4AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAEmC01z0mJ25w1JrE2iZjnXpLIiJmdojeZa2W8Y8drz2gFLXZePJwRPKvX3/AF81V2Zm1pmes85cAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABZ0OPjzcUxyrG/8AldzxE4Mm8b+rKHQU4cM272n4fW6bLMW097R0mkzH6AyQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAalZ4NHE15TFN/g5m+y0kx12rw/JJh+4x/2x+yrrs8THoqzvtPrT8gUgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAe4y5IjaMloiPN4AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAH/2Q=="
img_b64encode = "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAFA3PEY8MlBGQUZaVVBfeMiCeG5uePWvuZHI////////////////////////////////////////////////////2wBDAVVaWnhpeOuCguv/////////////////////////////////////////////////////////////////////////wAARCAEYARgDASIAAhEBAxEB/8QAGQABAAMBAQAAAAAAAAAAAAAAAAMEBQIB/8QAORAAAgIBAQQEDAYCAwEAAAAAAAECAxEEEiExUQUTQWEiMjNTcXKBkbHB0fAUNEJSkqFD4WJjovH/xAAUAQEAAAAAAAAAAAAAAAAAAAAA/8QAFBEBAAAAAAAAAAAAAAAAAAAAAP/aAAwDAQACEQMRAD8ApAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAD1JyeEm3yRNDSXT7FFf8gIAXVoFlZs3dyJI6KpPftS7m/oBQhCVktmCyydaK1rLcV3Nl6FcK1iEUjsDP/A2/uh739Diektgs4UvVNMAZDrsisyhJLm0cGy1lYfAq36NTalViL7V2AUAAAAJY6e6XCt+3d8QIgW46Gb8aUUu7eAKgAAAHqTbwllsDwFmvRWTScmoLv4lmOipjxTl6X9AM06jXOSzGEmu5GpCmuGNmCWO3G8kAzYaO2Sy0o+lnX4G390Pe/oaBw7a08OyKa7wKX4G390Pe/ocz0dsI53S7o8S/11XnIfyR2BjNNPDWGjw1NRQro9imuDMyUXCTjJYa7APACWrT2W+LHC5vcgIj2MZSeIxbfci/XooR8duT9yLMYqKxFJLkkBQr0VkvHagveyxDR0x4pyfeywQWauqH6tp/8d4EsYxisRiku5HRQnr5vxIKPp3kE7rLPGm2uXYBpu2uLw5xTXY2RvV0pZ2s9yTMwAXpa+OfBg36Xj6kf4639sPc/qVT1JyeIpt8kBPPWXS4NR9CIuut85P+TJq9FZLfLEF38SzXpKob2tp9/wBAItJqpOSrszLPB/fxLp4kksJJ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"
img_b64decode = base64.b64decode(img_b64encode)  # base64解码
image_x = 28
image_y = 28
img_array = np.fromstring(img_b64decode,np.uint8) # 转换np序列
print(img_array.shape)
img = cv2.imdecode(img_array,cv2.COLOR_BGR2RGB)  # 转换Opencv格式
img = cv2.resize(img, (image_x, image_y))
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
print(gray.shape)
 
cv2.imshow("img",img)
cv2.imwrite("test.bmp",img)
cv2.waitKey()