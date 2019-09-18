# MotionDet
### 使用opencv处理前端视频，通过一条线上，像素值的变化判断是否有动作的产生
#### 初始化
```
    def __init__(self, line_y1, line_y2, xmin, ymin, xmax, ymax):
```
> 参数初始化，需要给定一个矩形框，和触发动作的两条线

#### 连续帧的长度
```
        self.num_cache1 = [0] * 11
```
> 这里设置的是11

#### 阈值的设置
```
 1.   sum(self.num_cache1[5:8]) > 15
 2.   sum(self.num_cache2[5:8]) > 15
 3.  sum( self.num_cache1[2:5]) > 15
 4.  sum( self.num_cache2[2:5]) > 15
 5.  self.num_cache1[5] >= 30:
 6.  self.num_cache2[5] >= 30:
```
>  通过这些阈值来判断是否产生动作，这些阈值是可以修改的

#### 需要重新划线的话可以使用
```
python drawLine.py
```
