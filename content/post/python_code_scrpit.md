use qt to convert svg to png
```python
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtGui import QImage, QPainter
from PIL import Image
import sys

def svg_to_png_qt(svg_path, png_path, size=(2048, 2048)):
    # 创建 SVG 渲染器
    renderer = QSvgRenderer(svg_path)
    if not renderer.isValid():
        raise ValueError("Invalid SVG file")
    
    # 创建图像和绘制器
    image = QImage(size[0], size[1], QImage.Format_ARGB32)
    image.fill(0)  # 透明背景
    
    painter = QPainter(image)
    renderer.render(painter)
    painter.end()
    
    # 保存为 PNG
    image.save(png_path)
    
    # 使用 Pillow 打开（可选）
    # img = Image.open(png_path)
    # img.save(png_path)

# 使用示例
svg_to_png_qt('output_layout.svg', 'output.png')
```


