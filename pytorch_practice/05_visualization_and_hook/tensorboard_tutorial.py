"""
TensorBoard 是 TensorFlow 中强大的可视化工具，支持标量、文本、图像、音频、视频和 Embedding 等多种数据可视化。

在 PyTorch 中也可以使用 TensorBoard，具体是使用 TensorboardX 来调用 TensorBoard。
除了安装 TensorboardX，还要安装 TensorFlow 和 TensorBoard，其中 TensorFlow 和 TensorBoard 需要一致。

TensorBoardX 可视化的流程需要首先编写 Python 代码把需要可视化的数据保存到 event file 文件中，
然后再使用 TensorBoardX 读取 event file 展示到网页中。

pip install tensorboardX
"""
import numpy as np
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
max_epoch = 100

writer = SummaryWriter(comment='test_comment', filename_suffix="test_suffix")

for x in range(max_epoch):
    x = np.array(x)
    writer.add_scalar('y=2x', x * 2, x)
    writer.add_scalar('y=pow_2_x', 2 ** x, x)

    writer.add_scalars('data/scalar_group', 
                       {"xsinx": x * np.sin(x), "xcosx": x * np.cos(x)}, 
                       x)

writer.close()


"""
最上面的一栏显示的是数据类型，由于我们在代码中只记录了 scalar 类型的数据，因此只显示SCALARS。

点击INACTIVE显示我们没有记录的数据类型。设置里可以设置刷新 tensorboard 的间隔，在模型训练时可以实时监控数据的变化

左边的菜单栏如下，点击Show data download links可以展示每个图的下载按钮，如果一个图中有多个数据，需要选中需要下载的曲线，然后下载，格式有 csv和json可选。

第二个选项Ignore outliers in chart scaling可以设置是否忽略离群点

Horizontal Axis表示横轴：STEP表示原始数据作为横轴，RELATIVE和WALL都是以时间作为横轴，单位是小时，RELATIVE是相对时间，WALL是绝对时间。

runs显示所有的 event file，可以选择展示某些 event file 的图像，其中正方形按钮是多选，圆形按钮是单选。

上面的搜索框可以根据 tags 来搜索数据对应的图像
"""