# 两层VGG网络
#  1.卷积层1 (Conv2D) + ReLU激活函数 + 最大池化层 (MaxPool2D)
#  2.卷积层2 (Conv2D) + ReLU激活函数 + 最大池化层 (MaxPool2D)
#  3.展平层 (Flatten)
#  4.全连接层1 (Linear) + ReLU激活函数
#  5.全连接层2 (Linear) 输出层
# 假设卷积层的padding为1，步长为1，池化层为2x2，步长为2
# 将实现以下类：
# Conv2D  ReLU  MaxPool2D  Linear  Flatten  Sequential （用于组合层）






import numpy as np


# ==================== 基础层实现 ====================

# 卷积层的类，图像处理中的一个"特征提取器"
class Conv2D:
    """
    2D卷积层（包含前向和反向传播）
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, stride: int = 1, padding: int = 1):
        # in_channels：输入数据的通道数（eg：RGB图像有3个通道）
        # out_channels：输出通道数，也就是这一层我们要用多少个卷积核（滤波器），每个卷积核会提取一种特征
        # kernel_size：卷积核的大小（eg：3x3）
        # stride：步长，即卷积核每次移动的距离
        # padding：在输入数据周围填充0的圈数，用来保持图像尺寸
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # 初始化权重
        scale = np.sqrt(1.0 / (in_channels * kernel_size * kernel_size))
        # 假设：in_channels=3, out_channels=16, kernel_size=3
        # 那么 self.W 的形状是 (16, 3, 3, 3)
        # 表示：16个过滤器，每个过滤器有3个3×3的小矩阵，分别对应R、G、B三个通道
        # 为什么要 * scale：这是"He初始化"方法，目的是让初始化的权重值在合适的范围内，避免梯度消失或爆炸
        # 设输出的方差：Var(输出) = (输入节点数) × Var(W) × Var(X)= n_in × σ² × γ²
        # 为了保持每一层输出的方差稳定（避免指数级增长或衰减），我们希望：n_in × σ² ≈ 1(对于前向传播)
        # 因此：σ² = 1 / n_in  j️ σ = sqrt(1 / n_in)  ← 这就是scale因子

        self.W = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * scale
        self.b = np.zeros((out_channels, 1))

        # 缓存用于反向传播
        # input：存储输入数据，用于反向传播
        # output：存储输出数据，用于反向传播
        # dW和db：存储权重的梯度，用于更新参数
        self.input = None
        self.output = None
        self.dW = None
        self.db = None


    # 前向传播就是数据从输入到输出的计算过程
    def forward(self, x: np.ndarray) -> np.ndarray:

        # 获取输入数据的形状：batch_size是批处理大小，C是通道数，H和W是高度和宽度
        batch_size, C, H, W = x.shape

        # 添加padding
        # 如果padding大于0，就在输入数据的周围填充0。这样做的目的是为了保持图像尺寸（当stride = 1时，输出尺寸等于输入尺寸）并让边缘像素也可以被充分处理
        if self.padding > 0:
            # np.pad(数组, ((前填充, 后填充), (上填充, 下填充), (左填充, 右填充)), 模式：用常数填充，默认为0)
            #                         #batch维度不填充 #通道维度不填充   #高维度前后都填充   #宽维度前后都填充
            x_padded = np.pad(x,((0, 0), (0, 0),(self.padding, self.padding),(self.padding, self.padding)),mode='constant')
        else:
            x_padded = x

        # 计算输出尺寸
        # 公式：(输入尺寸 + 2 * padding - 卷积核尺寸) // stride + 1
        # 因为每次移动stride步，所以输出尺寸会缩小
        H_out = (H + 2 * self.padding - self.kernel_size) // self.stride + 1
        W_out = (W + 2 * self.padding - self.kernel_size) // self.stride + 1

        # 初始化输出
        # 创建一个全0的数组，准备存放计算结果
        output = np.zeros((batch_size, self.out_channels, H_out, W_out))

        # 执行卷积
        #第一重循环：batch_size 处理每个样本
        for b in range(batch_size):
            #第二重循环：out_channels 应用于卷积层
            for oc in range(self.out_channels):
                #第三、四重循环：遍历输出特征图的每个位置
                for i in range(H_out):
                    for j in range(W_out):
                        #计算窗口起始位置
                        h_start = i * self.stride #高度起始 = 输出位置*步长
                        w_start = j * self.stride #宽度起始 = 输出位置*步长
                        #计算窗口结束位置
                        h_end = h_start + self.kernel_size
                        w_end = w_start + self.kernel_size
                        #eg：假设输入为5x5，卷积核为3x3，步长为1，padding为0。那么输出特征图大小为3x3（因为(5-3)/1+1=3）。
                        # 对于输出位置(0,0)，窗口起始为(0,0)，结束为(3,3)（即行0-2，列0-2）。
                        # 对于输出位置(0,1)，窗口起始为(0,1)，结束为(3,4)（即行0-2，列1-3）。
                        # 对于输出位置(2,2)，窗口起始为(2,2)，结束为(5,5)（即行2-4，列2-4）

                        # 提取patch
                        # 在输入数据上取出与卷积核同样大小的patch（考虑到步长）
                        # x_padded[b, :, h_start:h_end, w_start:w_end]
                        # b: 第b个样本
                        # : : 所有输入通道
                        # h_start:h_end: 高度范围
                        # w_start:w_end: 宽度范围
                        patch = x_padded[b, :, h_start:h_end, w_start:w_end]

                        # 卷积运算
                        # 将patch与卷积核W[oc]逐元素相乘，然后求和，再加上偏置b[oc, 0]，就得到了卷积结果
                        # self.W[oc]: 第oc个卷积核，形状(in_channels, kernel_size, kernel_size)
                        # patch * self.W[oc]: 逐元素相乘
                        # np.sum(...): 所有元素求和
                        # + self.b[oc, 0]: 加上偏置
                        conv_result = np.sum(patch * self.W[oc]) + self.b[oc, 0]
                        output[b, oc, i, j] = conv_result

                        #eg：输入图像：   卷积核（过滤器）：
                        # 1 2 3 4      a b c
                        # 5 6 7 8      d e f
                        # 9 0 1 2      g h i
                        # 3 4 5 6
                        # 计算过程：
                        # 1. 取图像左上角3×3区域：[1,2,3; 5,6,7; 9,0,1]
                        # 2. 与卷积核对应相乘：[1*a, 2*b, 3*c; 5*d, 6*e, 7*f; 9*g, 0*h, 1*i]
                        # 3. 所有乘积相加：1*a + 2*b + ... + 1*i
                        # 4. 加上偏置b
                        # 5. 得到输出特征图的一个像素值
                        # 6. 向右滑动一步，重复计算...

        #保存输入输出用于反向传播
        self.input = x
        self.output = output
        #返回结果
        return output


    # 反向传播就是根据误差调整权重的过程
    def backward(self, dout: np.ndarray, lr: float = 0.001) -> np.ndarray:
        # dout：从上一层（或损失函数）传来的梯度，表示"输出应该怎么变化才能减少误差"

        x = self.input  # 取出前向传播时保存的输入
        # x是输入数据，其形状为(batch_size, in_channels, height, width)
        batch_size, C_in, H, W = x.shape
        # dout是从上一层（或损失函数）传过来的梯度，其形状为(batch_size, out_channels, H_out, W_out)
        # _, 表示不需要使用第一个维度
        _, C_out, H_out, W_out = dout.shape

        # 初始化梯度，这些梯度将在反向传播的过程中被计算并用于更新参数
        # 具体来说在卷积层的反向传播中需要计算三个梯度：
        # 1.对权重的梯度（self.dW）：每个卷积核的权重都会根据输出梯度和输入值进行更新
        # 2.对偏置的梯度（self.db）：每个卷积核的偏置都会根据输出梯度进行更新
        # 3.对输入的梯度（dx）：作为梯度传播到前一层（即前一层在反向传播中需要的梯度（接收到的dout）），所以需要计算输入数据的梯度
        # np.zeros_like(a) 是NumPy库中的一个函数，它返回一个与数组a形状相同，但所有元素都为0的数组
        # 为什么梯度数组的形状要和参数数组的形状一样？
        # 梯度下降更新参数时，每个参数都要减去其对应的梯度乘以学习率。因此，每个参数都需要一个梯度值，所以它们形状一致
        # eg：
        # 前向传播链：   Layer1 → Layer2 → Layer3 → ... → Loss
        # 反向传播链： Loss → ... → Layer3 → Layer2 → Layer1
        # 对于Layer2来说：
        # 它需要接收来自Layer3的梯度，并计算两个东西：
        # 1. 自己的参数梯度 (dW, db)   ← 用于更新自己的参数
        # 2. 给前一层的梯度 (dx)       ← 传递给Layer1
        # 关键：dx的形状必须与Layer2的输入（即Layer1的输出）相同
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        dx = np.zeros_like(x)

        # 添加padding到输入（为了在计算梯度时保持与前向传播相同的坐标映射，确保梯度能够正确地累加到对应的输入位置上）
        # 在前向传播中，对输入x进行padding，得到x_padded，然后在这个padded的输入上进行卷积操作
        # 在反向传播时，需要计算损失函数对原始输入x的梯度dx
        # 但是，在卷积操作中，由于使用了padding，并且卷积核在padded的输入上滑动，所以每个输入像素可能被多个卷积窗口覆盖，因此对原始输入x的梯度需要从多个卷积窗口的梯度累加得到
        # 具体来说，在反向传播过程中，先计算损失函数对padded输入x_padded的梯度dx_padded，然后去掉padding部分，得到对原始输入x的梯度dx
        if self.padding > 0:
            x_padded = np.pad(x,((0, 0), (0, 0),(self.padding, self.padding),(self.padding, self.padding)),mode='constant')
            dx_padded = np.zeros_like(x_padded)
        else:
            x_padded = x
            dx_padded = dx

        # 计算每个位置的梯度（同前向传播）
        for b in range(batch_size):
            for oc in range(C_out):
                for i in range(H_out):
                    for j in range(W_out):
                        h_start = i * self.stride
                        w_start = j * self.stride
                        h_end = h_start + self.kernel_size
                        w_end = w_start + self.kernel_size

                        # 提取patch（同前向传播）
                        patch = x_padded[b, :, h_start:h_end, w_start:w_end]

                        # 假设有一个3×3的卷积核：
                        # 前向传播：输出 = 输入×权重 +偏置
                        # 反向传播：如果输出误差大，说明权重需要调整
                        # 调整量 = 输入 × 误差
                        # 这就是 self.dW += dout * patch

                        # 计算dW
                        # 对于当前位置，损失函数对卷积核的梯度等于输入数据对应的patch乘以损失函数对输出的梯度
                        # self.dW[oc] 指的是第oc个卷积核（对应第oc个输出通道）的权重梯度
                        # 对于每个样本（b），每个输出通道（oc），以及输出特征图的每个位置（i, j），会计算一个梯度贡献
                        # 这个梯度贡献是：从上一层传来的梯度dout[b, oc, i, j]（一个标量）乘以对应的输入patch（形状为(in_channels, kernel_height,kernel_width)）
                        # 这个乘积是一个与卷积核形状相同的张量，它表示对于这个样本和这个位置，权重应该调整多少
                        self.dW[oc] += dout[b, oc, i, j] * patch

                        # 计算db
                        # 偏置的梯度等于损失函数对输出的梯度的累加
                        # 对于偏置self.db，每个输出通道有一个偏置，所以self.db[oc, 0]是一个标量
                        # 在反向传播中，对于每个样本和每个位置，偏置的梯度就是dout[b, oc, i, j]（因为偏置是加在每个输出上的，所以偏置的梯度就是输出梯度本身）
                        # 将所有样本和位置的梯度累加起来，得到self.db[oc, 0]
                        self.db[oc, 0] += dout[b, oc, i, j]

                        # 为什么self.dW[]和self.db[]的索引都使用oc（输出通道索引）？
                        # 在卷积层中有一组卷积核，每个卷积核生成一个输出通道。假设有out_channels个卷积核，那么对于每个卷积核，都有对应的权重和偏置
                        # 权重W的形状是(out_channels, in_channels, kernel_size, kernel_size)，这意味着对于每个输出通道（即每个卷积核），都有in_channels个2D卷积核（每个输入通道一个）

                        # 计算dx（损失函数对输入x的梯度）
                        # 损失函数对输入的梯度等于卷积核乘以损失函数对输出的梯度，然后累加到对应的位置
                        # 对于当前批次的第b个样本，当前输出通道为oc，输出位置为(i, j)，计算这个位置的梯度dout[b, oc, i, j]对输入梯度的贡献
                        # 计算一个梯度贡献矩阵：贡献 = W[oc] * dout[b, oc, i, j]（即权重矩阵的每个元素乘以该标量梯度）
                        # 这个贡献矩阵应该累加到输入梯度（dx）中对应样本（b）的所有输入通道的对应空间位置上。
                        # 具体来说，这个贡献矩阵对应的空间位置是输入特征图中与输出位置（i, j）对应的那个滑动窗口（即从输入特征图中提取的patch的位置，该位置由步长和填充决定）
                        dx_padded[b, :, h_start:h_end, w_start:w_end] += self.W[oc] * dout[b, oc, i, j]

        # 如果有padding，去掉padding得到dx
        # 如果之前对输入进行了padding，那么dx_padded比原始输入大，需要去掉padding，得到与原始输入同样大小的dx
        if self.padding > 0:
        # 第一个维度（batch）和第二个维度（通道）全部保留
        # 第三个维度（高度）：从self.padding开始，到 - self.padding结束（即倒数第self.padding个，不包括它）
        # 第四个维度（宽度）：同理
            dx = dx_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
        # 注意：如果padding = 0，那么self.padding: -self.padding就变成了0: 0，即空。所以需要判断padding是否大于0，如果等于0，则直接使用dx_padded
        else:
            dx = dx_padded

        # 梯度裁剪
        # 在深度神经网络中，梯度可能会在反向传播过程中变得非常大（梯度爆炸），这会导致参数更新步长过大使得训练不稳定，甚至导致数值溢出（NaN）。梯度裁剪通过限制梯度的最大值，可以稳定训练过程
        # 对于一个梯度向量（或张量）g，其模是各个元素平方和的平方根。如果每个元素的值都很大，那么模就会很大；反之，如果每个元素的值都很小，模就会很小。所以，限制了每个元素的大小后模的值也会受到限制
        # np.clip(arr, min, max)函数将数组arr中的元素限制在[min, max]之间。小于min的值被设置为min，大于max的值被设置为max
        self.dW = np.clip(self.dW, -1.0, 1.0)
        self.db = np.clip(self.db, -1.0, 1.0)

        # 更新参数
        # 使用梯度下降法更新权重和偏置，新权重 = 旧权重 - 学习率 × 梯度
        # 注意：在更新时使用减号，因为要最小化损失函数，所以需要朝着梯度的反方向移动。
        # 更新后的self.W和self.b将在下一个训练批次（batch）的前向传播中被使用，用来计算新的输出。
        # 然后再次计算损失，反向传播，更新参数，如此循环，直到损失函数的值达到一个可接受的范围，或者训练了指定的轮数（epoch）
        self.W -= lr * self.dW
        self.b -= lr * self.db

        # 返回输入梯度：用于上一层的反向传播
        return dx

    def __call__(self, x):
        return self.forward(x)

# 最大池化层的作用是下采样，即缩小特征图的尺寸，同时保留最重要的信息。它通过取每个小区域的最大值来实现
class MaxPool2D:
    """
    2D最大池化层（包含前向和反向传播）
    逻辑：
    前向传播时，这个窗口的最大值被选中，其他值被忽略
    反向传播时，梯度只传给最大值位置
    如果多个位置都是最大值（mask_sum>1），则梯度平均分配给这些位置
    """

    def __init__(self, kernel_size: int = 2, stride: int = 2):
        self.kernel_size = kernel_size  # 存储池化窗口大小
        self.stride = stride  # 存储步长
        self.input = None  # 用于在正向传播时保存输入，以便在反向传播时使用
        self.mask = None  # 用于记录正向传播时每个池化窗口最大值的位置（即哪些位置的输入是最大值，以便反向传播时将梯度传回正确的位置）

    # 前向传播就是对输入进行下采样，取每个窗口的最大值
    def forward(self, x: np.ndarray) -> np.ndarray:

    # 获取输入形状并计算输出形状
        batch_size, C, H, W = x.shape
        H_out = (H - self.kernel_size) // self.stride + 1
        W_out = (W - self.kernel_size) // self.stride + 1

        # output：用来存放池化结果（每个窗口的最大值）
        output = np.zeros((batch_size, C, H_out, W_out))
        # self.mask：用来记录输入x中哪些位置的值被选中作为最大值。它的形状和输入x一模一样（np.zeros_like(x)）
        # 在后面的池化计算中，当从一个小窗口选出最大值时就在self.mask中对应最大值的位置上标记1，这样反向传播时就知道梯度应该传给谁
        self.mask = np.zeros_like(x)

    # 循环是基于输出尺寸进行的，而不是直接遍历输入
    # 原因：
    #    一对一映射：每个输出位置对应一个池化窗口
    #    确定窗口位置：通过输出位置可以唯一确定输入中的窗口位置
    #    简化逻辑：直接处理输出位置比在输入上滑动窗口更直观
    # 池化计算（四重循环）
        for b in range(batch_size): #遍历每张图片
            for c in range(C): #遍历每个通道
                for i in range(H_out): #遍历输出高度
                    for j in range(W_out): #遍历输出宽度

                        #计算窗口位置
                        h_start = i * self.stride
                        w_start = j * self.stride
                        h_end = h_start + self.kernel_size
                        w_end = w_start + self.kernel_size

                        # 提取窗口内的数据
                        patch = x[b, c, h_start:h_end, w_start:w_end]

                        # 找到最大值
                        max_val = np.max(patch)
                        output[b, c, i, j] = max_val

                        # 记录最大值位置
                        max_pos = np.where(patch == max_val) # 找到最大值在patch中的位置，注意：可能有多个位置等于最大值（概率很小，但可能发生）
                        # np.where(condition)返回一个包含两个数组的元组，第一个数组是满足条件的行索引，第二个数组是列索引
                        if len(max_pos[0]) > 0:
                            self.mask[b, c, h_start + max_pos[0][0], w_start + max_pos[1][0]] = 1 # 取第一个最大值的位置（max_pos[0][0]和max_pos[1][0]）将mask中对应位置设为1

        # 保存输入x，反向传播时会用到
        # 返回池化后的结果
        self.input = x
        return output

    # 将梯度dout传回到前向传播时被选为最大值的位置
    # 由于在前向传播中，每个输出值只依赖于其对应窗口中的最大值，因此反向传播时，该输出位置的梯度（dout中的对应位置）只能传递给前向传播中被选为最大值的输入位置
    # 换句话说，只有那些在前向传播中贡献了最大值的输入位置才会获得梯度，其他位置因为在前向传播中没有贡献（被忽略）所以梯度为0
    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        反向传播：将梯度传回到最大值的位置
        """
        dx = np.zeros_like(self.input)
        batch_size, C, H_out, W_out = dout.shape

        # 遍历输出dout的每个位置
        # 每个输出位置对应前向传播中的一个池化窗口
        for b in range(batch_size):
            for c in range(C):
                for i in range(H_out):
                    for j in range(W_out):
                        # 和正向传播一样，计算当前输出位置对应的输入窗口
                        h_start = i * self.stride
                        w_start = j * self.stride
                        h_end = h_start + self.kernel_size
                        w_end = w_start + self.kernel_size

                        # 将dout的值放到最大值的位置
                        # patch_grad：输入梯度中当前窗口的部分
                        # patch_mask：前向传播时记录的mask中当前窗口的部分
                        # 前向传播中的self.mask是用来记录输入x中哪些位置的值被选中作为最大值，它的形状和输入x一模一样（np.zeros_like(x)）
                        patch_grad = dx[b, c, h_start:h_end, w_start:w_end] #获取视图
                        patch_mask = self.mask[b, c, h_start:h_end, w_start:w_end] #获取mask

                        # 视图（View）是NumPy中一种重要的概念，它允许不同的数组对象共享同一块数据内存
                        # 视图看起来像一个独立的数组，但实际上它只是原始数组数据的一个特定查看方式。它可以通过多种方式创建，最常见的是通过切片操作

                        # 如果有多个最大值位置，平均分配梯度
                        mask_sum = np.sum(patch_mask)
                        if mask_sum > 0:
                            patch_grad += patch_mask * (dout[b, c, i, j] / mask_sum) #修改视图就是修改原数组
        # 将累积的梯度dx返回给上一层
        return dx

    def __call__(self, x):
        return self.forward(x)

# ReLU：正数原样通过，负数变为0
class ReLU:
    """ReLU激活函数（包含前向和反向传播）"""

    def __init__(self):
        # 保存输入值，反向传播时需要
        self.input = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        return np.maximum(0, x) #对数组中的每个元素取max(0, 元素值)
        #eg: 输入[-2, -1, 0, 1, 2] → 输出[0, 0, 0, 1, 2]

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        反向传播：对于输入>0的位置，梯度为dout；否则为0
        """
        dx = dout * (self.input > 0)
        # 计算梯度掩码：self.input > 0, 返回一个布尔数组，True表示该位置输入 > 0 (True在计算中会被当作1，False当作0)
        # 应用掩码：dout * (self.input > 0), 将来自上一层的梯度dout乘以掩码.只让输入 > 0的位置保留梯度
        return dx

    def __call__(self, x):
        return self.forward(x)


# 每个输入节点都与每个输出节点相连。它主要完成的是矩阵乘法运算：y = Wx + b
class Linear:
    """
    全连接层（包含前向和反向传播）
    全连接层通常用于网络的最后几层，将学习到的特征进行组合并输出到输出层
    在卷积神经网络中，卷积层和池化层提取特征后，会通过全连接层进行分类或回归
    与卷积层的区别：
    全连接层每个输入节点都与输出节点相连，参数量大，且不考虑空间信息（输入被展平为一维）
    卷积层则通过卷积核在空间上共享参数，参数量小，且保留空间信息
    """

    def __init__(self, in_features: int, out_features: int):
        # in_features: 输入特征的数量（即输入向量的维度）
        # out_features: 输出特征的数量（即输出向量的维度）
        self.in_features = in_features
        self.out_features = out_features

        # scale来缩放随机初始化的权重，使得权重的方差控制在1/in_features，从而保持前向和反向传播中信号的方差稳定，避免梯度消失或爆炸
        scale = np.sqrt(1.0 / in_features)

        # W: 权重矩阵，形状为(out_features, in_features)
        # b: 偏置向量，形状为(out_features, 1）
        # 为什么权重矩阵可以随机初始化？
        # 在训练神经网络之前，因为不知道权重应该取什么值，所以通常用随机数来初始化
        # 这是为了打破对称性，如果所有权重都初始化为相同的值，那么每个神经元都会计算出相同的输出，在反向传播时也会得到相同的梯度，从而导致所有神经元的更新相同，不利于学习多样化的特征
        # 为什么W的形状是[out_features, in_features]？
        # 神经网络全连接层的公式是：y = x·W ^ T + b
        # 其中：     x形状：[batch_size, in_features]        W形状：[out_features, in_features]
        #           W ^ T形状：[in_features, out_features]（转置后）      y形状：[batch_size, out_features]
        #         注：x和y的形状在给定输入和层参数的情况下是固定的，它们分别表示一批样本的输入特征和输出特征
        # 矩阵乘法规则：(行, 列) × (行, 列) = (行, 列)    [batch, in] × [ in, out] = [batch, out]
        # 所以需要W ^ T的形状是[in_features, out_features]，那么原始W的形状就是[out_features, in_features]
        self.W = np.random.randn(out_features, in_features) * scale
        # b的形状：[out_features, 1]是一个列向量，每个输出特征对应一个偏置。通过转置和广播，加到每个样本的每个输出特征上
        self.b = np.zeros((out_features, 1))
        # 这样的设计使得全连接层可以方便地使用矩阵乘法进行批量计算，同时保持梯度的正确传递

        # input: 保存输入，用于反向传播
        # output: 保存输出，用于反向传播（但在这个实现中，反向传播没有用到output，而是用到了input）
        # dW, db: 权重的梯度和偏置的梯度
        self.input = None
        self.output = None
        self.dW = None
        self.db = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        x shape: (batch_size, in_features)
        """
        self.input = x
        # 矩阵乘法 + 偏置
        # 步骤：
        # 1.保存输入（用于反向传播）
        # 2.计算输出：np.dot(x, W.T) + b.T  这里b.T将偏置向量从(out_features, 1)转换为(1, out_features)，然后通过广播加到每个样本上
        self.output = np.dot(x, self.W.T) + self.b.T
        return self.output

    def backward(self, dout: np.ndarray, lr: float = 0.001) -> np.ndarray:
        """
        dout shape: (batch_size, out_features)
        """

        # 计算梯度

        # 对权重W的梯度：dW = dout ^ T · x
        # 简单说：权重的梯度 = 输入特征值 × 输出误差信号，并对所有样本求和。
        # 注意这里dout形状为(batch, out_features)，转置后为(out_features, batch)，与x(batch, in_features)相乘，得到(out_features, in_features)，正好与W形状相同
        # 所以，dW的计算用到了input（即x），是因为在前向传播中，x与W进行了矩阵乘法，每个输出都是x和W的每一行的点积
        # 因此，在反向传播时，每个权重W[j, k]的梯度自然与每个样本的输入x_{i, k}和该样本对应的输出误差dout_{i, j}有关
        # eg：想象一个灌溉系统： x = 水源（每个水源的水量不同）  W = 水管和阀门（控制每个水源流向哪里）  y = 农田的水量
        #     一个水管（W[j,i]）的调整，需要知道：
        #       1.农田缺水多少（dout[j]）
        #       2.水源有多少水（x[i]）
        #       3.不是看水管本身，而是看水管两端的"供需关系"
        self.dW = np.dot(dout.T, self.input)

        # 对偏置b的梯度：db = 对dout在batch维度上求和（因为b被加到每个样本上，所以梯度需要累加）
        # 偏置b是一个向量，每个输出特征j对应一个偏置b[j]
        # 在前向传播中，b[j]被加到了每个样本的第j个输出上。所以，每个样本的第j个输出的误差信号（dout[k, j]）都会影响b[j]
        # 因此，需要将所有样本在第j个输出上的误差信号加起来，作为b[j]的梯度
        # 简单说：偏置的梯度 = 输出误差信号，对所有样本求和
        # eg：工资计算：工资 = 基础工资 + 抽成    设基础工资1000
        #     小A应得 2000，实际得1800 ——— 误差 = +200
        #     小B应得 2000，实际得1900 ——— 误差 = +100
        #     小C应得 2000，实际得2180 ——— 误差 = -180
        # 基础工资应该如何调整？
        # 总误差=200+100-180=120 ➡️ 平均误差=120/3=40
        # 结论：基础工资过低，应上调40
        self.db = np.sum(dout, axis=0, keepdims=True).T  # (out_features, 1)
        # axis: 求和的方向。axis=0表示沿着第0维（即batch维度）求和，即对每个输出特征，把batch中所有样本的梯度加起来
        # keepdims: 是否保持维度。如果为True，则输出维度与输入维度相同，只是指定axis的长度为1。这里设为True，那么求和后的形状是(1, out_features)

        # 对输入x的梯度：dx = dout · W
        # 输入x的每个特征i通过权重矩阵W连接到所有输出特征j
        # 所以，当知道每个输出j的误差信号（dout的第j列）时，就可以通过权重矩阵 W 将这些误差信号“分配”回输入特征i
        # 具体来说，对于第i个输入特征，它的梯度是每个输出j的误差信号乘以对应的权重W[j, i]，然后对所有输出j求和
        # 简单说：输入的梯度 = 输出误差信号乘以权重矩阵，这样就知道每个输入特征应该承担多少责任
        # eg：团队项目的责任追究
        # x = 各部门的投入（人力资源、资金、设备）   W = 各部门对各个任务的影响力权重
        # y = 各个任务的完成情况    dout = 各个任务没完成好（误差）
        # 问题：任务A没完成好(dout[A]为负)，是哪个部门该负责？
        # 看W矩阵：  - 部门1对任务A的影响权重W[A, 部门1] = 0.8（很大）
        #           - 部门2对任务A的影响权重W[A, 部门2] = 0.1（很小）
        # 结论：部门1的责任更大
        dx = np.dot(dout, self.W)  # (batch_size, in_features)

        # 梯度裁剪
        # （将dW和db限制在[-1, 1]之间），这是为了防止梯度爆炸
        self.dW = np.clip(self.dW, -1.0, 1.0)
        self.db = np.clip(self.db, -1.0, 1.0)

        # 使用梯度下降更新参数
        self.W -= lr * self.dW
        self.b -= lr * self.db

        # 返回对输入x的梯度dx，以便继续向上一层反向传播
        return dx

    def __call__(self, x):
        return self.forward(x)


class Flatten:
    """
    展平层（包含前向和反向传播）
Flatten层是CNN中的必要桥梁：
功能简单：多维 → 一维
没有参数：不需要学习
关键作用：连接卷积层和全连接层
    """

    def __init__(self):
        self.input_shape = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input_shape = x.shape  #保存原始形状
        # eg：假设输入x的形状为 (32, 3, 28, 28)，即32个样本，每个样本是3通道、28x28的图像
        # 执行 batch_size = x.shape[0] 得到 batch_size = 32
        # 然后执行 x.reshape(batch_size, -1)，即 x.reshape(32, -1)，将得到形状为 (32, 3*28*28) = (32, 2352) 的数组
        batch_size = x.shape[0]  #获取批大小
        # new_array = old_array.reshape(shape)
        # 其中，shape是一个整数元组，指定新数组的形状。元组中的每个数字表示对应维度的大小
        # 特殊的是，在shape中可以使用-1，它表示该维度的大小将由数组的总元素个数和其他维度的大小自动推断出来
        return x.reshape(batch_size, -1)  #展平

    def backward(self, dout: np.ndarray) -> np.ndarray:
        # 输入：来自全连接层的梯度 dout
        # 形状：(batch_size, 展平后的特征数)
        # 输出：恢复成卷积层的形状
        # 形状：与原始输入相同 self.input_shape
        # reshape操作是可逆的线性变换，它的导数就是单位映射（重新排列）
        return dout.reshape(self.input_shape)

    def __call__(self, x):
        return self.forward(x)

# Dropout（随机失活层），这是神经网络中最重要的正则化技术之一，用于防止过拟合
class Dropout:
    """
    Dropout层（包含前向和反向传播）
    在前向传播时，随机将一些神经元的输出设置为0（即丢弃），在反向传播时，这些被丢弃的神经元不参与梯度的计算
    注意：Dropout只在训练时使用，在测试时不应该使用Dropout
    """

    def __init__(self, p: float = 0.5):
        ## p：丢弃概率（dropout rate）
        # 取值范围：0.0到1.0 ， 默认值：0.5 ， 表示每个神经元被"丢弃"的概率
        # 实际保留概率 = 1 - p
        ## mask：掩码矩阵
        # 记录哪些神经元被保留（1），哪些被丢弃（0） ， 形状与输入相同，反向传播时需要
        ## training：训练模式标志
        # True：训练模式，使用Dropout
        # False：评估 / 测试模式，不使用Dropout
        self.p = p #丢弃概率
        self.mask = None #记录哪些神经元被丢弃
        self.training = True #训练模式标志

    def forward(self, x: np.ndarray) -> np.ndarray:
        if self.training and self.p > 0:   # 只有在训练模式且p>0时才应用Dropout
            self.mask = (np.random.rand(*x.shape) > self.p) / (1 - self.p)
            # 步骤1：np.random.rand(*x.shape)
            # 生成与输入x形状相同的随机矩阵，每个元素是[0, 1)之间的均匀随机数
            # eg：如果x.shape = (2, 3)，则生成2×3的随机矩阵
            # 步骤2： > self.p
            # 比较每个随机数是否大于p，结果是一个布尔矩阵，元素为True的概率 = 1 - p（保留概率）
            # 步骤3： / (1 - self.p)
            # 将布尔矩阵转换为数值矩阵（True = 1, False = 0），然后除以(1 - p)
            # 为什么要除以(1 - p)？ 这是缩放（inverted dropout）技巧，在训练时对保留的神经元进行放大，在测试时就可以直接使用原始值

            #应用mask
            # 将输入x乘以mask，被丢弃的神经元输出为0，保留的神经元输出被放大
            return x * self.mask

        return x

    def backward(self, dout: np.ndarray) -> np.ndarray:
        if self.training and self.p > 0:
            return dout * self.mask
        return dout

    #训练和测试模式
    # 训练模式：使用Dropout，随机丢弃神经元，输出按比例放大
    # 测试模式：不使用Dropout，所有神经元都工作，输出不做特殊处理
    def train(self):
        self.training = True
    def eval(self):
        self.training = False

    def __call__(self, x):
        return self.forward(x)


# ==================== 损失函数 ====================
# 交叉熵损失通常用于多分类问题，与softmax函数结合使用，衡量模型预测与真实标签之间的差距。
class CrossEntropyLoss:
    """交叉熵损失函数（包含前向和反向传播）"""

    def __init__(self):
        self.probs = None #保存 softmax概率
        self.labels = None #保存 one-hot编码的标签

    def forward(self, logits: np.ndarray, labels: np.ndarray) -> float:
        """
        logits: (batch_size, num_classes)   # 未归一化的预测值
        labels: (batch_size,) 或 (batch_size, num_classes)  # 真实标签
        """
        batch_size = logits.shape[0]

        # 数值稳定性的softmax

        # 找到每个样本（每行）的最大值，所有值减去这个最大值
        # 结果：每行最大值变为0，所有值 ≤ 0。可以  1.防止exp溢出：exp(≤0) ≤ 1，不会爆炸  2.保持结果不变：softmax平移不变性  3.数值稳定：避免NaN，保证训练顺利进行
        # axis=1：按行(类别)操作      keepdims=True：保持维度(方便后续广播)
        logits_shifted = logits - np.max(logits, axis=1, keepdims=True)
        # 对减去最大值后的logits取指数函数
        exp_logits = np.exp(logits_shifted)
        # 把"相对可能性"变成"真正的概率"，保证每个样本的所有类别概率加起来等于1
        # eg：
        # 三个类别的exp_logits = np.array([9.97, 6.05, 1.65])
        # # 直接看数字：猫=9.97，狗=6.05，鸟=1.65 ➡️不是概率，相加不等于1
        # 解决方案：除以总和（归一化）  probs = exp_logits / sum_exp
        # # [9.97/17.67, 6.05/17.67, 1.65/17.67]
        # # = [0.564, 0.342, 0.094]  ← 相加等于1
        self.probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        # 将标签转换为one-hot编码，并保存到self.labels中
        # 判断标签的维度是否为1。如果是1维，则假设是整数标签，每个元素是一个类别的索引
        if labels.ndim == 1:
            # 创建一个与self.probs形状相同的全零数组。self.probs是在前一步通过softmax计算得到的，形状为(batch_size, num_classes)
            self.labels = np.zeros_like(self.probs)
            # 使用NumPy的高级索引（花式索引）来同时给多个位置赋值
            # self.labels是一个形状为(batch_size, num_classes)的二维数组，初始化为全0
            # 然后有两个索引数组： 1.  np.arange(batch_size)，生成一个从0到batch_size-1的整数数组，表示每个样本的行索引
            #                   2.  labels，它是一个一维数组，长度为batch_size，每个元素是一个整数，表示该样本的类别标签（即列索引）
            # self.labels[np.arange(batch_size), labels]选取这些位置：对于第i个样本（行索引为i），列索引为labels[i]。然后给这些选中的位置赋值为1
            # eg：
            # 假设batch_size=3，num_classes=5，labels=[2, 0, 4]（三个样本的标签分别是2,0,4），那么np.arange(batch_size)就是[0,1,2]
            # 索引组合为：  第一个样本：行索引0，列索引2 -> 位置(0,2)
            #             第二个样本：行索引1，列索引0 -> 位置(1,0)
            #             第三个样本：行索引2，列索引4 -> 位置(2,4)
            # 然后，将这些位置设置为1,最终的self.labels数组为：  [[0, 0, 1, 0, 0],
            #                                               [1, 0, 0, 0, 0],
            #                                               [0, 0, 0, 0, 1]]
            self.labels[np.arange(batch_size), labels] = 1
        # 如果标签的维度不是1（通常是2维，即已经是one-hot编码），则直接保存labels
        else:
            self.labels = labels

        #为什么要进行编码转换？
        # 通常训练数据中，每个样本的标签可能是整数，比如0代表猫，1代表狗。但在计算交叉熵损失时，需要将标签转换为one-hot编码，即每个标签变成一个向量，向量的长度等于类别数，在对应类别的位置为1，其余为0
        # eg：类别数=3
        # 整数标签0 -> one-hot: [1, 0, 0]
        # 整数标签1 -> one-hot: [0, 1, 0]
        # 整数标签2 -> one-hot: [0, 0, 1]
        # 为什么要这样做呢？因为交叉熵损失的计算公式是：-Σ (真实标签_i * log(预测概率_i))，其中真实标签需要是概率分布（在真实类别上为1，其他为0）。预测概率是模型输出的softmax概率
        # 所以，如果标签是整数，就需要将其转换为one-hot编码，才能与预测概率进行逐元素相乘，然后求和得到损失

        # 计算交叉熵损失
        # 计算log时不能有0，加一个极小值eps，确保永远不会出现log(0)的情况
        eps = 1e-15
        ## 最内层 - self.probs + eps
        # eg： probs = [0.9, 0.1, 0.0]
        # 加上eps（0.000000000000001）➡️结果：[0.900000000000001, 0.100000000000001, 0.000000000000001]
        # 如果概率是0.0，log(0.0) = 负无穷大 ❌ ；加上eps后，log(0.000000000000001) = -34.5 ✅
        ## 取对数 - np.log(safe_probs)
        # eg：log(0.900000000000001) = -0.105 ；log(0.000000000000001) = -34.5 ； log(0.000000000000001) = -34.5 ➡️ 结果：[-0.105, -34.5, -34.5]
        # 概率范围是0到1，log会把它们映射到负无穷到0，损失需要衡量"犯错程度"：预测越错，损失越大
        # 真实概率是1.0，你预测0.1 → log(0.1) = -2.3（大错误）；真实概率是1.0，你预测0.9 → log(0.9) = -0.105（小错误）
        ## 外乘法过滤 - self.labels * log_probs
        # eg： labels = [1, 0, 0] （one-hot编码，真实标签是第0类）； log_probs = [-0.105, -34.5, -34.5]
        # 计算过程 [1×(-0.105), 0×(-34.5), 0×(-34.5)] ➡️ 结果：[-0.105, 0, 0]
        # 只需关心真实类别的预测概率  1 × 值 = 保留该值（真实类别的log概率）  0 × 值 = 变成0（其他类别的log概率被丢弃）
        loss = -np.sum(self.labels * np.log(self.probs + eps)) / batch_size
        return loss

    def backward(self) -> np.ndarray:
        """
        返回梯度: dL/dlogits = (probs - labels) / batch_size
        """
        batch_size = self.probs.shape[0]
        # 预测概率 - 真实标签 = 梯度
        return (self.probs - self.labels) / batch_size

    def __call__(self, logits, labels):
        return self.forward(logits, labels)


# ==================== 两层VGG网络 ====================

class TwoLayerVGG:
    """
    两层VGG风格的网络
    结构: Conv1 -> ReLU -> MaxPool -> Conv2 -> ReLU -> MaxPool -> Flatten -> FC1 -> ReLU -> FC2
    """

    def __init__(self, input_channels: int = 3, num_classes: int = 10, dropout_rate: float = 0.3):

        # 参数设置问题
        # input_channels：取决于输入图像（如RGB为3，灰度图为1）
        # num_classes：取决于分类任务
        # dropout_rate：超参数，可以调整
        # 卷积层的输出通道数（如16、32）：超参数，可以调整，但通常使用2的幂次，且随着网络加深逐渐增加
        # 卷积核大小：这里固定为3，是VGG风格，也可以调整但需要保证前后尺寸匹配
        # 全连接层的神经元数（如128）：超参数，可以调整
        # 全连接层输入维度（如32*8*8）：这个不能随意调整，它是由输入图像尺寸和网络结构决定的。如果输入尺寸改变，这里必须重新计算

        # 卷积部分
        # input_channels: 输入通道数（彩色图=3，灰度图=1）
        # 16: 输出通道数（16个不同的特征检测器）
        # kernel_size=3: 3×3的卷积核（VGG标准）
        # padding=1: 边缘补1圈零，保持尺寸不变
        # 效果：输入(W,H,3) → 输出(W,H,16)
        self.conv1 = Conv2D(input_channels, 16, kernel_size=3, padding=1)
        # 激活函数：f(x) = max(0, x)
        # 作用：引入非线性，让网络能学习复杂模式
        self.relu1 = ReLU()
        # 最大池化：2×2窗口，步长2
        # 效果：尺寸减半 (W,H) → (W/2, H/2)
        self.pool1 = MaxPool2D(kernel_size=2, stride=2)

        self.conv2 = Conv2D(16, 32, kernel_size=3, padding=1)
        self.relu2 = ReLU()
        self.pool2 = MaxPool2D(kernel_size=2, stride=2)

        # 全连接部分
        # 把多维特征图"摊平"成一维向量
        self.flatten = Flatten()
        # 全连接层：2048个输入 → 128个神经元
        # 32×8×8的计算前提：假设输入是32×32的图片
        # 经过两次池化：32×32 → 16×16 → 8×8
        self.fc1 = Linear(32 * 8 * 8, 128)
        self.relu3 = ReLU()
        # 输出层：128个神经元 → num_classes个类别
        self.fc2 = Linear(128, num_classes)

        # Dropout（降低dropout率避免欠拟合）
        self.dropout = Dropout(p=dropout_rate)

        # 损失函数
        self.criterion = CrossEntropyLoss()

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """前向传播"""
                             # eg：输入x形状(batch_size, 3, 32, 32)

        # 卷积部分
        x = self.conv1(x)    # (batch_size, 3, 32, 32) → (batch_size, 16, 32, 32)
        x = self.relu1(x)    # 负值变0，正值保留
        x = self.pool1(x)    # (batch_size, 16, 32, 32) → (batch_size, 16, 16, 16)

        x = self.conv2(x)    # (batch_size, 16, 16, 16) → (batch_size, 32, 16, 16)
        x = self.relu2(x)    # 激活
        x = self.pool2(x)    # (batch_size, 32, 16, 16) → (batch_size, 32, 8, 8)

        # 全连接部分
        x = self.flatten(x)  # (batch_size, 32, 8, 8) → (batch_size, 2048)
        x = self.fc1(x)      # (batch_size, 2048) → (batch_size, 128)
        x = self.relu3(x)    # 激活

        if training:
            x = self.dropout(x)  # 随机丢弃部分神经元，eg：128 → 约90个活跃

        x = self.fc2(x)          # (batch_size, 128) → (batch_size, num_classes)

        return x

    def backward(self, dout: np.ndarray, lr: float = 0.001) -> None:
        """
        反向传播
        依次调用各层的backward方法，每一层都会接收从后面层传来的梯度（即损失函数对该层输出的梯度），
        然后计算 1.损失函数对该层参数的梯度（即对权重W和偏置b的梯度），并用学习率lr更新这些参数
                2.损失函数对该层输入的梯度（上一层的输出梯度），并将这个梯度返回，以便传递给前一层
        （也可以理解为：每一层其实都是在用本层的输出的梯度更新参数并计算上一层的输出梯度）
        所以，在反向传播中一直在更新dout变量，实际上是在沿着网络反向传递梯度，直到第一层。
        """

        # 学习率传入问题
        # 全连接层（Linear）和卷积层（Conv2D）有可训练参数（权重和偏置），所以它们的backward方法需要学习率lr来更新参数
        # 而激活函数（ReLU）、池化层（MaxPool2D）、Dropout、Flatten等层没有可训练参数，它们只是进行数据的变换，所以反向传播时只需要传递梯度dout，不需要学习率

        # 全连接部分反向传播（反向传播顺序：从后往前，与forward相反）
        dout = self.fc2.backward(dout, lr)     # 1. 更新fc2的权重
        dout = self.dropout.backward(dout)     # 2. 传递梯度时考虑dropout
        dout = self.relu3.backward(dout)       # 3. 只通过正向时>0的路径
        dout = self.fc1.backward(dout, lr)     # 4. 更新fc1的权重
        dout = self.flatten.backward(dout)     # 5. 重塑梯度形状

        # 卷积部分反向传播（ 现在dout形状: batch_size, 32, 8, 8 ）
        dout = self.pool2.backward(dout)       # 6. 反向最大池化
        dout = self.relu2.backward(dout)       # 7. 只通过正向时>0的路径
        dout = self.conv2.backward(dout, lr)   # 8. 更新conv2的权重
        dout = self.pool1.backward(dout)       # 9. 反向最大池化
        dout = self.relu1.backward(dout)       # 10. 只通过正向时>0的路径
        # dout最后是损失函数对网络输入的梯度，这个梯度在训练中通常不会使用（目的是更新网络中的参数，而不是更新输入数据）所以用下划线_来接收，表示不需要它
        _ = self.conv1.backward(dout, lr)      # 11. 更新conv1的权重

    def compute_loss(self, logits: np.ndarray, labels: np.ndarray):
        """计算损失和梯度"""
        loss = self.criterion(logits, labels)
        dout = self.criterion.backward()
        return loss, dout

    def train_step(self, x: np.ndarray, y: np.ndarray, lr: float = 0.001) -> float:
        """
        单步训练
        训练循环中的一个基本单元，负责执行一次前向传播、计算损失和梯度、反向传播更新参数，并返回损失值
        """
        # 前向传播
        logits = self.forward(x, training=True)

        # 计算损失和梯度
        loss, dout = self.compute_loss(logits, y)

        # 反向传播
        self.backward(dout, lr)

        return loss

    def predict(self, x: np.ndarray) -> np.ndarray:
        """预测：在训练好的模型上进行预测，并输出每个类别的概率"""
        logits = self.forward(x, training=False)
        # softmax得到概率
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        return probs

    def get_accuracy(self, x: np.ndarray, y: np.ndarray) -> float:
        """计算准确率"""
        #  得到预测值x概率分布
        #  eg：probs = [[0.9, 0.1, 0.0],  # 样本1：90%是类别0
        #             [0.3, 0.6, 0.1],  # 样本2：60%是类别1
        #             [0.1, 0.2, 0.7],  # 样本3：70%是类别2
        #             [0.4, 0.4, 0.2]]  # 样本4：两个类别都是40%
        probs = self.predict(x)
        # 找到每个样本概率最大的类别
        # axis=1：沿着类别维度找最大值的位置
        # eg：对于上面的probs：predictions = [0, 1, 2, 0]
        # 样本1：最大概率0.9在位置0 → 预测类别0    样本2：最大概率0.6在位置1 → 预测类别1    样本3：最大概率0.7在位置2 → 预测类别2    样本4：最大概率0.4在位置0和1，取第一个0 → 预测类别0
        predictions = np.argmax(probs, axis=1)

        # 处理真实标签y，可能是两种格式：
        # 1. 整数标签：[2, 0, 1, 2] ← 直接使用
        # 2. one-hot编码：[[0,0,1], [1,0,0], [0,1,0], [0,0,1]] ← 需要转换
        if y.ndim > 1:   # 如果是二维数组（one-hot）
            y_labels = np.argmax(y, axis=1)   # 找到每行中1的位置
        else:   # 如果是一维数组（整数标签）
            y_labels = y   # 直接使用

        # 1. predictions == y_labels：比较每个样本的预测和真实标签，返回布尔数组：[True, False, True, True]
        # 2. np.mean()：计算True的比例（True=1, False=0）
        accuracy = np.mean(predictions == y_labels)
        return accuracy

    def train(self):
        """设置为训练模式"""
        self.dropout.train()

    def eval(self):
        """设置为评估模式"""
        self.dropout.eval()


# ==================== 数据生成器（共1000数据，此部分由ds生成可以跳看） ====================

def generate_balanced_dataset(num_samples: int = 1000, img_size: int = 32,
                              num_classes: int = 10, channels: int = 3):
    """生成平衡且可学习的模拟数据集"""
    X = np.random.randn(num_samples, channels, img_size, img_size).astype(np.float32) * 0.2

    for i in range(num_samples):
        class_idx = i % num_classes

        # 每个类别有独特的模式，但不要过于简单
        if class_idx == 0:  # 左上角方块
            X[i, :, :5, :5] += 0.5
        elif class_idx == 1:  # 右上角方块
            X[i, :, :5, -5:] += 0.5
        elif class_idx == 2:  # 左下角方块
            X[i, :, -5:, :5] += 0.5
        elif class_idx == 3:  # 右下角方块
            X[i, :, -5:, -5:] += 0.5
        elif class_idx == 4:  # 中心横线
            X[i, :, img_size // 2 - 1:img_size // 2 + 1, :] += 0.4
        elif class_idx == 5:  # 中心竖线
            X[i, :, :, img_size // 2 - 1:img_size // 2 + 1] += 0.4
        elif class_idx == 6:  # 对角线1
            for d in range(img_size):
                X[i, :, d, d] += 0.3
        elif class_idx == 7:  # 对角线2
            for d in range(img_size):
                X[i, :, d, img_size - 1 - d] += 0.3
        elif class_idx == 8:  # 四角点
            X[i, :, 0, 0] += 0.6
            X[i, :, 0, -1] += 0.6
            X[i, :, -1, 0] += 0.6
            X[i, :, -1, -1] += 0.6
        else:  # 中心圆点
            center_h, center_w = img_size // 2, img_size // 2
            for h in range(max(0, center_h - 3), min(img_size, center_h + 4)):
                for w in range(max(0, center_w - 3), min(img_size, center_w + 4)):
                    if (h - center_h) ** 2 + (w - center_w) ** 2 <= 9:
                        X[i, :, h, w] += 0.4

        # 添加适度的噪声
        noise = np.random.randn(channels, img_size, img_size) * 0.1
        X[i] += noise

        # 随机亮度调整
        brightness = np.random.uniform(-0.05, 0.05)
        X[i] += brightness

    # 生成标签
    y = np.array([i % num_classes for i in range(num_samples)])

    # 打乱数据
    indices = np.random.permutation(num_samples)
    X = X[indices]
    y = y[indices]

    return X, y

def split_dataset(X, y, train_ratio=0.7, val_ratio=0.15):
    """分割数据集"""
    num_samples = X.shape[0]
    train_size = int(num_samples * train_ratio)
    val_size = int(num_samples * val_ratio)

    X_train, X_val, X_test = X[:train_size], X[train_size:train_size + val_size], X[train_size + val_size:]
    y_train, y_val, y_test = y[:train_size], y[train_size:train_size + val_size], y[train_size + val_size:]

    return X_train, X_val, X_test, y_train, y_val, y_test



# ==================== 训练循环 ====================

def train_model(model, X_train, y_train, X_val, y_val,
                epochs=10, batch_size=32, lr=0.001, verbose=True):
    # model: 要训练的神经网络模型（如TwoLayerVGG）
    # X_train, y_train: 训练数据（图片和标签）
    # X_val, y_val: 验证数据（用于监控训练效果）
    # epochs: 训练轮数（整个数据集过多少遍）
    # batch_size: 批次大小（一次处理多少样本）
    # lr: 学习率（参数更新的步伐大小）
    # verbose: 是否打印训练信息
    """训练模型"""
    num_samples = X_train.shape[0] #训练样本总数
    num_batches = num_samples // batch_size  #批次数

    train_losses = []  #记录每轮的训练损失
    val_accuracies = []  #记录每轮的验证准确率

    # 1个epoch = 整个训练集被模型学习一次
    # 如果训练10个epochs，模型就看训练集10遍
    for epoch in range(epochs):
        # 为什么要切换模式？
        # 训练时：需要正则化防止过拟合
        # 评估时：要得到稳定的最佳表现

        model.train()  # 设置为训练模式
        epoch_loss = 0
        # 随机打乱数据
        # np.random.permutation(n)：生成一个0到n - 1的随机排列数组。
        # 数组索引：使用一个索引数组来重新排列另一个数组，eg：arr[indices]会按照indices的顺序重新排列arr
        # 为什么生成数据集时已经打乱还需要再次打乱？
        # 生成数据集时的打乱是为了避免原始数据可能存在的顺序（比如前100个都是类别0，接着100个是类别1等），而每个epoch开始时的打乱是为了让每个epoch的数据顺序都不同
        indices = np.random.permutation(num_samples)
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]
        # 批次训练
        # 将整个训练数据分成多个小批次（mini-batches），然后对每个小批次执行一次训练步骤
        # 循环遍历所有批次
        for batch_idx in range(num_batches):
            # 计算当前批次的起始和结束索引
            start = batch_idx * batch_size
            end = start + batch_size
            # 从打乱后的数据中取出当前批次的数据（X_batch和y_batch）
            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]
            # 训练步骤
            loss = model.train_step(X_batch, y_batch, lr)
            epoch_loss += loss
        # 计算平均损失
        # max(num_batches, 1)防止除以0
        avg_loss = epoch_loss / max(num_batches, 1)


        # 验证
        # 在每个epoch结束后，模型会切换到评估模式，然后在验证集上计算准确率
        model.eval()  # 设置为评估模式
        val_accuracy = model.get_accuracy(X_val, y_val)

        # 记录训练损失和验证准确率
        train_losses.append(avg_loss)
        val_accuracies.append(val_accuracy)

        # 打印每轮结果
        if verbose:
            print(f"Epoch {epoch + 1:2d}/{epochs}: Loss={avg_loss:.4f}, Val Accuracy={val_accuracy:.4f}")

    return train_losses, val_accuracies


# ==================== 主程序 ====================
def main():

    np.random.seed(42)

    print("生成数据...")
    X, y = generate_balanced_dataset(1000, 32, 10, 3)
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y)

    print("创建模型...")
    model = TwoLayerVGG(3, 10, 0.3)

    print("开始训练...")
    train_losses, val_accuracies = train_model(
        model, X_train, y_train, X_val, y_val,
        epochs=10, batch_size=32, lr=0.001, verbose=True
    )

    # 测试
    model.eval()
    test_accuracy = model.get_accuracy(X_test, y_test)


    print(f"\n最终测试准确率: {test_accuracy:.2%}")
    print(f"最佳验证准确率: {max(val_accuracies):.2%}")

    return test_accuracy


if __name__ == "__main__":
    import numpy as np

    accuracy = main()