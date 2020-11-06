import torch
import torch.nn.functional as F

def my_cross_entropy(input,target,weight = None,reduce=True,reduction="mean"):
    # input.shape: torch.size([-1, class])
	# target.shape: torch.size([-1])
	# reduction = "mean" or "sum"
	# input是模型输出的结果，与target求loss
	# target的长度和input第一维的长度一致
	# target的元素值为目标class
	# reduction默认为mean，即对loss求均值
	# 还有另一种为sum，对loss求和
    #这里应实现一个自动降维的

    # 这里对input所有元素求exp
    exp = torch.exp(input)
    # 根据target的索引，在exp第一维取出元素值，这是softmax的分子
    tmp1 = exp.gather(1,target.unsqueeze(1)).squeeze()
    # 在exp第一维求和，这是softmax的分母
    tmp2 = exp.sum(1)
    # softmax公式：ei / sum(ej)
    softmax = tmp1/tmp2
    # cross-entropy公式： -yi * log(pi)
    # 因为target的yi为1，其余为0，所以在tmp1直接把目标拿出来，
    # 公式中的pi就是softmax的结果
    log = -torch.log(softmax)
    if weight is not None:
        log = log * weight
    # 官方实现中，reduction有mean/sum及none
    # 只是对交叉熵后处理的差别
    if not reduce:
        return log
    if reduction == "mean": return log.mean()
    elif reduction == "sum": return log.sum()
    else:
        raise NotImplementedError('unkowned reduction')

input = torch.randn(2, 5,2,2, requires_grad=True)
target = torch.randint(5, (2,2,2), dtype=torch.int64)
weight = torch.randn(2,2,2, requires_grad=True)

# input = torch.randn(3, 5, 214,214,requires_grad=True)
# target = torch.randint(5, (3,214,214), dtype=torch.int64)

loss1_mean = F.cross_entropy(input, target)
loss2_mean = my_cross_entropy(input, target)
print(loss1_mean)
print(loss2_mean)
# tensor(3.2158, grad_fn=<NllLossBackward>)
# tensor(3.2158, grad_fn=<MeanBackward0>)

loss1_sum = F.cross_entropy(input, target, reduction="sum")
loss2_sum = my_cross_entropy(input, target, reduction="sum")
print(loss1_sum)
print(loss2_sum)
# tensor(9.6475, grad_fn=<NllLossBackward>)
# tensor(9.6475, grad_fn=<SumBackward0>)

loss1_pixel = F.cross_entropy(input, target, reduce=False,reduction='mean')
loss2_pixel = my_cross_entropy(input, target, reduce=False,reduction='mean')
print(loss1_pixel)
print(loss2_pixel)

loss1_pixel = loss1_pixel*weight 
loss2_pixel = loss2_pixel*weight 
print(loss1_pixel)
print(loss2_pixel)