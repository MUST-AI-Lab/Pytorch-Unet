import matplotlib.pyplot as plt
import numpy as np

def plot_img_and_mask(img, mask):
    classes = mask.shape[2] if len(mask.shape) > 2 else 1
    fig, ax = plt.subplots(1, classes + 1)
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    if classes > 1:
        for i in range(classes):
            ax[i+1].set_title(f'Output mask (class {i+1})')
            ax[i+1].imshow(mask[:, :, i])
    else:
        ax[1].set_title(f'Output mask')
        ax[1].imshow(mask)
    plt.xticks([]), plt.yticks([])
    plt.show()


def visualize_pred_to_file(filename,x, y_true, y_pred, title1="Original", title2="True", title3="Predicted", cmap='gray'):
    fig = plt.figure()
    fig.set_figheight(15)
    fig.set_figwidth(15)

    ax = plt.subplot(1,3,1)
    ax.set_title(title1)
    if x.shape[0] !=1:
        x =  x.transpose(1, 2, 0)
        if x.max() <=1:
            x = x * 255
        ax.imshow(x.astype(int), cmap=None)
    else:
        if x.max() <=1:
            x = x * 255
        ax.imshow(np.squeeze(x).astype(int), cmap=cmap)

    ax = plt.subplot(1,3,2)
    ax.set_title(title2)
    if y_true.max() <1:
            y_true = y_true * 255
    ax.imshow(np.squeeze(y_true).astype(int), cmap=cmap)

    ax = plt.subplot(1,3,3)
    ax.set_title(title3)
    if y_pred.max() <1:
            y_pred = y_true * 255
    ax.imshow(np.squeeze(y_pred).astype(int), cmap=cmap)

    #print("save to {}".format(OUT_PATH+filename))
    plt.savefig(filename)
    plt.cla() # 清除axes，即当前 figure 中的活动的axes，但其他axes保持不变。
    plt.clf() # 清除当前 figure 的所有axes，但是不关闭这个 window，所以能继续复用于其他的 plot。
    plt.close() # 关闭 window，如果没有指定，则指当前 window。