import pickle
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.tri as tri
import cv2
import PIL.Image as Image
from tqdm import tqdm
import datetime
import glob
import os

def fig2data(fig):
    """
    fig = plt.figure()
    image = fig2data(fig)
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    
    # draw the renderer
    fig.canvas.draw()
 
    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)
 
    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    image = Image.frombytes("RGBA", (w, h), buf.tobytes())
    image = np.asarray(image)
    return image


result_files = glob.glob('result/*.pkl')
os.makedirs('videos', exist_ok=True)

for index, file in enumerate(result_files):

    with open(file, 'rb') as f:
        result, crds = pickle.load(f)
    triang = tri.Triangulation(crds[:, 0], crds[:, 1])

    file_name = 'videos/output%d.mp4'%index

    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 保存视频的编码
    out = cv2.VideoWriter(file_name, fourcc, 20.0, (1700,800))

    r_t = result[0][:, 0]

    v_max = np.max(r_t)
    v_min = np.min(r_t)

    colorbar = None
    skip=5
    
    def render(i):

        step = i*skip
        target = result[1][step]
        predicted = result[0][step]

        fig, axes = plt.subplots(2, 1, figsize=(17, 8))

        target_v = np.linalg.norm(target, axis=-1)
        predicted_v = np.linalg.norm(predicted,axis=-1)

        # diff = np.linalg.norm(target - predicted, axis=-1)
        
        for ax in axes:
            ax.cla()
            ax.triplot(triang, 'o-', color='k', ms=0.5, lw=0.3)

        handle1 = axes[0].tripcolor(triang, target_v, vmax=v_max, vmin=v_min)
        axes[1].tripcolor(triang, predicted_v, vmax=v_max, vmin=v_min)
        # handle2 = axes[2].tripcolor(triang, diff, vmax=1, vmin=0)


        axes[0].set_title('Target\nTime @ %.2f s'%(step*0.01))
        axes[1].set_title('Prediction\nTime @ %.2f s'%(step*0.01))
        # axes[2].set_title('Difference\nTime @ %.2f s'%(step*0.01))
        colorbar1 = fig.colorbar(handle1, ax=[axes[0], axes[1]])
        # colorbar2 = fig.colorbar(handle2, ax=axes[2])

        img = fig2data(fig)[:, :, :3]
        img = cv2.resize(img, (1700, 800))
        out.write(img)
        plt.close()

    for i in tqdm(range(599), total=600//skip):
        if i*skip < 599:
            render(i)
    out.release()
    print('video %s saved'%file_name)