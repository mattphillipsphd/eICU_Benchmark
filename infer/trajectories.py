import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import os

pe = os.path.exists
pj = os.path.join
HOME = os.path.expanduser("~")


class Trajectories:
    def __init__(self, data_mat, interval=100, color_dict={}, final_extra=0):
        self._ax = None
        self._color_dict = color_dict
        self._data_mat = data_mat # (N, T, 2), where N is number of points,
            # T is length of sequence
        self._fig = None
        self._final_extra = final_extra
        self._frame_idx = 0
        self._interval = interval
        self._num_frames = data_mat.shape[1]
        self._num_points = data_mat.shape[0]
        self._seq_len = data_mat.shape[1] + final_extra

        self._fig,self._ax = plt.subplots(figsize=(16,16))

        self._animation = animation.FuncAnimation(self._fig, self._update,
                interval=self._interval, init_func=self._setup_plot, blit=True)

    def save(self, save_path):
        self._animation.save(save_path)

    def _data_stream(self):
        self._frame_idx %= self._seq_len
        frame_idx = self._frame_idx if self._frame_idx<self._num_frames \
                else self._num_frames-1
        P = self._data_mat[:, frame_idx, :]
        self._frame_idx += 1
        return P

    def _setup_plot(self):
        finalX = self._data_mat[:,-1,0]
        finalY = self._data_mat[:,-1,1]
        self._scat = self._ax.scatter(finalX, finalY)
        return self._scat,

    def _update(self, i):
        P = self._data_stream()
        self._scat.set_offsets(P)
        colors = np.zeros((len(P),))
        sizes = np.zeros((len(P),))
        for c, (color,(ixs,size)) in enumerate( self._color_dict.items() ):
            colors[ixs] = 0.2 * c
            sizes[ixs] = size
        self._scat.set_sizes(sizes)
        self._scat.set_array( np.array(colors) )
        return self._scat,


if __name__ == "__main__":
    data_mat = np.random.rand(50,100,2)
    I = np.random.rand(50)
    color_d = { "r" : (I<0.5, 12),
            "b" : (I>=0.5, 24)
            }
    traj = Trajectories(data_mat, color_dict=color_d, final_extra=20)
    plt.show()

