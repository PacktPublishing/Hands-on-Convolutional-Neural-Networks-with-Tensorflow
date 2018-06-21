import fire
import numpy as np
import scipy.misc

class HandleDataFolder:
    __xs = []
    __xs_data = []
    __ys = []
    __folder = ''

    def load_dataset(self, path='./'):
        self.__folder = path
        # Open labels.txt
        with open(self.__folder + "labels.txt") as f:
            for line in f:
                # Image path
                line_p = line.split()
                self.__xs.append(self.__folder + line_p[0])
                # pc, x, y, bh, bw
                self.__ys.append(np.array([np.float32(line_p[1]), np.float32(line_p[2]), np.float32(line_p[3]),
                                  np.float32(line_p[4]), np.float32(line_p[5])]))

            # Use zip to create a list with images/labels
            c = list(zip(self.__xs, self.__ys))
            xs, ys = list(zip(*c))

        # Load data
        for img_path in xs:
            image = scipy.misc.imread(img_path, mode="RGB")
            self.__xs_data.append(image)

        # Return data and labels
        self.__ys = np.asarray(self.__ys)
        return self.__xs_data, self.__ys

if __name__ == '__main__':
  fire.Fire(HandleDataFolder)