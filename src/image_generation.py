import numpy as np
import matplotlib.pyplot as plt


def get_fourier(byte_string, numb_name, path_to_image_dir) -> None:
    try:
        x = np.frombuffer(byte_string, dtype=np.uint8).reshape(224, 224)
    except ValueError:
        print("Finish")
        return
    fft = np.fft.fft2(x)
    image_data_from_fourier = np.fft.ifft2(fft).real.astype(np.uint8)
    fig, ax = plt.subplots()
    ax.imshow(image_data_from_fourier)
    fig.savefig(path_to_image_dir + numb_name + ".png")
    plt.close()


def get_bytes_matrix(filename, path_to_image_dir) -> None:
    counter = 1
    with open(filename, 'rb') as f:
        while True:
            data = f.read(50176)
            if not data:
                break
            f.seek(-100, 1)
            counter += 1
            get_fourier(data, str(counter), path_to_image_dir)


filename = input("Enter path to pcap file: ")
path_to_image_dir = input("Enter path to image directory: ")
get_bytes_matrix(filename, path_to_image_dir)
