import numpy as np
import matplotlib.pyplot as plt

def get_fourier(byte_string, numb_name) -> None:
    pass_to_image_dir = "./pcap_test/image_control_test/"
    try:
        x = np.frombuffer(byte_string, dtype=np.uint8).reshape(224, 224)
    except ValueError:
        print("Finish")
        return
    fft = np.fft.fft2(x)
    image_data_from_fourier = np.fft.ifft2(fft).real.astype(np.uint8)
    fig, ax = plt.subplots()
    ax.imshow(image_data_from_fourier)
    fig.savefig(pass_to_image_dir + numb_name + ".png")
    plt.close()


def get_bytes_matrix(filename) -> None:
    counter = 1
    with open(filename, 'rb') as f:
        while True:
            data = f.read(50176)
            if not data:
                break
            f.seek(-100, 1)
            counter += 1
            # print(data, end="\n\n")
            get_fourier(data, str(counter))


filename = "./pcap_test/control_test.pcap"
get_bytes_matrix(filename)