
#include <cstring>

typedef struct mnist_data {
    double data[28][28]; /* 28x28 data for the image */
    unsigned int label; /* label : 0 to 9 */
} mnist_data;

/*
 * Load a unsigned int from raw data.
 * MSB first.
 */
static unsigned int mnist_bin_to_int(char *v)
{
    int i;
    unsigned int ret = 0;

    for (i = 0; i < 4; ++i) {
        ret <<= 8;
        ret |= (unsigned char)v[i];
    }

    return ret;
}

/*
 * MNIST dataset loader.
 *
 * Returns 0 if successed.
 * Check comments for the return codes.
 */
 int mnist_load(
    const char *image_filename,
    const char *label_filename,
    mnist_data **data,
    unsigned int *count)
{
    read_start:
    int return_code = 0;
    int i;
    char tmp[4];

    unsigned int image_cnt, label_cnt;
    unsigned int image_dim[2];

    FILE *ifp = fopen(image_filename, "rb");
    FILE *lfp = fopen(label_filename, "rb");

    if (!ifp || !lfp) {
        system("wget \"http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz\";"
           "wget \"http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz\";"
           "wget \"http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz\";"
           "wget \"http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz\";"
           "gunzip train-images-idx3-ubyte.gz;"
           "gunzip train-labels-idx1-ubyte.gz;"
           "gunzip t10k-images-idx3-ubyte.gz;"
           "gunzip t10k-labels-idx1-ubyte.gz;"
           "mkdir dataset;"
           "mv train-images-idx3-ubyte dataset/;"
           "mv train-labels-idx1-ubyte dataset/;"
           "mv t10k-images-idx3-ubyte dataset/;"
           "mv t10k-labels-idx1-ubyte dataset/" );
        goto read_start;
    }

    fread(tmp, 1, 4, ifp);
    if (mnist_bin_to_int(tmp) != 2051) {
        return_code = -2; /* Not a valid image file */
        goto cleanup;
    }

    fread(tmp, 1, 4, lfp);
    if (mnist_bin_to_int(tmp) != 2049) {
        return_code = -3; /* Not a valid label file */
        goto cleanup;
    }

    fread(tmp, 1, 4, ifp);
    image_cnt = mnist_bin_to_int(tmp);

    fread(tmp, 1, 4, lfp);
    label_cnt = mnist_bin_to_int(tmp);

    if (image_cnt != label_cnt) {
        return_code = -4; /* Element counts of 2 files mismatch */
        goto cleanup;
    }

    for (i = 0; i < 2; ++i) {
        fread(tmp, 1, 4, ifp);
        image_dim[i] = mnist_bin_to_int(tmp);
    }

    if (image_dim[0] != 28 || image_dim[1] != 28) {
        return_code = -2; /* Not a valid image file */
        goto cleanup;
    }

    *count = image_cnt;
    *data = (mnist_data *)malloc(sizeof(mnist_data) * image_cnt);

    for (i = 0; i < image_cnt; ++i) {
        int j;
        unsigned char read_data[28 * 28];
        mnist_data *d = &(*data)[i];
        fread(read_data, 1, 28*28, ifp);
        for (j = 0; j < 28*28; ++j) {
            d->data[j/28][j%28] = read_data[j] / 255.0;
        }

        fread(tmp, 1, 1, lfp);
        d->label = tmp[0];
    }

cleanup:
    if (ifp) fclose(ifp);
    if (lfp) fclose(lfp);
    return return_code;
}


