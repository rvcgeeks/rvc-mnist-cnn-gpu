
#include "cnn/propagation.cu"
#include "mnist_load.c"

#include <unistd.h> //for sleep in test()

static mnist_data *train_set, *test_set;
static unsigned int train_cnt, test_cnt;

// Data and model loading methods
static inline void load_data(int if_train)
{
    if(if_train)
        mnist_load("dataset/train-images-idx3-ubyte", "dataset/train-labels-idx1-ubyte",
            &train_set, &train_cnt);
    else 
        mnist_load("dataset/t10k-images-idx3-ubyte", "dataset/t10k-labels-idx1-ubyte",
            &test_set, &test_cnt);
}

// Unfold the input layer
static void unfold_input(double input[28][28], double unfolded[24*24][5*5])
{
    int a = 0;
    (void)unfold_input;

    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j) {
            int b = 0;
            for (int x = i; x < i + 2; ++x)
                for (int y = j; y < j+2; ++y)
                    unfolded[a][b++] = input[x][y];
            a++;
        }
}

static void learn(int iter)
{
    static cublasHandle_t blas;
    cublasCreate(&blas);

    float err;
    
    double time_taken = 0.0;

    fprintf(stdout ,"Learning\n");

    while (iter < 0 || iter-- > 0) {
        err = 0.0f;

        for (int i = 0; i < train_cnt; ++i) {
            float tmp_err;

            time_taken += forward_propagation(train_set[i].data);

            l_f.bp_clear();
            l_s1.bp_clear();
            l_c1.bp_clear();

            // Euclid distance of train_set[i]
            makeError<<<10, 1>>>(l_f.d_preact, l_f.output, train_set[i].label, 10);
            cublasSnrm2(blas, 10, l_f.d_preact, 1, &tmp_err);
            err += tmp_err;

            time_taken += back_propagation();
        }

        err /= train_cnt;
        fprintf(stdout, "error: %e, time_on_gpu: %lf\n", err, time_taken);

        if (err < threshold) {
            fprintf(stdout, "Training complete, error less than threshold\n\n");
            break;
        }

    }
    
    fprintf(stdout, "\n Time - %lf\n", time_taken);
}

// Perform forward propagation of test data
static void test()
{
    int error = 0, res; char opt;
    fprintf(stdout, "Show images? [y/n]:");
    fscanf(stdin, "%c", &opt);
    for (int i = 0; i < test_cnt; ++i) {
        if(opt == 'y')
            fprintf(stdout, "\033[2J\033[1;1H");
        res = classify(test_set[i].data, opt);
        if (res != test_set[i].label)
            ++error;
        if(opt == 'y') {
            fprintf(stdout, "\033[1;3");
            if (res != test_set[i].label)
                fprintf(stdout, "1");
            else
                fprintf(stdout, "2");
            fprintf(stdout, "m  %f percent correct network\n  ████████████████████████\n  ████████████████████████\033[0m", 100 * ( 1 - error /float(1 + i)));
            sleep(1);
        }
    }
    fprintf(stdout, "%f percent correct network\n", 100 * ( 1 - error /float(test_cnt)));
}

int main(int argc, const  char **argv)
{
    srand(time(NULL));

    CUresult err = cuInit(0);
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "CUDA initialisation failed with error code - %d\n", err);
        return 1;
    }
    if (argc == 2) {
        load_data(1);
        load_model();
        learn(atoi(argv[1]));
        save_model();
    } else {
        load_data(0);
        load_model();
        test();
    }

    return 0;
}
