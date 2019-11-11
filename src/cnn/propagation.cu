
#include "layer.cu"
#include "kernels.cu"

#include <time.h>

// Define layers of CNN
static Layer l_input = Layer(0, 0, 28*28);
static Layer l_c1 = Layer(5*5, 6, 24*24*6);
static Layer l_s1 = Layer(4*4, 1, 6*6*6);
static Layer l_f = Layer(6*6*6, 10, 10);

const static float threshold = 1.0E-02f;

static void load_model() {
    FILE *model = fopen("model.dat", "rb");
    if(!model)
        return;
    l_input.read_from_file(model);
    l_c1.read_from_file(model);
    l_s1.read_from_file(model);
    l_f.read_from_file(model);
    fclose(model);
}

static void save_model() {
    FILE *model = fopen("model.dat", "wb");
    l_input.save_to_file(model);
    l_c1.save_to_file(model);
    l_s1.save_to_file(model);
    l_f.save_to_file(model);
    fclose(model);
}

// Forward propagation of a single row in dataset
static double forward_propagation(double data[28][28])
{
    float input[28][28];

    for (int i = 0; i < 28; ++i) {
        for (int j = 0; j < 28; ++j) {
            input[i][j] = data[i][j];
        }
    }

    l_input.clear();
    l_c1.clear();
    l_s1.clear();
    l_f.clear();

    clock_t start, end;
    start = clock();

    l_input.setOutput((float *)input);
    
    fp_preact_c1<<<64, 64>>>((float (*)[28])l_input.output, (float (*)[24][24])l_c1.preact, (float (*)[5][5])l_c1.weight);
    fp_bias_c1<<<64, 64>>>((float (*)[24][24])l_c1.preact, l_c1.bias);
    apply_step_function<<<64, 64>>>(l_c1.preact, l_c1.output, l_c1.O);

    fp_preact_s1<<<64, 64>>>((float (*)[24][24])l_c1.output, (float (*)[6][6])l_s1.preact, (float (*)[4][4])l_s1.weight);
    fp_bias_s1<<<64, 64>>>((float (*)[6][6])l_s1.preact, l_s1.bias);
    apply_step_function<<<64, 64>>>(l_s1.preact, l_s1.output, l_s1.O);

    fp_preact_f<<<64, 64>>>((float (*)[6][6])l_s1.output, l_f.preact, (float (*)[6][6][6])l_f.weight);
    fp_bias_f<<<64, 64>>>(l_f.preact, l_f.bias);
    apply_step_function<<<64, 64>>>(l_f.preact, l_f.output, l_f.O);
    
    end = clock();
    return ((double) (end - start)) / CLOCKS_PER_SEC;
}

// Back propagation to update weights
static double back_propagation()
{
    clock_t start, end;

    start = clock();

    bp_weight_f<<<64, 64>>>((float (*)[6][6][6])l_f.d_weight, l_f.d_preact, (float (*)[6][6])l_s1.output);
    bp_bias_f<<<64, 64>>>(l_f.bias, l_f.d_preact);

    bp_output_s1<<<64, 64>>>((float (*)[6][6])l_s1.d_output, (float (*)[6][6][6])l_f.weight, l_f.d_preact);
    bp_preact_s1<<<64, 64>>>((float (*)[6][6])l_s1.d_preact, (float (*)[6][6])l_s1.d_output, (float (*)[6][6])l_s1.preact);
    bp_weight_s1<<<64, 64>>>((float (*)[4][4])l_s1.d_weight, (float (*)[6][6])l_s1.d_preact, (float (*)[24][24])l_c1.output);
    bp_bias_s1<<<64, 64>>>(l_s1.bias, (float (*)[6][6])l_s1.d_preact);

    bp_output_c1<<<64, 64>>>((float (*)[24][24])l_c1.d_output, (float (*)[4][4])l_s1.weight, (float (*)[6][6])l_s1.d_preact);
    bp_preact_c1<<<64, 64>>>((float (*)[24][24])l_c1.d_preact, (float (*)[24][24])l_c1.d_output, (float (*)[24][24])l_c1.preact);
    bp_weight_c1<<<64, 64>>>((float (*)[5][5])l_c1.d_weight, (float (*)[24][24])l_c1.d_preact, (float (*)[28])l_input.output);
    bp_bias_c1<<<64, 64>>>(l_c1.bias, (float (*)[24][24])l_c1.d_preact);

    apply_grad<<<64, 64>>>(l_f.weight, l_f.d_weight, l_f.M * l_f.N);
    apply_grad<<<64, 64>>>(l_s1.weight, l_s1.d_weight, l_s1.M * l_s1.N);
    apply_grad<<<64, 64>>>(l_c1.weight, l_c1.d_weight, l_c1.M * l_c1.N);

    end = clock();
    return ((double) (end - start)) / CLOCKS_PER_SEC;
}

// Returns label of given data (0-9)
static unsigned int classify(double data[28][28], char opt)
{
    float res[10];

    forward_propagation(data);

    unsigned int max = 0;

    cudaMemcpy(res, l_f.output, sizeof(float) * 10, cudaMemcpyDeviceToHost);
    if(opt == 'y')
        for (int i = 0; i < 28; ++i) {
            for (int j = 0; j < 28; ++j) {
                int color = int(255 * data[i][j]);
                fprintf(stdout, "\033[48;2;%d;%d;%dm  \033[0m", color, color, color);
            }
            fprintf(stdout, "\n");
        }
    for (int i = 0; i < 10; ++i) {
        if(opt == 'y') {
            int color = int(res[i]*255);
            fprintf(stdout, " [ %d ] \033[48;2;%d;0;%dm%02.15f\033[0m\n", i, color/4, color, res[i]);
        }
        if (res[max] < res[i]) {
            max = i;
        }
    }

    return max;
}
