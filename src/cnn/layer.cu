
#include <cstdio>

class Layer {
    public:
    int M, N, O;

    float *output;
    float *preact;

    float *bias;
    float *weight;

    float *d_output;
    float *d_preact;
    float *d_weight;

    Layer(int M, int N, int O);

    ~Layer();

    void setOutput(float *data);
    void clear();
    void bp_clear();
    void save_to_file(FILE*);
    void read_from_file(FILE*);
};

// Constructor
Layer::Layer(int M, int N, int O)
{
    this->M = M;
    this->N = N;
    this->O = O;

    float h_bias[N];
    float h_weight[N][M];

    output = NULL;
    preact = NULL;
    bias   = NULL;
    weight = NULL;

    for (int i = 0; i < N; ++i) {
        h_bias[i] = 0.5f - float(rand()) / float(RAND_MAX);
        /*h_bias[i] = 0.0f;*/

        for (int j = 0; j < M; ++j) {
            h_weight[i][j] = 0.5f - float(rand()) / float(RAND_MAX);
            /*h_weight[i][j] = 0.05f;*/
        }
    }

    cudaMalloc(&output, sizeof(float) * O);
    cudaMalloc(&preact, sizeof(float) * O);

    cudaMalloc(&bias, sizeof(float) * N);

    cudaMalloc(&weight, sizeof(float) * M * N);

    cudaMalloc(&d_output, sizeof(float) * O);
    cudaMalloc(&d_preact, sizeof(float) * O);
    cudaMalloc(&d_weight, sizeof(float) * M * N);

    cudaMemcpy(bias, h_bias, sizeof(float) * N, cudaMemcpyHostToDevice);

    cudaMemcpy(weight, h_weight, sizeof(float) * M * N, cudaMemcpyHostToDevice);
}

// Destructor
Layer::~Layer()
{
    cudaFree(output);
    cudaFree(preact);

    cudaFree(bias);

    cudaFree(weight);

    cudaFree(d_output);
    cudaFree(d_preact);
    cudaFree(d_weight);
}

// Send data one row from dataset to the GPU
void Layer::setOutput(float *data)
{
    cudaMemcpy(output, data, sizeof(float) * O, cudaMemcpyHostToDevice);
}

// Reset GPU memory between iterations
void Layer::clear()
{
    cudaMemset(output, 0x00, sizeof(float) * O);
    cudaMemset(preact, 0x00, sizeof(float) * O);
}

void Layer::bp_clear()
{
    cudaMemset(d_output, 0x00, sizeof(float) * O);
    cudaMemset(d_preact, 0x00, sizeof(float) * O);
    cudaMemset(d_weight, 0x00, sizeof(float) * M * N);
}

void Layer::save_to_file(FILE *model)
{
    char buffer[100000];
    fwrite((char*)&M, sizeof(int), 1, model);
    fwrite((char*)&N, sizeof(int), 1, model);

    cudaMemcpy(buffer, (char*)bias, sizeof(float) * N, cudaMemcpyDeviceToHost);
    fwrite(buffer, sizeof(float) * N, 1, model);
    cudaMemcpy(buffer, (char*)weight, sizeof(float) * M * N, cudaMemcpyDeviceToHost);
    fwrite(buffer, sizeof(float) * M * N, 1, model);
}

void Layer::read_from_file(FILE *model)
{
    char buffer[100000];
    
    fread((char*)&M, sizeof(int), 1, model);
    fread((char*)&N, sizeof(int), 1, model);
    
    fread(buffer, sizeof(float) * N, 1, model);
    cudaMemcpy(bias, (float*)buffer, sizeof(float) * N, cudaMemcpyHostToDevice);
    fread(buffer, sizeof(float) * M * N, 1, model);
    cudaMemcpy(weight, (float*)buffer, sizeof(float) * M * N, cudaMemcpyHostToDevice);
}
