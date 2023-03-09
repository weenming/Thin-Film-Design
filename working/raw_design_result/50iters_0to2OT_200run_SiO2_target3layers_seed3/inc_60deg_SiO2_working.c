#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <windows.h>

#include "Film.h"
#include "IOptimLM.h"
#include "MaterialC.h"
#include "MaterialN.h"
#include "MaterialS.h"
#include "MatrixD.h"
#include "OptimLM.h"
#include "VectorD.h"

typedef struct pass_argument {
    double thickness;
    int run_count;
} args;

int File_Size(char* file_path) {
    char c;
    int lines;
    FILE* fp;

    lines = 0;
    fp = fopen(file_path, "r");
    if (fp == NULL) {
        printf("file path does not exist!");
        abort();
    }
    while ((c = fgetc(fp)) != EOF) {
        if (c == '\n') {
            lines++;
        }
    }
    fclose(fp);

    return lines;
}

VectorD* VectorD_ReadFromFile(VectorD* this, char* file_path) {
    register int inc, i;
    register double tmp;
    register double* vector;
    FILE* file;
    char buffer[1000];
    register char* token;

    vector = VectorD_GetVector(this);
    inc = VectorD_GetInc(this);
    file = fopen(file_path, "r");
    if (file == NULL) {
        printf("read file path does not exist!\n");
        abort();
    }
    i = 0;
    while (fgets(buffer, 1000, file)) {
        token = strtok(buffer, "\n");
        tmp = atof(token);
        *(vector + i * inc) = tmp;
        i++;
    }
    fclose(file);

    return this;
}

VectorD* VectorD_WriteToFile(VectorD* this, char* file_path) {
    register int i, inc, size;
    int errNum = 0;
    register double* vector;
    FILE* file;
    char buffer[4000];
    char file_path_no_name[100];
    strcpy(file_path_no_name, file_path);
    for (i = strlen(file_path) - 1; i >= 0; i--) {
        if (file_path_no_name[i] == '/') {
            file_path_no_name[i + 1] = '\0';
            break;
        }
    }
    vector = VectorD_GetVector(this);
    inc = VectorD_GetInc(this);
    size = VectorD_GetSize(this);

    // when dir not exist, make dir
    file = fopen(file_path, "w");
    if (file == NULL) {
        // make dir iff input fpath is a "path"
        printf("making dir: %s\n", file_path_no_name);
        CreateDirectory(file_path_no_name, NULL);
    }
    fclose(file);

    printf("save to: %s \n", file_path);
    file = fopen(file_path, "w");
    if (file == NULL) {
        errNum = errno;
        printf("write to file fail! errno = %d reason = %s \n", errNum,
               strerror(errNum));
    }
    for (i = 0; i < size; i++) {
        sprintf(buffer, "%lf", *(vector + i * inc));
        strcat(buffer, "\n");
        fputs(buffer, file);
    }
    // 不close的话会句柄泄露！！！！
    fclose(file);

    return this;
}

VectorD* Func(void* implementor, VectorD* param, VectorD* wls,
              VectorD* output) {
    register int i;
    int size, half_size;
    Film* film;
    // printf("call func\n");
    film = (Film*)implementor;
    size = VectorD_GetSize(wls);
    half_size = size >> 1;
    // �趨��ǰ����
    Film_SetOptParam(film, param);
    for (i = 0; i < size; i++) {
        double wl, r;
        wl = VectorD_GetElement(wls, i);
        // s-polarized
        Film_SetConfig(film, 60, wl, 1, 0);
        Film_Calculate(film);
        r = Film_GetEnergyCoeff(film);
        // p-polarized must build after each reset...
        Film_SetConfig(film, 60, wl, 0, 0);
        Film_Calculate(film);
        r += Film_GetEnergyCoeff(film);

        r /= 2;
        VectorD_SetElement(output, i, r);
    }
    // for (i = half_size; i < size; i++) {
    //     double wl, t;

    //     wl = VectorD_GetElement(wls, i);
    //     Film_SetConfig(film, 60, wl, 1, 1);
    //     Film_Calculate(film);
    //     t = Film_GetEnergyCoeff(film);
    //     Film_SetConfig(film, 60, wl, 0, 1);
    //     Film_Calculate(film);
    //     t += Film_GetEnergyCoeff(film);
    //     t /= 2;
    //     VectorD_SetElement(output, i, t);
    // }

    return output;
}

MatrixD* JacMatrix(void* implementor, VectorD* param, VectorD* wls,
                   MatrixD* jac_matrix) {
    Film* film;
    register int i, ld;
    int layer_number, size, half_size;
    double* jacobi_ptr;
    VectorD *grad, *tmp;
    double wl;
    film = (Film*)implementor;
    size = VectorD_GetSize(wls);
    half_size = size >> 1;
    layer_number = MatrixD_GetCol(jac_matrix);
    // printf("call jacobi\n");
    // system("pause");
    Film_SetOptParam(film, param);
    grad = VectorD_Wrap(NULL, layer_number, 1);
    tmp = VectorD_New(layer_number);
    jacobi_ptr = MatrixD_GetMatrix(jac_matrix);
    ld = MatrixD_GetLd(jac_matrix);
    // Jacobi matrix: a layer_number by spec_points matrix
    // if multiple incidence angle and/or multiple polarization,
    //   need to connect the spec
    for (i = 0; i < size; i++) {
        double wl;
        wl = VectorD_GetElement(wls, i);
        // calculate spec R
        // polarization: (te + tm) / 2
        // te
        Film_SetConfig(film, 60, wl, 1, 0);
        Film_Calculate(film);
        VectorD_ReWrap(grad, jacobi_ptr + i * ld, layer_number, 1);
        Film_GetOptGrad(film, grad);
        // tm
        Film_SetConfig(film, 60, wl, 0, 0);
        Film_Calculate(film);
        Film_GetOptGrad(film, tmp);
        // average
        VectorD_AddVectorD(grad, tmp, grad);
        VectorD_MulNumD(grad, 0.5, grad);
    }
    // for (i = half_size; i < size; i++) {
    //     double wl;
    //     // calculate spec T
    //     // polarization: (te + tm) / 2
    //     // te
    //     Film_SetConfig(film, 60, wl, 1, 1);
    //     Film_Calculate(film);
    //     VectorD_ReWrap(grad, jacobi_ptr + i * ld, layer_number, 1);
    //     Film_GetOptGrad(film, grad);
    //     // tm
    //     Film_SetConfig(film, 60, wl, 0, 1);
    //     Film_Calculate(film);
    //     Film_GetOptGrad(film, tmp);
    //     // take average over te and tm
    //     VectorD_AddVectorD(grad, tmp, grad);
    //     VectorD_MulNumD(grad, 0.5, grad);
    // }
    VectorD_Del(tmp);
    VectorD_UnWrap(grad);

    return jac_matrix;
}

void insert_layer(Film* film, VectorD* wls, VectorD* target_spec,
                  IMaterial* odd_material, IMaterial* even_material) {
    /*even and odd: count from 0-th layer, even_material: CURRENT material of
     * 0-th layer...*/

    IMaterial* insert_material;
    IMaterial* inserted_material;
    IMaterial* best_insert_material;
    IMaterial* best_inserted_material;

    int layer_number = Film_GetOptSize(film);
    int layer, best_insert_layer_num;
    VectorD* vector_d = VectorD_New(layer_number);
    double* d = VectorD_GetVector(Film_GetOptParam(film, vector_d));
    double pos, best_insert_pos, pos_max, pos_step, grad;
    // not sure if negative is better or the other way around
    double best_grad = 0;
    double wl, j, f;
    int wl_i, wl_i_max = VectorD_GetSize(wls);
    // iter all layers
    float time = 0;
    for (layer = 0; layer < layer_number; layer++) {
        pos_max = d[layer];
        if (pos_max < 0.0001) {
            continue;
        }
        if (layer % 2 == 0) {
            insert_material = odd_material;
            inserted_material = even_material;
        } else {
            insert_material = even_material;
            inserted_material = odd_material;
        }

        // printf("layer %d thickness %f mu m, insert material %x \n", layer,
        //        VectorD_GetElement(vector_d, layer), insert_material);
        if (pos_max < 0.001) {
            pos_step = pos_max / 11;
        } else {
            pos_step = pos_max / 51;
        }
        time -= (float)clock();
        // iter all insertion positions
        for (pos = 0; pos < pos_max; pos += pos_step) {
            // printf("%f\n", pos);
            // iter all sampling points (all wls)
            // printf("insert point %f  ", pos);
            grad = 0;
            for (wl_i = 0; wl_i < wl_i_max; wl_i++) {
                // te
                wl = VectorD_GetElement(wls, wl_i);
                Film_SetConfig(film, 60, wl, 1, 0);
                Film_Calculate(film);
                // HERE (in function get_insert_grad) INDEX OF INSERTION LAYER
                // IS COUNTED FROM 1 !!!!!!
                j = Film_GetInsertGrad(film, layer + 1, pos, insert_material);
                f = (Film_GetEnergyCoeff(film) -
                     VectorD_GetElement(target_spec, wl_i));
                // tm
                wl = VectorD_GetElement(wls, wl_i);
                Film_SetConfig(film, 60, wl, 0, 0);
                Film_Calculate(film);
                j += Film_GetInsertGrad(film, layer + 1, pos, insert_material);
                f += (Film_GetEnergyCoeff(film) -
                      VectorD_GetElement(target_spec, wl_i));
                grad = grad + (j * f) / 4;
            }

            // this is gradient for merit functions defined as
            // \sum_i(f_i - f_{0i})^2
            grad /= wl_i_max;
            // printf("grad: %.9f\n", grad);
            if (grad < best_grad) {
                best_grad = grad;
                best_insert_layer_num = layer;
                best_insert_pos = pos;
                best_insert_material = insert_material;
                best_inserted_material = inserted_material;
            }
        }
        time += (float)clock();
    }
    if (best_grad == 0) {
        printf("cannot find insertion position!\n");
        VectorD_Del(vector_d);
        return;
    }
    time /= (float)CLOCKS_PER_SEC;
    printf("insertion takes %f sec\n insert at %d-th layer, %f\n", time,
           best_insert_layer_num, best_insert_pos);
    Film_InsertLayer(film, best_insert_layer_num + 2, best_inserted_material,
                     d[best_insert_layer_num] - best_insert_pos, 1);
    Film_InsertLayer(film, best_insert_layer_num + 2, insert_material, 0.00, 1);
    VectorD* vector_d_insert = VectorD_New(layer_number + 2);
    Film_GetOptParam(film, vector_d_insert);
    Film_SetOptParam(film,
                     VectorD_SetElement(vector_d_insert, best_insert_layer_num,
                                        best_insert_pos));
    Film_Build(film);
    Film_Calculate(film);

    VectorD_Del(vector_d);
    VectorD_Del(vector_d_insert);

    return;
}

void delete_film(Film* film) {
    int layer_num = Film_GetOptSize(film);
    int i = 1;
    VectorD* d;
    while (i != Film_GetOptSize(film)) {
        d = VectorD_New(layer_num);
        Film_GetOptParam(film, d);
        // VectorD_Show(d);

        if (VectorD_GetElement(d, i) == 0 && i < layer_num - 2) {
            VectorD_SetElement(
                d, i - 1,
                VectorD_GetElement(d, i - 1) + VectorD_GetElement(d, i + 1));
            Film_SetOptParam(film, d);
            // count from -1??? remove the layer before index...
            Film_RemoveLayer(film, i + 1);
            Film_RemoveLayer(film, i + 1);
            i--;
            printf("layer deleted!\n");
        }
        layer_num = Film_GetOptSize(film);
        i++;
        VectorD_Del(d);
    }
}

void test_deletion() {
    MaterialN* air;
    MaterialS *sio2, *tio2;
    IMaterial *Air, *SiO2, *TiO2;
    Film* film;
    double result;

    // build layers
    air = MaterialN_New(1, 0);
    sio2 = MaterialS_New(0.28604141, 1.07044083, 1.10202242, 0, 0.0100585997,
                         100.);
    tio2 =
        MaterialS_New(4.913, 0.2441 / 0.0803, -0.2441 / 0.0803, 0, 0.0803, 0);
    SiO2 = MaterialS_GetIMaterial(sio2);
    TiO2 = MaterialS_GetIMaterial(tio2);
    printf("SiO2 at %x; TiO2 at %x\n", SiO2, TiO2);
    Air = MaterialN_GetIMaterial(air);
    film = Film_New(0, 0.5, 1, 1);
    Film_AddLayer(film, Air, 0, 0);
    Film_AddLayer(film, SiO2, 1, 1);
    Film_AddLayer(film, SiO2, 2, 1);
    Film_AddLayer(film, SiO2, 0, 1);
    Film_AddLayer(film, SiO2, 4, 1);
    Film_AddLayer(film, TiO2, 0, 1);
    Film_AddLayer(film, SiO2, 3, 1);
    Film_AddLayer(film, SiO2, 0, 1);
    Film_AddLayer(film, SiO2, 0, 1);
    Film_AddLayer(film, SiO2, 0, 1);
    Film_AddLayer(film, SiO2, 9, 1);
    Film_AddLayer(film, SiO2, 0, 1);
    Film_AddLayer(film, SiO2, 0, 1);
    Film_AddLayer(film, SiO2, 10, 1);
    Film_AddLayer(film, SiO2, 0, 0);
    // 0 thickness means this layer is substrate
    Film_Build(film);
    Film_Calculate(film);
    VectorD* d = VectorD_New(Film_GetOptSize(film));
    delete_film(film);
    d = VectorD_New(Film_GetOptSize(film));
    Film_GetOptParam(film, d);
    VectorD_Show(d);
    return;
}

void get_insert_gradients(Film* film) {}

DWORD WINAPI run(LPVOID args_ptr) {
    double init_thickness = (*(args*)args_ptr).thickness;
    int run_count = (*(args*)args_ptr).run_count;
    MaterialN* air;
    MaterialS *sio2, *tio2;
    IMaterial *Air, *SiO2, *TiO2;
    Film* film;
    double result;

    // build layers
    air = MaterialN_New(1, 0);
    sio2 = MaterialS_New(0.28604141, 1.07044083, 1.10202242, 0, 0.0100585997,
                         100.);
    tio2 =
        MaterialS_New(4.913, 0.2441 / 0.0803, -0.2441 / 0.0803, 0, 0.0803, 0);
    SiO2 = MaterialS_GetIMaterial(sio2);
    TiO2 = MaterialS_GetIMaterial(tio2);
    printf("SiO2 at %x; TiO2 at %x\n", SiO2, TiO2);
    Air = MaterialN_GetIMaterial(air);
    film = Film_New(0, 0.5, 1, 1);
    Film_AddLayer(film, Air, 0, 0);
    Film_AddLayer(film, SiO2, init_thickness, 1);

    // 0 thickness means this layer is substrate
    Film_AddLayer(film, SiO2, 0, 0);
    Film_Build(film);
    Film_Calculate(film);

    // // Insert gradient
    // result = Film_GetInsertGrad(film, 1, 0.025, SiO2);
    // printf("%lf\n", result);

    // Optimize d: gradient descent

    IOptimLM iOptimLM = {film, Func, JacMatrix};
    OptimLM* optimLM;
    VectorD *d, *wls, *target_spec, *d_final;
    int opt_size, sample_pts_num;
    opt_size = Film_GetOptSize(film);

    d_final = VectorD_New(opt_size);
    optimLM = OptimLM_New(&iOptimLM);

    sample_pts_num = File_Size("wl_60deg_R.txt");
    // wavelengths of all sampling points
    wls = VectorD_New(sample_pts_num);
    VectorD_ReadFromFile(wls, "wl_60deg_R.txt");
    sample_pts_num = File_Size("generaetd_spectrum-INC_ANG60.0-WLS500.0to1000.0-R_3layers_seed3.txt");
    target_spec = VectorD_New(sample_pts_num);
    VectorD_ReadFromFile(target_spec, "generaetd_spectrum-INC_ANG60.0-WLS500.0to1000.0-R_3layers_seed3.txt");

    insert_layer(film, wls, target_spec, TiO2, SiO2);
    d = VectorD_New(Film_GetOptSize(film));
    Film_GetOptParam(film, d);
    printf("\nthread%d: starting iteration\n", run_count);
    char fpath[200];
    sprintf(fpath, "./../result/3layers_SiO2_0to2_seed3/run_%d/init", run_count);
    VectorD_WriteToFile(d, fpath);

    // start design, using needle optimization
    // 1st material argument: odd layer; 2nd material argument: even layer (the
    // layer at the top)
    int i;
    
    for (i = 0; i < 50; i++) {
        d = VectorD_New(Film_GetOptSize(film));
        Film_GetOptParam(film, d);
        // fit
        float time = -(float)clock();
        OptimLM_Fit(optimLM, d, wls, target_spec, d, NULL);
        time += (float)clock();
        time /= (float)CLOCKS_PER_SEC;
        printf("thread %d: gradient descent takes %f sec\n", run_count, time);
        // VectorD_Show(d);
        delete_film(film);
        printf("thread %d: insertion start\n", run_count);
        insert_layer(film, wls, target_spec, TiO2, SiO2);

        // save to file: want to save inserted d
        d = VectorD_New(Film_GetOptSize(film));
        Film_GetOptParam(film, d);
        sprintf(fpath, "./../result/3layers_SiO2_0to2_seed3/run_%d/iter_%d", run_count, i);
        VectorD_WriteToFile(d, fpath);
        printf("thread %d: %d-th iteration finished\n", run_count, i);
    }
    sprintf(fpath, "./../result/3layers_SiO2_0to2_seed3/run_%d/final", run_count);
    VectorD_WriteToFile(d, fpath);
    // // update current designed spectrum
    // The bug should be caused by this problem: d_final is not given the value
    // after insertions of d, and therefore has wrong shape for the film when
    // calculating the spectrum. The peculiar part is that the problem happens
    // occasionally...

    // IOptimLM_Func(&iOptimLM, d_final, wls,
    // target_spec);
    // // save spec to mem.
    // VectorD_WriteToFile(target_spec, "result.txt");

    // free mem

    // int errNum = errno;
    // printf("write to file fail! errno = %d reason = %s \n", errNum,
    //        strerror(errNum));
    VectorD_Del(d_final);
    VectorD_Del(target_spec);
    VectorD_Del(d);
    VectorD_Del(wls);
    OptimLM_Del(optimLM);
    MaterialN_Del(air);
    MaterialS_Del(sio2);
    MaterialS_Del(tio2);
    Film_Del(film);
    printf("thread %d: design terminated\n", run_count);
    int errNum = errno;
    printf("errno = %d reason = %s \n", errNum, strerror(errNum));
    return 0L;
}

int main() {
    int i;
    const int RATIO_SIZE = 200;
    HANDLE thread;
    void* p;
    VectorD* ratios = VectorD_New(RATIO_SIZE);
    VectorD_Linspace(ratios, 0, 2);
    VectorD_MulNumD(ratios, 2.05, ratios); // SiO2
    // VectorD_MulNumD(ratios, 1.23, ratios); // TiO2 don't forget to change the starting material

    // 必须给不同参数指定不同的地址，传入不同线程的函数
    // 初始化参数
    args p_arr[RATIO_SIZE];
    for (i = 0; i < RATIO_SIZE; i++) {
        p_arr[i].thickness = VectorD_GetElement(ratios, i);
        p_arr[i].run_count = i;
    }

    printf("The initial thicknesses of SiO2:\n");
    VectorD_Show(ratios);

    for (i = 0; i < RATIO_SIZE; i++) {
        p = &p_arr[i];
        thread = CreateThread(NULL, 0, run, p, 0, NULL);
        CloseHandle(thread);
    }
    // here 1e10 overflows: implicitly cast to int...
    Sleep(1e8);
    printf("main func terminated..");
    return 0;
}