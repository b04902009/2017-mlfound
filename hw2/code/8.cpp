#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <fstream>
using namespace std;
#define N 20
#define T 1000
double data[N][2];
int cmp(const void *a, const void *b){
	double* A = (double*)a;
	double *B = (double*)b;
	return (A[0] > B[0]);
}
void generate_data_set(int seed){
	srand(seed);
	for(int i = 0; i < N; i++){
		data[i][0] = (-100000 + rand()%200000) / 100000.0;
		data[i][1] = (data[i][0] > 0)? 1 : -1;
	}
	qsort(data, N, sizeof(double[2]), cmp);
}
void make_noise(int seed){
	int noise_num = 0.2 * N;
	srand(seed);
	for(int i = 0; i < noise_num; i++)
		data[rand()%N][1] *= -1;
}

int best_s, min_error = N;
double best_theta;
int compute_error(int s, double theta){
	int error = 0;
	int sign;
	double h;
	for(int i = 0; i < N; i++){
		sign = (data[i][0] > theta)? 1 : -1;
		h = s * sign;
		if(h != data[i][1])  error++;
	}
	return error;
}
double compute_Ein(){
	double theta_list[N+1];
	theta_list[0] = (-1+data[0][0])/2.0;
	theta_list[N] = (data[N-1][0]+1)/2.0;
	for(int i = 1; i < N; i++)
		theta_list[i] = (data[i][0]+data[i+1][0])/2.0;

	for(int s = -1; s <= 1; s += 2)
		for(int i = 0; i < N+1; i++){
			int error = compute_error(s, theta_list[i]);
			// printf("theta:%lf, error:%d\n", theta_list[i], error);
			if(error < min_error){
				best_s = s;
				best_theta = theta_list[i];
				min_error = error;
			}
		}
	return (double)min_error / (double)N;
}
double compute_Eout(){
	return 0.5 + 0.3 * best_s * (fabs(best_theta)-1.0);
}
int main(){
	double Ein, Eout, avg_Ein = 0.0, avg_Eout = 0.0;
	// ofstream file;
	// file.open ("Ein_Eout.csv");
	for(int i = 0; i < T; i++){
		best_s = 0;
		best_theta = 0.0;
		min_error = N;
		generate_data_set(i);
		make_noise(i);
		Ein = compute_Ein();
		Eout = compute_Eout();
		printf("Ein:%lf\tEout:%lf\n", Ein, Eout);
		avg_Ein += Ein;
		avg_Eout += Eout;
		// file << Ein << "," << Eout << endl;
	}
  	// file.close();
	printf("Averaged_Ein = %lf\n", avg_Ein/T);
	printf("Averaged_Eout = %lf\n", avg_Eout/T);
	return 0;
}