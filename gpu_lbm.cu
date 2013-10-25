/*

Simulation of flow inside a 2D square cavity
using the lattice Boltzmann method (LBM)

Written by:       Abhijit Joshi (abhijit@accelereyes.com)

Last modified on: Thursday, July 18 2013 @12:08 pm

Build instructions: make (uses Makefile present in this folder)

Run instructions: optirun ./gpu_lbm

*/

#include<iostream>
#include<stdio.h>
#include<arrayfire.h>
using namespace af;

// problem parameters

const int     N = 128;                  // number of node points along X and Y (cavity length in lattice units)
const int     TIME_STEPS = 10000;       // number of time steps for which the simulation is run
const double  REYNOLDS_NUMBER = 1E6;    // REYNOLDS_NUMBER = LID_VELOCITY * N / kinematicViscosity

// don't change these unless you know what you are doing

const int     Q = 9;                    // number of discrete velocity aections used
const double  DENSITY = 2.7;            // fluid density in lattice units
const double  LID_VELOCITY = 0.05;      // lid velocity in lattice units

// D2Q9 parameters 

__constant__ double ex[Q];
__constant__ double ey[Q];
__constant__ int oppos[Q];
__constant__ double wt[Q];

// populate D3Q19 parameters and copy them to __constant__ memory on the GPU

void D3Q9(double *ex_h, double *ey_h, int *oppos_h, double *wt_h)
{
    // D2Q9 model base velocities and weights

    ex_h[0] =  0.0;   ey_h[0] =  0.0;   wt_h[0] = 4.0 /  9.0;
    ex_h[1] =  1.0;   ey_h[1] =  0.0;   wt_h[1] = 1.0 /  9.0;
    ex_h[2] =  0.0;   ey_h[2] =  1.0;   wt_h[2] = 1.0 /  9.0;
    ex_h[3] = -1.0;   ey_h[3] =  0.0;   wt_h[3] = 1.0 /  9.0;
    ex_h[4] =  0.0;   ey_h[4] = -1.0;   wt_h[4] = 1.0 /  9.0;
    ex_h[5] =  1.0;   ey_h[5] =  1.0;   wt_h[5] = 1.0 / 36.0;
    ex_h[6] = -1.0;   ey_h[6] =  1.0;   wt_h[6] = 1.0 / 36.0;
    ex_h[7] = -1.0;   ey_h[7] = -1.0;   wt_h[7] = 1.0 / 36.0;
    ex_h[8] =  1.0;   ey_h[8] = -1.0;   wt_h[8] = 1.0 / 36.0;

    // define opposite (anti) aections (useful for implementing bounce back)

    oppos_h[0] = 0;      //      6        2        5
    oppos_h[1] = 3;      //               ^
    oppos_h[2] = 4;      //               |
    oppos_h[3] = 1;      //               |
    oppos_h[4] = 2;      //      3 <----- 0 -----> 1
    oppos_h[5] = 7;      //               |
    oppos_h[6] = 8;      //               |
    oppos_h[7] = 5;      //               v
    oppos_h[8] = 6;      //      7        4        8

    // copy to constant (read-only) memory
    cudaMemcpyToSymbol(ex,    ex_h,    Q * sizeof(double));  // x-component of velocity direction
    cudaMemcpyToSymbol(ey,    ey_h,    Q * sizeof(double));  // y-component of velocity direction
    cudaMemcpyToSymbol(oppos, oppos_h, Q * sizeof(int));     // opposite direction for each velocity direction
    cudaMemcpyToSymbol(wt,    wt_h,    Q * sizeof(double));  // weight factor for velocity direction
}

// initialize values for aection vectors, density, velocity and distribution functions on the GPU

__global__ void initialize(const int N, const int Q, const double DENSITY, const double LID_VELOCITY, 
                           double *rho, double *ux, double *uy, double* sigma, 
                           double *f, double *feq, double *f_new)
{
    // compute the global "i" and "j" location handled by this thread

    const int i = blockIdx.x * blockDim.x + threadIdx.x ;
    const int j = blockIdx.y * blockDim.y + threadIdx.y ;

    // bound checking
    if( (i > (N-1)) || (j > (N-1)) ) return;

    // natural index for location (i,j)

    const int index = i*N+j;  // column-ordering

    // initialize density and velocity fields inside the cavity

      rho[index] = DENSITY;   // density
       ux[index] = 0.0;       // x-component of velocity
       uy[index] = 0.0;       // x-component of velocity
    sigma[index] = 0.0;       // rate-of-strain field

    // specify boundary condition for the moving lid

    if(j==0) ux[index] = LID_VELOCITY;

    // assign initial values for distribution functions
    // along various aections using equilibriu, functions

    #pragma unroll
    for(int a=0;a<Q;a++) {

        int index_f = a + index*Q;

        double edotu = ex[a]*ux[index] + ey[a]*uy[index];
        double udotu = ux[index]*ux[index] + uy[index]*uy[index];

        feq[index_f]   = rho[index] * wt[a] * (1.0 + 3.0*edotu + 4.5*edotu*edotu - 1.5*udotu);
        f[index_f]     = feq[index_f];
        f_new[index_f] = feq[index_f];

    }
}

// this function updates the values of the distribution functions at all points along all directions
// carries out one lattice time-step (streaming + collision) in the algorithm

__global__ void collideAndStream(// READ-ONLY parameters (used by this function but not changed)
                                 const int N, const int Q, const double DENSITY, const double LID_VELOCITY, const double REYNOLDS_NUMBER,
                                 // READ + WRITE parameters (get updated in this function)
                                 double *rho,         // density
                                 double *ux,         // X-velocity
                                 double *uy,         // Y-velocity
                                 double *sigma,      // rate-of-strain
                                 double *f,          // distribution function
                                 double *feq,        // equilibrium distribution function
                                 double *f_new)      // new distribution function
{
    // compute the global "i" and "j" location handled by this thread

    const int i = blockIdx.x * blockDim.x + threadIdx.x ;
    const int j = blockIdx.y * blockDim.y + threadIdx.y ;

    // bound checking
    if( (i < 1) || (i > (N-2)) || (j < 1) || (j > (N-2)) ) return;

    // natural index
    const int index = i*N + j;  // column-major ordering

    // calculate fluid viscosity based on the Reynolds number
    double kinematicViscosity = LID_VELOCITY * (double) N / REYNOLDS_NUMBER;

    // calculate relaxation time tau
    double tau =  0.5 + 3.0 * kinematicViscosity;

    // collision
    #pragma unroll
    for(int a=0;a<Q;a++) {
        int index_f = a + index*Q;
        double edotu = ex[a]*ux[index] + ey[a]*uy[index];
        double udotu = ux[index]*ux[index] + uy[index]*uy[index];
        feq[index_f] = rho[index] * wt[a] * (1 + 3*edotu + 4.5*edotu*edotu - 1.5*udotu);
    }

    // streaming from interior node points

    #pragma unroll
    for(int a=0;a<Q;a++) {

        int index_f = a + index*Q;
        int index_nbr = (i+ex[a])*N + (j+ey[a]);
        int index_nbr_f = a + index_nbr * Q;
        int indexoppos = oppos[a] + index*Q;

        double tau_eff, tau_t, C_Smagorinsky;  // turbulence model parameters

        C_Smagorinsky = 0.16;

        // tau_t = additional contribution to the relaxation time 
        //         because of the "eddy viscosity" model
        // =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
        // REFERENCE: Krafczyk M., Tolke J. and Luo L.-S. (2003)
        //            Large-Eddy Simulations with a Multiple-Relaxation-Time LBE Model
        //            International Journal of Modern Physics B, Vol.17, 33-39
        // =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

        tau_t = 0.5*(pow(pow(tau,2) + 18.0*pow(C_Smagorinsky,2)*sigma[index],0.5) - tau);

        // the effective relaxation time accounts for the additional "eddy viscosity"
        // effects. Note that tau_eff now varies from point to point in the domain, and is
        // larger for large strain rates. If the strain rate is zero, tau_eff = 0 and we
        // revert back to the original (laminar) LBM scheme where tau_eff = tau.

        tau_eff = tau + tau_t;

        // post-collision distribution at (i,j) along "a"
        double f_plus = f[index_f] - (f[index_f] - feq[index_f])/tau_eff;

        int iS = i + ex[a]; int jS = j + ey[a];

        if((iS==0) || (iS==N-1) || (jS==0) || (jS==N-1)) {
            // bounce back
            double ubdote = ux[index_nbr]*ex[a] + uy[index_nbr]*ey[a];
            f_new[indexoppos] = f_plus - 6.0 * DENSITY * wt[a] * ubdote;
        }
        else {
            // stream to neighbor
            f_new[index_nbr_f] = f_plus;
        }
    }
}

__global__ void everythingElse( // READ-ONLY parameters (used by this function but not changed)
                                const int N, const int Q, const double DENSITY, const double LID_VELOCITY, const double REYNOLDS_NUMBER,
                                // READ + WRITE parameters (get updated in this function)
                                double *rho,         // density
                                double *ux,         // X-velocity
                                double *uy,         // Y-velocity
                                double *sigma,      // rate-of-strain
                                double *f,          // distribution function
                                double *feq,        // equilibrium distribution function
                                double *f_new)      // new distribution function
{
    // compute the global "i" and "j" location of this thread

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;

    // bound checking
    if( (i < 1) || (i > (N-2)) || (j < 1) || (j > (N-2)) ) return;

    // natural index
    const int index = i*N + j;  // column-major ordering

    // push f_new into f
    #pragma unroll
    for(int a=0;a<Q;a++) {
        int index_f = a + index*Q;
        f[index_f] = f_new[index_f];
    }

    // update density at interior nodes
    rho[index]=0.0;
    for(int a=0;a<Q;a++) {
        int index_f = a + index*Q;
        rho[index] += f_new[index_f];
    }

    // update velocity at interior nodes
    double velx=0.0;
    double vely=0.0;
    for(int a=0;a<Q;a++) {
        int index_f = a + index*Q;
        velx += f_new[index_f]*ex[a];
        vely += f_new[index_f]*ey[a];
    }
    ux[index] = velx/rho[index];
    uy[index] = vely/rho[index];

    // update the rate-of-strain field
    double sum_xx = 0.0, sum_xy = 0.0, sum_xz = 0.0;
    double sum_yx = 0.0, sum_yy = 0.0, sum_yz = 0.0;
    double sum_zx = 0.0, sum_zy = 0.0, sum_zz = 0.0;
    for(int a=1; a<Q; a++)
    {
        int index_f = a + index*Q;

        sum_xx = sum_xx + (f_new[index_f] - feq[index_f])*ex[a]*ex[a];
        sum_xy = sum_xy + (f_new[index_f] - feq[index_f])*ex[a]*ey[a];
        sum_xz = 0.0;
        sum_yx = sum_xy;
        sum_yy = sum_yy + (f_new[index_f] - feq[index_f])*ey[a]*ey[a];
        sum_yz = 0.0;
        sum_zx = 0.0;
        sum_zy = 0.0;
        sum_zz = 0.0;
    }

    // evaluate |S| (magnitude of the strain-rate)
    sigma[index] = pow(sum_xx,2) + pow(sum_xy,2) + pow(sum_xz,2)
                 + pow(sum_yx,2) + pow(sum_yy,2) + pow(sum_yz,2)
                 + pow(sum_zx,2) + pow(sum_zy,2) + pow(sum_zz,2);

    sigma[index] = pow(sigma[index],0.5);
}

int main(int argc, char* argv[])
{
    try {

        // check whether to do graphics stuff or not
        bool isconsole = (argc == 2 && argv[1][0] == '-');

        // allocate memory on the GPU

        // distribution functions
        double *f, *feq, *f_new;
        cudaMalloc((void **)&f,N*N*Q*sizeof(double));
        cudaMalloc((void **)&feq,N*N*Q*sizeof(double));
        cudaMalloc((void **)&f_new,N*N*Q*sizeof(double));

        // density and velocity
        double *rho, *ux, *uy;
        cudaMalloc((void **)&rho,N*N*sizeof(double));
        cudaMalloc((void **)&ux,N*N*sizeof(double));
        cudaMalloc((void **)&uy,N*N*sizeof(double));

        // rate-of-strain
        double *sigma;
        cudaMalloc((void **)&sigma,N*N*sizeof(double));

        // allocate space for D3Q9 parameters on the host
        double *ex_h    = new double[Q];
        double *ey_h    = new double[Q];
        int *oppos_h = new int[Q];
        double *wt_h = new double[Q];

        // fill D3Q9 parameters in constant memory on the GPU
        D3Q9(ex_h, ey_h, oppos_h, wt_h);

        // assign a 2D distribution of CUDA "threads" within each CUDA "block"    
        int threadsAlongX=16, threadsAlongY=16;
        dim3 dimBlock(threadsAlongX, threadsAlongY, 1);

        // calculate number of blocks along X and Y in a 2D CUDA "grid"
        dim3 dimGrid( ceil(float(N)/float(dimBlock.x)), ceil(float(N)/float(dimBlock.y)), 1 );

        // launch GPU kernel to initialize all fields
        initialize<<<dimGrid,dimBlock>>>(N, Q, DENSITY, LID_VELOCITY,
                                         rho, ux, uy, sigma,
                                         f, feq, f_new);

        // time integration
        int time=0;
        while(time<TIME_STEPS) {

            time++;

            std::cout << "Time = " << time << std::endl;

            collideAndStream<<<dimGrid,dimBlock >>>(N, Q, DENSITY, LID_VELOCITY, REYNOLDS_NUMBER,
                                                    rho, ux, uy, sigma,
                                                    f, feq, f_new);

            // collideAndStream and everythingElse were originally one kernel
            // they were separated out to make all threads synchronize globally
            // before moving on to the next set of calculations

            everythingElse<<<dimGrid,dimBlock >>>(N, Q, DENSITY, LID_VELOCITY, REYNOLDS_NUMBER,
                                                  rho, ux, uy, sigma,
                                                  f, feq, f_new);

            // this is where ArrayFire is currently used
            // the cool thing is you don't need to move the GPU arrays back to the
            // CPU for visualizing them. And of course, we have in-situ graphics

    //      double curl_min = 0, curl_max = 0;

            if (time % 10 == 0) {
                if(!isconsole) {
                    array U(N,N,ux,afDevice);
                    array V(N,N,uy,afDevice);
                    array umag = pow(U*U + V*V, 0.5);

//                  array dUdx,dUdy,dVdx,dVdy;
//                  grad(dUdx,dUdy,U);
//                  grad(dVdx,dVdy,V);
//                  array curl = dVdx - dUdy;

//                  double2 extrema = minmax<double2>(curl);
//                  std::cout << "Curl --- min " << extrema.x << "  max " << extrema.y << std::endl;

//                  if (extrema.x < curl_min) curl_min = extrema.x;
//                  if (extrema.y > curl_max) curl_max = extrema.y;

//                  curl(0) = -0.1;
//                  curl(N) = +0.1;
                    fig("color","heat");
                    image(umag);
                }
            }
    
        }

    } catch (af::exception& e) {
        fprintf(stderr, "%s\n", e.what());
        throw;
    }

    return 0;
}
